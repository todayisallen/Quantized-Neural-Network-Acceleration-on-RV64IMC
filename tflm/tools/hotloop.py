#!/usr/bin/env python3
"""
hotloop.py — 找出某個函式真正的熱點迴圈，並給出它的**動態**指令組成。

主要用法（有 histogram，建議）:
    python3 tools/hotloop.py --hist results/B1.hist \\
        --objdump results/B1.dis --func XqConvEval

退路（沒有 histogram，只有反組譯）:
    riscv64-unknown-elf-objdump -d --no-show-raw-insn nn_bare \\
      | python3 tools/hotloop.py --func XqConvEval

====================================================================
為什麼用 histogram，而不是從反組譯推迴圈結構
====================================================================
這個檔案前三版都在試著從靜態的反組譯「推論」哪一段是熱點迴圈，
三次都錯：

  v1  從函式開頭印 N 行 -> 印到的全是取張量的 boilerplate。
  v2  滑動視窗找 mul 最密的一段 -> 選到空間迴圈的 setup 碼，
      那裡有一串索引用的乘除法，密度比真正的 MAC 迴圈還高。
  v3  加「迴圈深度」過濾（把往後跳的分支當 back edge）->
      實測在一個只有 4 層迴圈的函式上報出「最深巢狀 57 層」。
      因為 GCC 會重排基本區塊，往後跳的分支大多不是 back edge：
      冷路徑跳轉、tail-merge 的共用區塊、跳到被排到前面的守衛區塊，
      全都往後跳。89 個區間互相重疊就疊出 57 層。
      要做對需要 CFG + 支配關係分析。

Spike 的 -g 直接給出每個 PC 執行了幾次。最熱的 PC **定義上**就是
最內層迴圈 —— 這是量到的，不是推的，沒有東西可以搞錯。
而且順便得到熱點迴圈的**動態**指令組成（每類指令實際執行了幾次），
那才是算「指令/MAC」要用的數字；靜態計數只能告訴你碼長什麼樣。

histogram 怎麼產生：bash scripts/instr_mix.sh <CONFIG>
（它會把 spike -g 的 stderr 存成 results/<CONFIG>.hist，
  反組譯存成 results/<CONFIG>.dis）

====================================================================
判讀
====================================================================
累加器溢出的特徵是對 (sp) 的 **lw/sw**（32 位元堆疊往返）。
ld/sd 到 (sp) 是指標與 callee-save，任何函式都有，所以分開數。

load 的次數應該是 mul 的兩倍（input 與 weight 各一個）。
X1 之後應該看到 custom（xqmac8）取代 mul，而 load 從 lb 變成 ld，
次數掉到每個 custom 兩個 —— 用 --key custom 確認它真的在熱點裡。
"""
import argparse
import re
import sys
from collections import Counter

# objdump -d --no-show-raw-insn:   80001e0e:   auipc   a2,0x574
# 帶 raw insn:                     80001e0e:   00574617    auipc  a2,0x574
# 助記符第一個字元要容許 '.' —— objdump 不認得的自訂指令會印成
# ".insn 4, 0x..." 或 ".4byte 0x..."。只收 [a-z] 的話整行會被丟掉，
# 於是 xqmac8 從視窗裡消失、迴圈看起來少一條指令（實測踩過）。
INSN = re.compile(
    r"^\s*([0-9a-f]+):\s+(?:([0-9a-f]{4,8})\s+)?([.a-z][a-z0-9._]*)\s*(.*)$")
FUNC = re.compile(r"^[0-9a-f]+\s+<([^>]+)>:\s*$")

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"}
STORE = {"sb", "sh", "sw", "sd"}
MUL = {"mul", "mulh", "mulhu", "mulhsu", "mulw"}
# div 跟 mul 分開：setup 碼有一串索引用的乘除法，
# 混在一起數會讓 setup 冒充 MAC 迴圈。
DIV = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
          "blez", "bgez", "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu"}
JUMP = {"jal", "jalr", "j", "jr", "ret", "call", "tail"}

CLS = ("load", "store", "mul", "div", "alu", "branch", "jump", "custom")
FLAG = {"load": "L", "store": "S", "mul": "M", "div": "D", "custom": "X"}


def strip_c(m):
    return m[2:] if m.startswith("c.") else m


def classify(mnem, enc):
    m = strip_c(mnem)
    if enc is not None:
        try:
            if (int(enc, 16) & 0x7F) in (0x0B, 0x2B):
                return "custom"
        except ValueError:
            pass
    if m in ("unknown", ".insn", ".word", ".4byte", ".2byte"):
        return "custom"
    if m in LOAD:
        return "load"
    if m in STORE:
        return "store"
    if m in MUL:
        return "mul"
    if m in DIV:
        return "div"
    if m in BRANCH:
        return "branch"
    if m in JUMP:
        return "jump"
    return "alu"


def parse_hist(path):
    """{pc: count}。Spike -g 印到 stderr，格式是 '<hexpc> <count>'。"""
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            if "PC Histogram" in line:
                continue
            p = line.split()
            if len(p) != 2:
                continue
            try:
                out[int(p[0], 16)] = out.get(int(p[0], 16), 0) + int(p[1])
            except ValueError:
                continue
    if not out:
        sys.exit(f"{path} 裡沒有可解析的 histogram。\n"
                 "確認是 spike -g 的 stderr（scripts/instr_mix.sh 會存好）。")
    return out


def collect(src, want):
    """回傳 (函式名, [(addr, mnem, ops, cls)], 其他候選函式名)。"""
    cur, chosen, insns, cand = None, None, [], []
    for line in src:
        f = FUNC.match(line)
        if f:
            if chosen is not None:
                break
            cur = f.group(1)
            if want in cur:
                chosen, insns = cur, []
            else:
                cand.append(cur)
            continue
        if chosen is None:
            continue
        m = INSN.match(line)
        if m:
            insns.append((int(m.group(1), 16), m.group(3),
                          m.group(4).strip(),
                          classify(m.group(3), m.group(2))))
    return chosen, insns, cand


def best_window(weight, W):
    """回傳總權重最大的視窗起點。"""
    run = sum(weight[:W])
    best, bi = run, 0
    for i in range(1, len(weight) - W + 1):
        run += weight[i + W - 1] - weight[i - 1]
        if run > best:
            best, bi = run, i
    return bi, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True,
                    help="函式名的一部分（子字串比對，不必給完整 mangled name）")
    ap.add_argument("--hist", help="spike -g 的 histogram（強烈建議）")
    ap.add_argument("--objdump", help="objdump 的輸出檔；省略則讀 stdin")
    ap.add_argument("--key", default="mul",
                    help="沒有 histogram 時，要找最密的哪一類："
                         "mul（預設）/ custom / load / store")
    ap.add_argument("--window", type=int, default=56, help="視窗大小（指令數）")
    ap.add_argument("--pad", type=int, default=4, help="視窗前後各多印幾條")
    ap.add_argument("--top", type=int, default=8, help="另外列出最熱的幾個 PC")
    a = ap.parse_args()

    src = open(a.objdump, errors="replace") if a.objdump else sys.stdin
    chosen, insns, cand = collect(src, a.func)

    if chosen is None:
        print(f"找不到名字裡含 '{a.func}' 的函式。候選：")
        hit = [c for c in cand if re.search(r"Conv|conv|Eval|[Mm]ac", c)]
        for c in (hit or cand)[:25]:
            print("  " + c)
        return 1
    if not insns:
        print(f"{chosen} 裡沒有指令？")
        return 1

    hist = parse_hist(a.hist) if a.hist else None
    static = Counter(c for _, _, _, c in insns)
    print(f"=== {chosen} ===")
    print(f"靜態 {len(insns)} 條指令：" +
          "  ".join(f"{k} {static[k]}" for k in CLS if static[k]))

    # ------------------------------------------------------------------
    if hist is None:
        print("\n!! 沒有給 --hist，只能用靜態的 '{}' 密度猜熱點。"
              .format(a.key))
        print("   這個模式會選到 setup 碼（實測踩過），結論不要當真。")
        print("   先跑 bash scripts/instr_mix.sh <CONFIG> 產生 histogram。")
        weight = [1 if c == a.key else 0 for _, _, _, c in insns]
        label = f"'{a.key}' 最密"
        counts = None
    else:
        counts = [hist.get(ad, 0) for ad, _, _, _ in insns]
        tot = sum(counts)
        if tot == 0:
            print(f"\n{chosen} 的指令在 histogram 裡執行次數全是 0 —— "
                  f"這個函式沒被執行到，或是位址對不上"
                  f"（.dis 與 .hist 必須來自**同一個** binary）。")
            return 1
        print(f"動態：這個函式總共執行 {tot:,} 條指令")
        dyn = Counter()
        for (_, _, _, c), n in zip(insns, counts):
            dyn[c] += n
        print("      " + "  ".join(
            f"{k} {dyn[k]:,} ({100.0*dyn[k]/tot:.1f}%)"
            for k in CLS if dyn[k]))
        weight = counts
        label = "執行次數最高"

    W = min(a.window, len(insns))
    bi, _ = best_window(weight, W)
    lo, hi = max(0, bi - a.pad), min(len(insns), bi + W + a.pad)
    win = insns[bi:bi + W]
    wt = Counter(c for _, _, _, c in win)

    # ------------------------------------------------------------------ 列印
    if counts is not None:
        wc = counts[bi:bi + W]
        wdyn = Counter()
        for (_, _, _, c), n in zip(win, wc):
            wdyn[c] += n
        wtot = sum(wc)
        print(f"\n{label}的 {W} 條指令視窗（第 {bi} 條起），"
              f"動態 {wtot:,} 條，占這個函式 {100.0*wtot/tot:.1f}%：")
        print("      " + "  ".join(f"{k} {wdyn[k]:,}" for k in CLS if wdyn[k]))
    else:
        print(f"\n{label}的 {W} 條指令視窗（第 {bi} 條起），靜態組成："
              + "  ".join(f"{k} {wt[k]}" for k in CLS if wt[k]))
        wdyn = None

    print("-" * 68)
    for i in range(lo, hi):
        ad, mnem, ops, cls = insns[i]
        mark = " " if (bi <= i < bi + W) else "."
        n = f"{counts[i]:>12,}" if counts else " " * 12
        print(f"{mark}{n}  {ad:08x}: {FLAG.get(cls,' ')} {mnem:<10} {ops}")
    print("-" * 68)
    print("（欄位：視窗標記 / 執行次數 / 位址 / 類別 / 指令）")

    if counts:
        print(f"\n最熱的 {a.top} 個 PC：")
        order = sorted(range(len(insns)), key=lambda i: -counts[i])[:a.top]
        for i in order:
            ad, mnem, ops, cls = insns[i]
            print(f"  {counts[i]:>14,}  {ad:08x}: "
                  f"{FLAG.get(cls,' ')} {mnem:<10} {ops}")

    # ------------------------------------------------------------------ 判讀
    base = wdyn if wdyn else wt
    nmul = base["mul"] + base["custom"]
    if not nmul:
        print("\n視窗裡沒有 mul 或 custom —— 選到的可能不是 MAC 迴圈。")
        return 0

    def sp(mnems):
        if wdyn:
            return sum(n for (_, m, o, _), n in zip(win, counts[bi:bi + W])
                       if strip_c(m) in mnems and "(sp)" in o)
        return sum(1 for _, m, o, _ in win
                   if strip_c(m) in mnems and "(sp)" in o)

    w32, w64 = sp(("lw", "sw")), sp(("ld", "sd"))
    unit = "執行次數" if wdyn else "靜態條數"
    print(f"\n每個 mul/custom 平均配（{unit}）: "
          f"load {base['load']/nmul:.2f}  store {base['store']/nmul:.2f}  "
          f"alu {base['alu']/nmul:.2f}")
    print(f"對 (sp) 的存取: 32 位元 (lw/sw) {w32:,}  64 位元 (ld/sd) {w64:,}")

    r = w32 / nmul
    if r >= 0.8:
        print("** 每個 mul 都配著 32 位元堆疊往返 —— 累加器溢出了。")
        if "Dw" in chosen or "Depthwise" in chosen:
            print("   調小 DW_XQ_BLK（make CONFIG=B1 BLK=4 ...）再看一次。")
        else:
            print("   這個 kernel 沒有分塊寬度的旋鈕，要看是哪些變數被擠出去。")
    elif r <= 0.15:
        print("** 幾乎沒有 32 位元堆疊往返 —— 累加器留在暫存器裡。")
        if base["load"] / nmul > 2.4:
            print("   但 load 超過每個 mul 兩個，可能有多餘的載入。")
    else:
        print(f"** 32 位元堆疊往返是每個 mul {r:.2f} 個 —— 介於兩者之間，"
              f"自己看上面的碼。")
    return 0


if __name__ == "__main__":
    sys.exit(main())