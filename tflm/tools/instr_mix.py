#!/usr/bin/env python3
"""
instr_mix.py — 從 Spike 的 PC histogram 算出指令組成。

用法:
    python3 instr_mix.py --hist run.hist --objdump nn_bare.dis [--nm nn_bare.nm]
                         [--macs 385187552] [--top 15]

為什麼用 PC histogram 而不是 commit log:
    B0 的 Invoke 有 19.58 B 條指令，--log-commits 的 trace 會是 TB 級。
    PC histogram 只記錄「每個 PC 執行了幾次」，檔案大小跟程式碼量成正比
    （數萬筆），跟執行次數無關。再用 objdump 把 PC 對回指令，
    得到的組成是**精確的**，不是取樣估計。

    代價是 histogram 涵蓋整個程式（含 AllocateTensors 與啟動碼），
    不只 Invoke。但 B0 實測 Invoke 占 99.92%，污染可以忽略 ——
    報告裡要註明這一點。

--nm 可選：給 nm 的輸出就能把 PC 歸屬到函式，得到**每個 kernel 的**
指令組成，而不只是整體。規格書 §2.B 要求「主要 operator 的執行貢獻」，
這個比整體數字有用得多。
"""
import argparse
import re
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# 指令分類
#
# 刻意不寫成「其他」一把抓：無法分類的指令會單獨列出來，
# 如果那一欄不是趨近 0，就代表這張表漏了東西，要補。
# 靜默歸到「其他」會讓錯誤看不見。
# ----------------------------------------------------------------------
LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "lr.w", "lr.d"}
STORE = {"sb", "sh", "sw", "sd", "sc.w", "sc.d"}
MULDIV = {"mul", "mulh", "mulhu", "mulhsu", "mulw",
          "div", "divu", "divw", "divuw",
          "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu",
          "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble",
          "bgtu", "bleu"}
JUMP = {"jal", "jalr", "j", "jr", "ret", "call", "tail"}
CSR = {"csrr", "csrw", "csrs", "csrc", "csrrw", "csrrs", "csrrc",
       "csrrwi", "csrrsi", "csrrci", "csrwi", "csrsi", "csrci",
       "rdcycle", "rdinstret", "rdtime", "rdcycleh", "rdinstreth"}
SYSTEM = {"ecall", "ebreak", "fence", "fence.i", "fence.tso", "wfi",
          "mret", "sret", "uret", "sfence.vma"}
ALU = {
    "add", "addi", "addw", "addiw", "sub", "subw", "neg", "negw",
    "sll", "slli", "sllw", "slliw", "srl", "srli", "srlw", "srliw",
    "sra", "srai", "sraw", "sraiw",
    "and", "andi", "or", "ori", "xor", "xori", "not",
    "slt", "slti", "sltu", "sltiu", "seqz", "snez", "sltz", "sgtz",
    "lui", "auipc", "li", "mv", "sext.w", "zext.w", "nop",
}

CLASSES = ["load", "store", "mul/div", "alu", "branch", "jump",
           "csr", "system", "custom", "未分類"]


def classify(mnem, encoding):
    """回傳指令類別。encoding 是 objdump 印的原始十六進位（可能為 None）。"""
    # 壓縮指令的 c. 前綴去掉再分類，但會另外統計壓縮比例
    m = mnem[2:] if mnem.startswith("c.") else mnem
    # objdump 對未知的自訂指令會印 .insn / unknown / 直接印數值
    if m in LOAD:
        return "load"
    if m in STORE:
        return "store"
    if m in MULDIV:
        return "mul/div"
    if m in BRANCH:
        return "branch"
    if m in JUMP:
        return "jump"
    if m in CSR:
        return "csr"
    if m in SYSTEM:
        return "system"
    if m in ALU:
        return "alu"
    # 自訂指令：從 opcode 認。custom-0 = 0x0b, custom-1 = 0x2b
    if encoding is not None and (encoding & 0x7f) in (0x0b, 0x2b):
        return "custom"
    if m in ("unknown", ".insn", ".word", ".2byte", ".4byte"):
        return "custom"
    return "未分類"


def parse_objdump(path):
    """回傳 {pc: (mnemonic, encoding, 是否壓縮)}。"""
    # 格式:    80000000:\t00101117          \tauipc\tsp,0x101
    pat = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)")
    out = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            hexs = m.group(2)
            mnem = m.group(3)
            try:
                enc = int(hexs, 16)
            except ValueError:
                enc = None
            out[pc] = (mnem, enc, len(hexs) <= 4)
    return out


def parse_hist(path):
    """回傳 {pc: count}。Spike 印到 stderr，格式是 '<hexpc> <count>'。"""
    out = {}
    started = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "PC Histogram size" in line:
                started = True
                continue
            p = line.split()
            if len(p) != 2:
                continue
            try:
                pc = int(p[0], 16)
                n = int(p[1])
            except ValueError:
                continue
            out[pc] = out.get(pc, 0) + n
    if not out and not started:
        sys.exit("histogram 檔裡沒有可解析的資料。\n"
                 "確認 spike 有 -g，而且 stderr 有被導到這個檔案。\n"
                 "舊版 Spike 可能要在 configure 時加 --enable-histogram。")
    return out


def parse_nm(path):
    """回傳依位址排序的 [(addr, name)]，用來把 PC 歸屬到函式。"""
    syms = []
    pat = re.compile(r"^([0-9a-f]+)\s+[tTwW]\s+(\S+)")
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m:
                syms.append((int(m.group(1), 16), m.group(2)))
    syms.sort()
    return syms


def func_of(syms, pc):
    """二分搜尋：pc 落在哪個函式。"""
    lo, hi = 0, len(syms) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if syms[mid][0] <= pc:
            best = syms[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best or "?"


def report(title, counts, total, extra=None):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)
    print(f"  {'類別':<10} {'指令數':>16} {'占比':>8}")
    for c in CLASSES:
        n = counts.get(c, 0)
        if n == 0 and c in ("system", "custom", "未分類"):
            continue
        pct = 100.0 * n / total if total else 0.0
        print(f"  {c:<10} {n:>16,} {pct:>7.2f}%")
    print(f"  {'合計':<10} {total:>16,}")
    if extra:
        for k, v in extra:
            print(f"  {k}  {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--objdump", required=True)
    ap.add_argument("--nm")
    ap.add_argument("--macs", type=int, default=385187552,
                    help="整個模型的 MAC 數，用來算 MAC/instruction")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    code = parse_objdump(a.objdump)
    hist = parse_hist(a.hist)
    syms = parse_nm(a.nm) if a.nm else None

    total = 0
    counts = Counter()
    mnem_counts = Counter()
    compressed = 0
    missing_pc = 0
    per_func = defaultdict(Counter)
    per_func_total = Counter()
    unknown_mnems = Counter()

    for pc, n in hist.items():
        total += n
        info = code.get(pc)
        if info is None:
            # PC 不在反組譯範圍內（啟動碼之外的東西，或 objdump 沒涵蓋的節）
            missing_pc += n
            counts["未分類"] += n
            continue
        mnem, enc, is_c = info
        cls = classify(mnem, enc)
        counts[cls] += n
        mnem_counts[mnem] += n
        if is_c:
            compressed += n
        if cls == "未分類":
            unknown_mnems[mnem] += n
        if syms is not None:
            fn = func_of(syms, pc)
            per_func[fn][cls] += n
            per_func_total[fn] += n

    extra = [
        ("壓縮指令 (RVC)   ", f"{compressed:,} ({100.0*compressed/total:.2f}%)"),
    ]
    if a.macs:
        lsu = counts.get("load", 0) + counts.get("store", 0)
        extra += [
            ("MAC / instruction", f"{a.macs/total:.4f}"),
            ("instruction / MAC", f"{total/a.macs:.2f}"),
            ("load+store / MAC ", f"{lsu/a.macs:.2f}"),
        ]
    if missing_pc:
        extra.append(("反組譯找不到的 PC",
                      f"{missing_pc:,} ({100.0*missing_pc/total:.3f}%)"))

    report("整體指令組成", counts, total, extra)

    if unknown_mnems:
        print()
        print("  !! 有未分類的指令，分類表要補：")
        for m, n in unknown_mnems.most_common(10):
            print(f"     {m:<16} {n:>14,}")

    print()
    print("=" * 68)
    print(f"指令使用頻率（前 {a.top} 名）")
    print("=" * 68)
    for m, n in mnem_counts.most_common(a.top):
        print(f"  {m:<16} {n:>16,}  {100.0*n/total:>6.2f}%")

    if syms is not None:
        print()
        print("=" * 68)
        print(f"各函式的指令組成（前 {a.top} 名）")
        print("=" * 68)
        hdr = f"  {'函式':<34}{'合計':>14} {'占比':>7}  " \
              f"{'load':>6} {'store':>6} {'mul':>6} {'alu':>6} {'br':>6} {'cust':>6}"
        print(hdr)
        for fn, n in per_func_total.most_common(a.top):
            c = per_func[fn]
            def pc_(k):
                return f"{100.0*c.get(k,0)/n:5.1f}%" if n else "    -"
            short = fn if len(fn) <= 33 else fn[:30] + "..."
            print(f"  {short:<34}{n:>14,} {100.0*n/total:>6.2f}%  "
                  f"{pc_('load'):>6} {pc_('store'):>6} {pc_('mul/div'):>6} "
                  f"{pc_('alu'):>6} {pc_('branch'):>6} {pc_('custom'):>6}")


if __name__ == "__main__":
    main()
