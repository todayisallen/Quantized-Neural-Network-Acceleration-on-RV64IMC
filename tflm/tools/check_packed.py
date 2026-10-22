#!/usr/bin/env python3
"""
check_packed.py — 查每個 convolution 層能不能餵給 8 寬的 packed MAC 指令。

用法:
    python3 tools/check_packed.py <model>.tflite [--width 8]

====================================================================
這個工具回答什麼
====================================================================
xqmac8 一次吃兩個 64 位元暫存器（各 8 個 int8），所以 1x1 conv 的
歸約維度 K（= input channel 數）必須能切成 8 的倍數。

K 不是 8 的倍數時有兩種處理：
  (a) 整層退回純量  —— 正確但浪費，K=20 連能用的 16 個都放棄
  (b) 切 K/8 塊 + 尾巴 —— K=20 用 2 條 xqmac8 加 4 次純量

本工具算出：有幾層不是 8 的倍數、那些層占多少 MAC，
也就是 (a) 與 (b) 的差距到底值不值得寫。

**不回答**位址對齊。tensor 在 arena 裡的位移由 TFLM 的 memory planner
在 AllocateTensors 時決定，靜態看不出來。那要在 runtime 量
（conv_xq.cc 的 Eval 裡報告 packed_ok 實際成立與否）。
不過有一個推論：若 K 是 8 的倍數、且該 tensor 的基底 8 對齊，
則所有 oc*K 與 (y*W+x)*K 的位移都自動 8 對齊 —— 所以 K 的條件
通常是主要限制，對齊是次要的。這個推論要被 runtime 的量測確認。

====================================================================
"""
import argparse
import sys

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
    except ImportError:
        sys.exit("找不到 ai_edge_litert。"
                 "  pip install --only-binary=:all: ai-edge-litert")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--width", type=int, default=8,
                    help="packed 指令的寬度（xqmac8 是 8）")
    # 預設值是 B0/B1/X1 三個配置實測到的 MultiplyByQuantizedMultiplier
    # 指令數（三者完全相同）。X2 做完後換成 X2 的量測值。
    ap.add_argument("--rq", type=int, default=246278038,
                    help="requantize 的實測指令數，用來算「條/輸出元素」")
    a = ap.parse_args()
    W = a.width

    interp = Interpreter(
        model_path=a.model,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    if not hasattr(interp, "_get_ops_details"):
        sys.exit("這個版本的 interpreter 沒有 _get_ops_details()")
    ops = interp._get_ops_details()
    td = {t["index"]: t for t in interp.get_tensor_details()}

    def shape(i):
        return [int(s) for s in td[i]["shape"]] if i in td else None

    rows = []
    for o in ops:
        name = o["op_name"]
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            continue
        ins, outs = list(o.get("inputs", [])), list(o.get("outputs", []))
        if len(ins) < 2 or not outs:
            continue
        fs, os_ = shape(ins[1]), shape(outs[0])
        if not fs or not os_ or len(fs) != 4 or len(os_) != 4:
            continue
        _, OH, OW, OC = os_
        if name == "CONV_2D":
            # filter = [OC, FH, FW, IC]
            FH, FW, K = fs[1], fs[2], fs[3]
            macs = OH * OW * OC * FH * FW * K
            red = K                      # 歸約維度（連續的那一維）
            kind = "1x1" if (FH == 1 and FW == 1) else f"{FH}x{FW}"
        else:
            # filter = [1, FH, FW, OC]；depthwise 的平行維度是 channel
            FH, FW = fs[1], fs[2]
            macs = OH * OW * OC * FH * FW
            red = OC                     # 分塊是對 channel 做的
            kind = f"dw {FH}x{FW}"
        rows.append(dict(idx=o["index"], op=name, kind=kind, OH=OH, OW=OW,
                         OC=OC, FH=FH, FW=FW, red=red, macs=macs,
                         # 輸出元素數。requantize 的呼叫次數由這個決定，
                         # 與 MAC 怎麼算無關 —— 這是規格書 §3 要求的
                         # 「instructions / output element」的分母。
                         outs=OH * OW * OC))

    if not rows:
        sys.exit("模型裡找不到 CONV_2D / DEPTHWISE_CONV_2D")

    # ---------------------------------------------------------------- 明細
    print("=" * 78)
    print(f"每層的歸約/分塊維度能否被 {W} 整除")
    print("=" * 78)
    print(f"  {'idx':>4} {'類型':<9} {'輸出':<16} {'維度':>6} {'%'+str(W):>4} "
          f"{'整塊':>6} {'尾巴':>5} {'MAC':>14}")
    for r in rows:
        q, rem = divmod(r["red"], W)
        flag = "" if rem == 0 else "  <-- 有尾巴"
        out = "{}x{}x{}".format(r["OH"], r["OW"], r["OC"])
        print(f"  {r['idx']:>4} {r['kind']:<9} {out:<16} "
              f"{r['red']:>6} {rem:>4} {q:>6} {rem:>5} {r['macs']:>14,}{flag}")

    # ---------------------------------------------------------------- 彙總
    def summarize(title, sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return
        tot = sum(r["macs"] for r in sub)
        ok = [r for r in sub if r["red"] % W == 0]
        bad = [r for r in sub if r["red"] % W != 0]
        ok_macs = sum(r["macs"] for r in ok)
        # 方案 (b)：按比例估可被 packed 覆蓋的 MAC
        #   整塊部分 = floor(red/W)*W / red
        cov = sum(r["macs"] * ((r["red"] // W) * W) / r["red"] for r in sub)
        print()
        print("-" * 78)
        print(f"{title}：{len(sub)} 層，{tot:,} MAC")
        print(f"  維度是 {W} 的倍數的層: {len(ok)}/{len(sub)}")
        print(f"  這些層占 MAC          : {ok_macs:,} ({100*ok_macs/tot:.2f}%)")
        if bad:
            names = ", ".join("idx{}(維度{})".format(r["idx"], r["red"])
                              for r in bad)
            print(f"  不是倍數的層          : {names}")
            print(f"  方案(a) 整層退回純量  : 可 packed 的 MAC = "
                  f"{ok_macs:,} ({100*ok_macs/tot:.2f}%)")
            print(f"  方案(b) 切塊加尾巴    : 可 packed 的 MAC = "
                  f"{cov:,.0f} ({100*cov/tot:.2f}%)")
            print(f"  -> 寫尾巴迴圈多換到    {100*(cov-ok_macs)/tot:.2f}% 的 MAC")
        else:
            print(f"  全部都是 {W} 的倍數 —— 方案(a) 與 (b) 在這個模型上等價。")
            print(f"     尾巴迴圈仍應實作（不依賴這個巧合），"
                  f"但驗證測試必須自己造非倍數的案例，")
            print(f"     因為真實模型走不到那條路徑。")

    summarize("1x1 CONV_2D（xqmac8 的目標）",
              lambda r: r["op"] == "CONV_2D" and r["kind"] == "1x1")
    summarize("非 1x1 CONV_2D（目前走參考實作）",
              lambda r: r["op"] == "CONV_2D" and r["kind"] != "1x1")
    summarize("DEPTHWISE_CONV_2D（DW_XQ_BLK 的目標）",
              lambda r: r["op"] == "DEPTHWISE_CONV_2D")

    # ---------------------------------------------------------------- 全局
    allm = sum(r["macs"] for r in rows)
    o1 = sum(r["macs"] for r in rows
             if r["op"] == "CONV_2D" and r["kind"] == "1x1")
    print()
    print("=" * 78)
    print(f"卷積類 MAC 合計 {allm:,}")
    print(f"  其中 1x1 conv  {o1:,} ({100*o1/allm:.2f}%)")
    print()
    print("位址對齊未在此檢查 —— 由 TFLM 的 memory planner 在 runtime 決定。")
    print("用 conv_xq.cc 的 XQ_REPORT 診斷輸出確認 packed_ok 實際成立與否。")

    # ------------------------------------------------- 輸出元素數（X2 的分母）
    #
    # 為什麼要單獨算這個：規格書 §3 對 requantization 指令要求的指標是
    # 「Instructions / output element」，不是 instructions / MAC。這兩個
    # 分母在這個模型上差了兩個數量級，而 requantize 的成本只跟前者成正比
    # —— 這正是它的占比會隨著 MAC 被優化而上升的原因（1.26% → 15.5%）。
    #
    # 實測佐證這個分母是對的：MultiplyByQuantizedMultiplier 在 B0/B1/X1
    # 三個配置的指令數完全相同（246,278,038），而 per-operator 的歸屬拆分
    # 顯示它幾乎全部來自 CONV_2D 與 DEPTHWISE_CONV_2D
    # （162,170,300 + 84,107,503 = 246,277,803，差 235 條）。
    oe_conv = sum(r["outs"] for r in rows if r["op"] == "CONV_2D")
    oe_dw = sum(r["outs"] for r in rows if r["op"] == "DEPTHWISE_CONV_2D")
    oe = oe_conv + oe_dw
    print()
    print("=" * 78)
    print("輸出元素數（requantize 的呼叫次數 = 規格書 §3 的分母）")
    print("=" * 78)
    print(f"  CONV_2D            {oe_conv:>12,}")
    print(f"  DEPTHWISE_CONV_2D  {oe_dw:>12,}")
    print(f"  合計               {oe:>12,}")
    print(f"  每個輸出元素的 MAC {allm / oe:>12.1f}")
    print()
    print(f"  以 --rq 給的 requantize 指令數 {a.rq:,} 條計算：")
    print(f"    {a.rq / oe:.2f} 條指令 / 輸出元素")
    print("  （分子來自 scripts/instr_mix.sh 的 PC histogram，不是本工具算的。")
    print("   X2 做完後用 --rq 換成 X2 的量測值再跑一次，就是 §3 要的對照）")


if __name__ == "__main__":
    main()