#!/usr/bin/env python3
"""
explore_cnn.py — EfficientNet-Lite0 (uint8 全整數) 的結構探勘。

一次回答 Phase 1 需要知道的所有事：

  1. 每個 CONV_2D / DEPTHWISE_CONV_2D / FULLY_CONNECTED 的形狀與 MAC 數
  2. MAC 分布 —— 決定哪幾層當 microbenchmark 與 major operators
  3. 權重是 per-channel 對稱 (z_w = 0) 還是 per-tensor 非對稱
     ★ 這一項直接決定 xqmac8 的指令語意
  4. 1x1 conv 的佔比 —— 1x1 本質是 GEMM，直接對應 packed dot product
  5. 每層的量化參數，以及 bias scale 是否等於 s_in * s_w

用法:
    python3 explore_cnn.py <model.tflite>
"""

import os
import sys
import numpy as np

from ai_edge_litert.interpreter import Interpreter
try:
    from ai_edge_litert.interpreter import OpResolverType
except ImportError:
    OpResolverType = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmath import quantize_multiplier

CONV_OPS = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")


def build_interpreter(model, preserve=True):
    """關掉預設 delegate，確保走 TFLite 的定點參考路徑而非 XNNPACK 的 FP32。"""
    kw = {"model_path": model}
    if preserve:
        kw["experimental_preserve_all_tensors"] = True
    if OpResolverType is not None:
        for attr in ("BUILTIN_WITHOUT_DEFAULT_DELEGATES", "BUILTIN_REF", "BUILTIN"):
            if hasattr(OpResolverType, attr):
                kw["experimental_op_resolver_type"] = getattr(OpResolverType, attr)
                break
    try:
        return Interpreter(**kw)
    except TypeError:
        return Interpreter(model_path=model)


def q(det, idx):
    """取出某 tensor 的量化參數摘要。"""
    d = det[idx]
    qp = d["quantization_parameters"]
    return {
        "shape": tuple(int(x) for x in d["shape"]),
        "dtype": d["dtype"].__name__,
        "scales": np.asarray(qp["scales"], dtype=np.float64),
        "zps": np.asarray(qp["zero_points"], dtype=np.int64),
        "qdim": qp.get("quantized_dimension", 0),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model = sys.argv[1]

    it = build_interpreter(model)
    it.allocate_tensors()
    det = {t["index"]: t for t in it.get_tensor_details()}

    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    print("model : %s" % model)
    print("input : %s %s" % (tuple(int(x) for x in inp["shape"]), inp["dtype"].__name__))
    print("output: %s %s" % (tuple(int(x) for x in out["shape"]), out["dtype"].__name__))

    layers = []
    for op in it._get_ops_details():
        if op["op_name"] not in CONV_OPS:
            continue
        ins = list(op["inputs"])
        outs = list(op["outputs"])
        if len(ins) < 2 or not outs:
            continue
        i_in, i_w = ins[0], ins[1]
        i_b = ins[2] if len(ins) > 2 and ins[2] >= 0 else None
        i_out = outs[0]
        if i_in not in det or i_w not in det or i_out not in det:
            continue

        A, W, O = q(det, i_in), q(det, i_w), q(det, i_out)
        B = q(det, i_b) if i_b is not None else None

        # MAC 數
        name = op["op_name"]
        if name == "CONV_2D":
            # W: (out_c, kh, kw, in_c)   O: (1, oh, ow, out_c)
            oc, kh, kw, ic = W["shape"]
            _, oh, ow, _ = O["shape"]
            macs = oh * ow * oc * kh * kw * ic
            kernel = "%dx%d" % (kh, kw)
            K = kh * kw * ic                  # 等效 GEMM 的收縮維度
        elif name == "DEPTHWISE_CONV_2D":
            # W: (1, kh, kw, ch)
            _, kh, kw, ch = W["shape"]
            _, oh, ow, _ = O["shape"]
            macs = oh * ow * ch * kh * kw
            kernel = "%dx%ddw" % (kh, kw)
            K = kh * kw
        else:  # FULLY_CONNECTED   W: (out, in)
            n_out, n_in = W["shape"]
            macs = n_out * n_in
            kernel = "fc"
            K = n_in

        layers.append(dict(op=name, idx=op["index"], kernel=kernel, macs=macs, K=K,
                           i_in=i_in, i_w=i_w, i_b=i_b, i_out=i_out,
                           A=A, W=W, B=B, O=O))

    total = sum(l["macs"] for l in layers)
    print("\n共 %d 個計算層，總計 %.3f GMAC" % (len(layers), total / 1e9))

    # ---- 權重量化形態（決定 xqmac8 語意）----
    print("\n=== 權重量化形態 ===")
    sym = asym = 0
    perchan = pertensor = 0
    for l in layers:
        if np.all(l["W"]["zps"] == 0):
            sym += 1
        else:
            asym += 1
        if l["W"]["scales"].size > 1:
            perchan += 1
        else:
            pertensor += 1
    print("  權重 zero-point 全為 0 (對稱): %d / %d 層" % (sym, len(layers)))
    print("  per-channel scale: %d 層,  per-tensor scale: %d 層"
          % (perchan, pertensor))
    if sym == len(layers):
        print("  => xqmac8 只需純有號點積，zero-point 修正可離線預算 sum_w")
    else:
        print("  => 權重有非零 zero-point，點積展開會多出 z_w * sum_a 一項，")
        print("     sum_a 需在 runtime 計算 —— 這會影響 xqmac8 的設計")

    # ---- 逐層明細 ----
    print("\n=== 逐層明細（依 MAC 由大到小）===")
    print("%-18s %-7s %9s %6s %6s %-16s %-16s %s"
          % ("op", "kern", "MMAC", "%", "K", "in(s,z)", "out(s,z)", "W shape"))
    for l in sorted(layers, key=lambda x: -x["macs"])[:25]:
        A, W, O = l["A"], l["W"], l["O"]
        print("%-18s %-7s %9.2f %5.1f%% %6d %-16s %-16s %s"
              % (l["op"][:18], l["kernel"], l["macs"] / 1e6,
                 100.0 * l["macs"] / total, l["K"],
                 "%.4g,%d" % (A["scales"][0], A["zps"][0]),
                 "%.4g,%d" % (O["scales"][0], O["zps"][0]),
                 W["shape"]))

    # ---- 依 kernel 形狀彙總：1x1 佔比決定 xqmac8 的涵蓋率 ----
    print("\n=== 依 kernel 形狀彙總 ===")
    agg = {}
    for l in layers:
        a = agg.setdefault(l["kernel"], [0, 0])
        a[0] += 1
        a[1] += l["macs"]
    for k in sorted(agg, key=lambda x: -agg[x][1]):
        n, m = agg[k]
        print("  %-8s %3d 層  %9.2f MMAC  %5.1f%%" % (k, n, m / 1e6, 100.0 * m / total))
    gemm_like = sum(m for k, (n, m) in agg.items() if k in ("1x1", "fc"))
    print("  --> 1x1 + fc（可直接對應 packed dot product）: %.1f%%"
          % (100.0 * gemm_like / total))

    # ---- K 是否為 8 的倍數（xqmac8 需不需要處理尾數）----
    print("\n=== K 對 8 的整除性 ===")
    bad = [(l["kernel"], l["K"], l["macs"]) for l in layers if l["K"] % 8]
    okm = sum(l["macs"] for l in layers if l["K"] % 8 == 0)
    print("  K 可被 8 整除的層佔 %.1f%% 的 MAC" % (100.0 * okm / total))
    if bad:
        print("  不可整除的 K: %s" % sorted({(k, K) for k, K, _ in bad}))

    # ---- bias scale 慣例檢查 ----
    print("\n=== bias scale == s_in * s_w ? ===")
    nb = ok = 0
    for l in layers:
        if l["B"] is None or l["B"]["scales"].size == 0:
            continue
        nb += 1
        expect = l["A"]["scales"][0] * l["W"]["scales"][0]
        got = l["B"]["scales"][0]
        if abs(got - expect) / max(expect, 1e-30) < 1e-5:
            ok += 1
    print("  %d / %d 層符合（符合則 bias 可直接累加進 int32 accumulator）" % (ok, nb))

    # ---- requantize 乘數的範圍 ----
    print("\n=== requantize multiplier / shift ===")
    shifts = []
    for l in layers:
        s_in = l["A"]["scales"][0]
        s_out = l["O"]["scales"][0]
        for sw in np.atleast_1d(l["W"]["scales"]):
            M = float(s_in * sw / s_out)
            if M <= 0:
                continue
            _, sh = quantize_multiplier(M)
            shifts.append(sh)
    if shifts:
        shifts = np.array(shifts)
        print("  shift 範圍 [%d, %d]，total = 31-shift 落在 [%d, %d]"
              % (shifts.min(), shifts.max(), 31 - shifts.max(), 31 - shifts.min()))

    # ---- 建議的 benchmark 選擇 ----
    print("\n=== 建議的 benchmark ===")
    g = [l for l in layers if l["kernel"] in ("1x1", "fc")]
    g.sort(key=lambda x: -x["macs"])
    for tag, l in zip(("microbenchmark", "major operator 1", "major operator 2"), g[:3]):
        print("  %-18s op#%-4d %-18s K=%-5d %.2f MMAC  (in=%d w=%d out=%d)"
              % (tag, l["idx"], l["kernel"], l["K"], l["macs"] / 1e6,
                 l["i_in"], l["i_w"], l["i_out"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
