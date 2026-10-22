#!/usr/bin/env python3
"""
count_params.py — 統計 tflite 模型的參數量與 MAC 數，對照規格書的規模要求。

規格書第 1 節的硬性條件：
    參數量  2 – 12 M
    計算量  0.2 – 2 GMAC / inference

「參數」的定義有兩種，兩個都報：
  (a) 權重參數     只算 conv / fc 的 kernel，這是論文慣例
  (b) 全部常數張量 再加上 bias、BN 折疊後的常數等，等於模型檔的實際內容

用法:
    python3 count_params.py <model.tflite>
"""

import os
import sys
import numpy as np

from ai_edge_litert.interpreter import Interpreter
try:
    from ai_edge_litert.interpreter import OpResolverType
except ImportError:
    OpResolverType = None

WEIGHT_OPS = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
              "TRANSPOSE_CONV", "BATCH_MATMUL")


def build_interpreter(model):
    kw = {"model_path": model}
    if OpResolverType is not None:
        for attr in ("BUILTIN_WITHOUT_DEFAULT_DELEGATES", "BUILTIN_REF", "BUILTIN"):
            if hasattr(OpResolverType, attr):
                kw["experimental_op_resolver_type"] = getattr(OpResolverType, attr)
                break
    try:
        return Interpreter(**kw)
    except TypeError:
        return Interpreter(model_path=model)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model = sys.argv[1]

    it = build_interpreter(model)
    it.allocate_tensors()
    det = {t["index"]: t for t in it.get_tensor_details()}

    print("model: %s" % model)
    print("file : %.2f MB" % (os.path.getsize(model) / 1e6))

    # ---- 哪些 tensor 是常數（有實際資料的）----
    const = {}
    for i, d in det.items():
        try:
            arr = it.get_tensor(i)
        except Exception:
            continue                      # 中間張量，invoke 前沒有資料
        if arr is None or arr.size == 0:
            continue
        const[i] = arr

    # ---- 逐層統計 ----
    w_params = 0        # (a) 只算 kernel
    b_params = 0        # bias
    macs = 0
    rows = []
    seen_w = set()

    for op in it._get_ops_details():
        if op["op_name"] not in WEIGHT_OPS:
            continue
        ins, outs = list(op["inputs"]), list(op["outputs"])
        if len(ins) < 2 or not outs:
            continue
        i_w = ins[1]
        i_b = ins[2] if len(ins) > 2 and ins[2] >= 0 else None
        i_o = outs[0]
        if i_w not in det or i_o not in det:
            continue

        ws = tuple(int(x) for x in det[i_w]["shape"])
        osh = tuple(int(x) for x in det[i_o]["shape"])
        n_w = int(np.prod(ws)) if ws else 0
        n_b = int(np.prod([int(x) for x in det[i_b]["shape"]])) if i_b is not None else 0

        name = op["op_name"]
        if name == "CONV_2D" and len(ws) == 4 and len(osh) == 4:
            oc, kh, kw, ic = ws
            m = osh[1] * osh[2] * oc * kh * kw * ic
            tag = "%dx%d" % (kh, kw)
        elif name == "DEPTHWISE_CONV_2D" and len(ws) == 4 and len(osh) == 4:
            _, kh, kw, ch = ws
            m = osh[1] * osh[2] * ch * kh * kw
            tag = "%dx%ddw" % (kh, kw)
        elif name == "FULLY_CONNECTED" and len(ws) == 2:
            n_out, n_in = ws
            m = int(np.prod(osh[:-1])) * n_out * n_in
            tag = "fc"
        else:
            m = 0
            tag = name.lower()[:8]

        # 同一組權重若被多個 op 共用，參數只算一次
        if i_w not in seen_w:
            seen_w.add(i_w)
            w_params += n_w
            b_params += n_b
        macs += m
        rows.append((tag, ws, n_w, n_b, m))

    # ---- (b) 全部常數張量 ----
    all_const = sum(int(a.size) for a in const.values())
    const_bytes = sum(int(a.nbytes) for a in const.values())

    print("\n=== 參數量 ===")
    print("  (a) 權重 (conv/fc kernel)      %10s  = %.3f M" % (f"{w_params:,}", w_params / 1e6))
    print("      bias                       %10s  = %.3f M" % (f"{b_params:,}", b_params / 1e6))
    print("      權重 + bias                %10s  = %.3f M"
          % (f"{w_params + b_params:,}", (w_params + b_params) / 1e6))
    print("  (b) 全部常數張量               %10s  = %.3f M  (%.2f MB)"
          % (f"{all_const:,}", all_const / 1e6, const_bytes / 1e6))

    print("\n=== 計算量 ===")
    print("  %s MAC = %.4f GMAC" % (f"{macs:,}", macs / 1e9))

    print("\n=== 對照規格書（2–12 M 參數, 0.2–2 GMAC）===")
    p = (w_params + b_params) / 1e6
    g = macs / 1e9
    print("  參數 %.2f M   %s" % (p, "OK" if 2 <= p <= 12 else "超出範圍"))
    print("  計算 %.3f GMAC  %s" % (g, "OK" if 0.2 <= g <= 2 else "超出範圍"))
    print("  規格書表列 EfficientNet-Lite0: 4.7 M 參數, ~0.39 GMAC")

    print("\n=== 依 kernel 形狀彙總 ===")
    agg = {}
    for tag, ws, n_w, n_b, m in rows:
        a = agg.setdefault(tag, [0, 0, 0])
        a[0] += 1
        a[1] += n_w
        a[2] += m
    print("  %-9s %5s %14s %7s %12s %7s" % ("kernel", "層數", "參數", "參數%", "MMAC", "MAC%"))
    for k in sorted(agg, key=lambda x: -agg[x][2]):
        n, w, m = agg[k]
        print("  %-9s %5d %14s %6.1f%% %12.2f %6.1f%%"
              % (k, n, f"{w:,}", 100.0 * w / max(w_params, 1),
                 m / 1e6, 100.0 * m / max(macs, 1)))

    print("\n=== 權重最大的 10 層 ===")
    for tag, ws, n_w, n_b, m in sorted(rows, key=lambda r: -r[2])[:10]:
        print("  %-9s %-22s %12s params  %9.2f MMAC" % (tag, str(ws), f"{n_w:,}", m / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
