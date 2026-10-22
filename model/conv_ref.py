#!/usr/bin/env python3
"""
conv_ref.py — 1x1 CONV_2D 的純整數黃金參考，對照 LiteRT 驗證。

1x1 conv 在數學上就是 GEMM：

    acc[y][x][oc] = sum_ic (a[y][x][ic] - z_a) * w[oc][ic] + bias[oc]
                  = sum_ic a[y][x][ic]*w[oc][ic] - z_a*sum_w[oc] + bias[oc]
                    └──── xqmac8 負責 ────┘  └── 離線預算 ──┘

    out = requantize(acc, mult[oc], shift[oc], z_out)

本模型 50 層權重的 zero-point 全為 0（對稱量化），故 sum_w[oc] 可離線
預先計算，xqmac8 只需執行純粹的有號點積。

同時驗證 xqmac8 的 8 元素分段累加等於一次算完（整數加法可結合）。

用法:
    python3 conv_ref.py <model.tflite> [op_idx ...]
不給 op_idx 則自動挑 MAC 最大的三個 1x1 層。
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
from qmath import quantize_multiplier, requantize

TOL_RATE = 0.005          # 與 LiteRT 可容忍的 1-LSB 分歧比例


def build_interpreter(model):
    """關掉預設 delegate，走 TFLite 的定點參考路徑而非 XNNPACK 的 FP32。"""
    kw = {"model_path": model, "experimental_preserve_all_tensors": True}
    if OpResolverType is not None:
        for attr in ("BUILTIN_WITHOUT_DEFAULT_DELEGATES", "BUILTIN_REF", "BUILTIN"):
            if hasattr(OpResolverType, attr):
                kw["experimental_op_resolver_type"] = getattr(OpResolverType, attr)
                break
    try:
        return Interpreter(**kw)
    except TypeError:
        return Interpreter(model_path=model)


def collect_1x1(it, det):
    """找出所有 1x1 CONV_2D，附上 MAC 數。"""
    out = []
    for op in it._get_ops_details():
        if op["op_name"] != "CONV_2D":
            continue
        ins, outs = list(op["inputs"]), list(op["outputs"])
        if len(ins) < 2 or not outs:
            continue
        i_in, i_w = ins[0], ins[1]
        i_b = ins[2] if len(ins) > 2 and ins[2] >= 0 else None
        i_o = outs[0]
        if i_w not in det or i_o not in det or i_in not in det:
            continue
        ws = tuple(int(x) for x in det[i_w]["shape"])
        if len(ws) != 4 or ws[1] != 1 or ws[2] != 1:
            continue
        osh = tuple(int(x) for x in det[i_o]["shape"])
        macs = osh[1] * osh[2] * ws[0] * ws[3]
        out.append(dict(idx=op["index"], i_in=i_in, i_w=i_w, i_b=i_b, i_o=i_o,
                        macs=macs, K=ws[3], OC=ws[0]))
    return out


def check_layer(it, det, L, verbose=True):
    a = it.get_tensor(L["i_in"])          # (1, H, W, K)  int8
    w = it.get_tensor(L["i_w"])           # (OC, 1, 1, K) int8
    o = it.get_tensor(L["i_o"])           # (1, H, W, OC) int8
    b = it.get_tensor(L["i_b"]) if L["i_b"] is not None else None

    qa = det[L["i_in"]]["quantization_parameters"]
    qw = det[L["i_w"]]["quantization_parameters"]
    qo = det[L["i_o"]]["quantization_parameters"]

    z_in = int(qa["zero_points"][0])
    s_in = float(qa["scales"][0])
    z_out = int(qo["zero_points"][0])
    s_out = float(qo["scales"][0])
    s_w = np.asarray(qw["scales"], dtype=np.float64)
    assert np.all(np.asarray(qw["zero_points"]) == 0), "weights must be symmetric"

    # 輸出域：int8 還是 uint8
    odt = det[L["i_o"]]["dtype"].__name__
    qmin, qmax = (0, 255) if odt == "uint8" else (-128, 127)

    a = np.squeeze(a, axis=0).astype(np.int32)        # (H, W, K)
    o = np.squeeze(o, axis=0).astype(np.int32)        # (H, W, OC)
    W2 = w.reshape(w.shape[0], -1).astype(np.int32)   # (OC, K)
    H, Wd, K = a.shape
    OC = W2.shape[0]
    assert o.shape == (H, Wd, OC), (o.shape, (H, Wd, OC))
    if s_w.size == 1:
        s_w = np.repeat(s_w, OC)

    sum_w = W2.sum(axis=1)                            # 離線可預算
    bias = (b.astype(np.int32) if b is not None else np.zeros(OC, dtype=np.int32))

    mult = np.zeros(OC, dtype=np.int64)
    shift = np.zeros(OC, dtype=np.int64)
    for c in range(OC):
        M = float(s_in * s_w[c] / s_out)
        mult[c], shift[c] = quantize_multiplier(M) if M > 0 else (0, 0)

    # 1x1 conv == GEMM：把 (H,W) 攤平成列
    A = a.reshape(H * Wd, K)
    dot = A @ W2.T - np.int64(z_in) * sum_w[None, :]     # xqmac8 負責這段
    acc = dot + bias[None, :]                            # 再加 bias

    total = (31 - shift)[None, :]
    prod = acc.astype(np.int64) * mult[None, :]
    got = (prod + (np.int64(1) << (total - 1))) >> total
    got = np.clip(got + z_out, qmin, qmax).astype(np.int32)

    ref = o.reshape(H * Wd, OC)
    diff = got - ref
    nbad = int((diff != 0).sum())
    rate = nbad / float(ref.size)
    maxd = int(np.abs(diff).max()) if nbad else 0

    fp_ok = fp_tot = 0
    if verbose:
        print("\n=== op#%d  1x1 conv  %dx%d x K=%d -> OC=%d  (%.2f MMAC) ==="
              % (L["idx"], H, Wd, K, OC, L["macs"] / 1e6))
        print("  s_in=%.6g z_in=%-5d  s_out=%.6g z_out=%-5d  out dtype=%s"
              % (s_in, z_in, s_out, z_out, odt))
        def _bits(x):
            return int(max(abs(int(x.min())), abs(int(x.max())))).bit_length() + 1
        print("  點積 (xqmac8 範圍) [%d, %d] -> %d bits" % (dot.min(), dot.max(), _bits(dot)))
        print("  bias 範圍         [%d, %d] -> %d bits" % (bias.min(), bias.max(), _bits(bias)))
        print("  acc = 點積+bias   [%d, %d] -> %d bits  (int32 上限 %d)"
              % (acc.min(), acc.max(), _bits(acc), (1 << 31) - 1))
        print("  K %% 8 = %d   每個輸出需要 %d 條 xqmac8" % (K % 8, K // 8))
        print("  shift 範圍 [%d, %d]" % (shift.min(), shift.max()))
        if nbad == 0:
            print("  MATCH: bit-exact vs LiteRT (%d 元素)" % ref.size)
        elif maxd <= 1 and rate <= TOL_RATE:
            print("  PASS (容忍範圍內): %d / %d 差 1 LSB (%.4f%% <= %.2f%%)"
                  % (nbad, ref.size, 100 * rate, 100 * TOL_RATE))
        else:
            print("  不符: %d / %d (%.4f%%)，最大差 %d LSB"
                  % (nbad, ref.size, 100 * rate, maxd))
            bad = np.argwhere(diff != 0)[:3]
            for m, n in bad:
                print("    [%d,%d] acc=%d mult=%d shift=%d got=%d ref=%d"
                      % (m, n, acc[m, n], mult[n], shift[n], got[m, n], ref[m, n]))

        # 指紋檢查：所有不符是否都落在距捨入邊界不到一個 M_n 的位置
        # （這才是「sub-LSB 捨入差異」的判準，固定比例門檻會被 M 的大小誤導）
        fp_ok = fp_tot = 0
        Mmax = 0.0
        for m, n in np.argwhere(diff != 0):
            t = int(31 - shift[n])
            frac = (int(prod[m, n]) % (1 << t)) / float(1 << t)
            Mn = int(mult[n]) / float(1 << t)
            Mmax = max(Mmax, Mn)
            fp_tot += 1
            if 0.5 - frac < Mn:
                fp_ok += 1
        if fp_tot:
            print("  指紋 (0.5-frac) < M_n: %d / %d  (%.1f%%)   M_n 最大 %.4g"
                  % (fp_ok, fp_tot, 100.0 * fp_ok / fp_tot, Mmax))
            Mall = (mult / (1 << (31 - shift)).astype(np.float64))
            print("  各通道 M_n 範圍 [%.3g, %.3g]，平均 %.3g -> 預期不符率約 %.3f%%"
                  % (Mall.min(), Mall.max(), Mall.mean(), 100.0 * Mall.mean()))

        # xqmac8 的分段累加正確性
        if K % 8 == 0:
            row = A[0].astype(np.int64)
            for c in (0, OC // 2, OC - 1):
                wr = W2[c].astype(np.int64)
                seg = sum(int(np.dot(row[k:k + 8], wr[k:k + 8])) for k in range(0, K, 8))
                assert seg == int(np.dot(row, wr)), (L["idx"], c)
            print("  xqmac8 8-way 分段累加 == 完整點積: OK")

    return nbad, ref.size, maxd, fp_ok, fp_tot


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model = sys.argv[1]
    want = [int(x) for x in sys.argv[2:]]

    it = build_interpreter(model)
    it.allocate_tensors()
    det = {t["index"]: t for t in it.get_tensor_details()}

    # 固定輸入，結果可重現
    inp = it.get_input_details()[0]
    rng = np.random.default_rng(0)
    info = np.iinfo(inp["dtype"])
    x = rng.integers(info.min, info.max + 1, size=inp["shape"]).astype(inp["dtype"])
    it.set_tensor(inp["index"], x)
    it.invoke()
    print("invoke ok (input dtype=%s)" % inp["dtype"].__name__)

    layers = collect_1x1(it, det)
    if want:
        layers = [l for l in layers if l["idx"] in want]
    else:
        layers = sorted(layers, key=lambda x: -x["macs"])[:3]

    tb = tn = tm = 0
    fo = ft = 0
    for L in layers:
        nb, n, md, a_, b_ = check_layer(it, det, L)
        tb += nb; tn += n; tm = max(tm, md); fo += a_; ft += b_

    print("\n" + "=" * 60)
    print("合計: %d / %d 不符 (%.4f%%)，最大差 %d LSB"
          % (tb, tn, 100.0 * tb / tn, tm))
    if ft:
        print("指紋 (0.5-frac) < M_n: %d / %d  (%.2f%%)" % (fo, ft, 100.0 * fo / ft))
    # 判定不看比例 —— 比例會隨 M_n 大小變動。
    # 看的是 (a) 誤差恆為 1 LSB (b) 全部落在捨入邊界。
    ok = (tm <= 1) and (ft == 0 or fo == ft)
    print("結論: %s" % ("全部分歧皆為捨入邊界的 1-LSB 差異，參考實作正確"
                      if ok else "出現非邊界或 >1 LSB 的分歧，需檢查"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())