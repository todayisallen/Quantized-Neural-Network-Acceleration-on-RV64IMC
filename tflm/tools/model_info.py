#!/usr/bin/env python3
"""
model_info.py — 盤點 .tflite 用到哪些 operator、張量長什麼樣，
                並產生 TFLM 要用的 op resolver 標頭檔。

用法:
    python3 model_info.py <model.tflite> [--out <dir>]

產出:
    <dir>/resolver_gen.h      TFLM_OP_COUNT 與 TFLM_ADD_OPS(r)

為什麼要生成而不是手寫:
    MicroMutableOpResolver<N> 的 N 寫錯會在 AllocateTensors 時才爆，
    而且訊息不會告訴你少了哪個 op。這裡直接從模型列舉，不靠記憶。
    這份清單同時就是 B1「只保留用到的 operator」的依據。
"""
import argparse
import os
import re
import sys
from collections import Counter, OrderedDict

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
    except ImportError:
        sys.exit("找不到 ai_edge_litert 或 tensorflow。"
                 "  pip install --only-binary=:all: ai-edge-litert")

def adds_from_header(tflm_root):
    """從 micro_mutable_op_resolver.h 直接讀出所有 Add 方法。

    TFLM 有 120 個以上的 Add 方法，手維護對應表遲早會漏或寫錯一個，
    而寫錯的下場是 AllocateTensors() 失敗、訊息不告訴你是哪個 op。
    所以優先從原始碼推導：op 名稱去掉底線、忽略大小寫去比對，
    FULLY_CONNECTED -> AddFullyConnected、PADV2 -> AddPadV2、
    L2_NORMALIZATION -> AddL2Normalization 這些都能正確解析。
    """
    hdr = os.path.join(tflm_root, "tensorflow", "lite", "micro",
                       "micro_mutable_op_resolver.h")
    if not os.path.isfile(hdr):
        return None, hdr
    with open(hdr, "r", errors="replace") as f:
        src = f.read()
    names = set(re.findall(r"TfLiteStatus\s+(Add[A-Za-z0-9_]+)\s*\(", src))
    names -= {"AddBuiltin", "AddCustom", "AddDecode"}   # resolver 機制，不是 operator
    # 有些 Add 方法接受自訂的 TFLMRegistration，預設值是 Register_XXX()。
    # 把那個預設值也抓出來，產生的 resolver 才能在編譯期替換成自訂 kernel。
    defaults = dict(re.findall(
        r"TfLiteStatus\s+(Add[A-Za-z0-9_]+)\s*\(\s*const\s+TFLMRegistration&\s+"
        r"\w+\s*=\s*([A-Za-z0-9_]+)\(\)\s*\)", src))
    return ({n[3:].lower(): (n, defaults.get(n)) for n in names}, hdr)


def resolve(op_name, from_hdr):
    """op 名稱 -> (Add 方法名, 預設 registration 或 None)；解析不出來回 (None, None)。"""
    key = op_name.replace("_", "").lower()
    if from_hdr and key in from_hdr:
        return from_hdr[key]
    return (OP_TO_ADD.get(op_name), None)


# 後備對應表：找不到 TFLM 原始碼時才用這個。
OP_TO_ADD = {
    "ADD": "AddAdd",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "CONCATENATION": "AddConcatenation",
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "DEQUANTIZE": "AddDequantize",
    "FULLY_CONNECTED": "AddFullyConnected",
    "HARD_SWISH": "AddHardSwish",
    "LOGISTIC": "AddLogistic",
    "MAX_POOL_2D": "AddMaxPool2D",
    "MEAN": "AddMean",
    "MUL": "AddMul",
    "PAD": "AddPad",
    "PADV2": "AddPadV2",
    "QUANTIZE": "AddQuantize",
    "RELU": "AddRelu",
    "RELU6": "AddRelu6",
    "RESHAPE": "AddReshape",
    "RESIZE_BILINEAR": "AddResizeBilinear",
    "RESIZE_NEAREST_NEIGHBOR": "AddResizeNearestNeighbor",
    "SHAPE": "AddShape",
    "SOFTMAX": "AddSoftmax",
    "SQUEEZE": "AddSqueeze",
    "STRIDED_SLICE": "AddStridedSlice",
    "SUB": "AddSub",
    "SUM": "AddSum",
    "TRANSPOSE": "AddTranspose",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=".")
    ap.add_argument("--tflm-root",
                    default=os.path.expanduser("~/ca-project/work/tflite-micro"),
                    help="TFLM 原始碼根目錄，用來驗證 Add 方法名稱")
    a = ap.parse_args()

    from_hdr, hdr_path = adds_from_header(a.tflm_root)
    if from_hdr:
        print(f"（Add 方法由 {os.path.relpath(hdr_path, a.tflm_root)} 推導，"
              f"共 {len(from_hdr)} 個）")
    else:
        print(f"（找不到 {hdr_path}，改用內建後備對應表 {len(OP_TO_ADD)} 筆；"
              f"必要時用 --tflm-root 指定）")

    # BUILTIN_WITHOUT_DEFAULT_DELEGATES 不能省。
    # 用預設 resolver 的話 XNNPACK 會把一段子圖換成自己的節點，
    # op 盤點就會多出一個叫 DELEGATE 的東西 —— 那不是模型裡的 operator，
    # 而是 x86 的加速路徑。照著它去找 TFLM 的 Add 方法會找不到。
    interp = Interpreter(
        model_path=a.model,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()

    # ---------------------------------------------------------------- 輸入輸出
    print("=" * 66)
    print("輸入 / 輸出")
    print("=" * 66)
    for label, dets in (("輸入", interp.get_input_details()),
                        ("輸出", interp.get_output_details())):
        for d in dets:
            scale, zp = d["quantization"]
            shape = [int(s) for s in d["shape"]]      # numpy 2.x 的 repr 很難讀
            nelem = 1
            for s in shape:
                nelem *= s
            itemsize = 4 if "float" in str(d["dtype"]) else 1
            print(f"  {label}: {d['name']}")
            print(f"        shape={shape}  dtype={d['dtype'].__name__}"
                  f"  bytes={nelem * itemsize}")
            print(f"        scale={scale}  zero_point={zp}")
            if "float" in str(d["dtype"]):
                print("        !! 是浮點 —— 無 FPU 的目標不能這樣餵，"
                      "要確認模型是否真的全整數")

    # ---------------------------------------------------------------- op 盤點
    if not hasattr(interp, "_get_ops_details"):
        sys.exit("這個版本的 interpreter 沒有 _get_ops_details()，無法盤點 op")

    ops = interp._get_ops_details()
    tensors = {t["index"]: t for t in interp.get_tensor_details()}

    print()
    print("=" * 66)
    print(f"執行順序（共 {len(ops)} 個 op）")
    print("=" * 66)
    for o in ops:
        outs = list(o.get("outputs", []))
        shape = ([int(s) for s in tensors[outs[0]]["shape"]]
                 if outs and outs[0] in tensors else [])
        print(f"  [{o['index']:3d}] {o['op_name']:<24} -> {shape}")

    counts = Counter(o["op_name"] for o in ops)
    print()
    print("=" * 66)
    print(f"用到的 operator 種類：{len(counts)}")
    print("=" * 66)
    unknown = []
    for name, n in counts.most_common():
        add, reg = resolve(name, from_hdr)
        if add is None:
            unknown.append(name)
            print(f"  {name:<26} x{n:<4}  ** 無法解析 **")
        else:
            print(f"  {name:<26} x{n:<4}  {add}()"
                  + (f"  [可替換: {reg}]" if reg else ""))

    # ---------------------------------------------------------------- 浮點檢查
    nfloat = sum(1 for t in tensors.values() if "float" in str(t["dtype"]))
    print()
    print(f"float32 張量數：{nfloat}", end="")
    print("  （全整數模型應為 0；非 0 代表有 FP32 子圖）" if nfloat else "  OK")

    # ---------------------------------------------------------------- 產生標頭
    if unknown:
        print()
        print("!! 有無法對應的 operator，resolver_gen.h 仍會產生，")
        print("   但那幾個 op 會留 #error，請自行補上正確的 Add 方法名稱。")

    ordered = OrderedDict()
    for o in ops:                       # 依首次出現順序，讀起來比較像模型本身
        ordered.setdefault(o["op_name"], None)

    lines = [
        "// 由 tools/model_info.py 自動產生，請勿手改。",
        f"// 來源模型: {os.path.basename(a.model)}",
        f"// op 總數 {len(ops)}，種類 {len(counts)}",
        "#ifndef RESOLVER_GEN_H_",
        "#define RESOLVER_GEN_H_",
        "",
        f"#define TFLM_OP_COUNT {len(counts)}",
        "",
        "// 每個 operator 的 registration 都可以在編譯期換成自訂 kernel，例如",
        "//     -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()'",
        "// 這是 B1 / X1 替換 1x1 conv 實作的方式 —— 不需要改 TFLM 的原始碼。",
        "",
    ]
    for name in ordered:
        add, reg = resolve(name, from_hdr)
        if add is None or reg is None:
            continue
        lines += [
            f"#ifndef TFLM_REG_{name}",
            f"#define TFLM_REG_{name} tflite::micro::{reg}()",
            "#endif",
        ]
    lines += ["", "#define TFLM_ADD_OPS(r)  do {  \\"]
    for name in ordered:
        add, reg = resolve(name, from_hdr)
        if add is None:
            lines.append(f"    /* {name}: 無對應方法 */ \\")
            lines.append(f"#error \"補上 {name} 的 Add 方法\" \\")
        elif reg is None:
            # 這個 Add 方法不收 registration 參數，無法替換
            lines.append(f"    r.{add}();    /* {name} x{counts[name]} */ \\")
        else:
            lines.append(
                f"    r.{add}(TFLM_REG_{name});    /* {name} x{counts[name]} */ \\")
    lines += ["  } while (0)", "", "#endif  // RESOLVER_GEN_H_", ""]

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "resolver_gen.h")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print()
    print(f"已寫出 {path}")


if __name__ == "__main__":
    main()