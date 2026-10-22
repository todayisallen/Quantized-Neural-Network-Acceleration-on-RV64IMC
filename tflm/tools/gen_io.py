#!/usr/bin/env python3
"""
gen_io.py — 產生固定的測試輸入，並用 LiteRT 算出 golden 輸出。

用法:
    python3 gen_io.py <model.tflite> [--out <dir>] [--seed 0]
    python3 gen_io.py <model.tflite> --image cat.jpg --norm pm1

產出:
    <dir>/input_data.{h,cc}    量化後的輸入（int8/uint8 原始位元）
    <dir>/golden_data.{h,cc}   LiteRT 的輸出
    <dir>/input_int8.bin       同一份輸入的 raw bytes（除錯用）
    <dir>/golden_int8.bin

兩個關鍵設計:

1. 輸入以「量化後的原始位元」固定下來，不是以圖片或浮點固定。
   這樣 LiteRT 與 TFLM 吃到的是完全相同的 bytes，
   任何差異都只能來自 kernel 本身，不會混進前處理的差異。

2. OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES 不能省。
   預設的 XNNPACK delegate 會用 FP32 做 rescale，
   跟 TFLM 的純整數路徑對不起來 —— 之前追了半小時就是這件事。
"""
import argparse
import os
import sys

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
    except ImportError:
        sys.exit("找不到 ai_edge_litert。"
                 "  pip install --only-binary=:all: ai-edge-litert")

PER_LINE = 16


def emit(stem, data, out_dir, ctype, src="unknown"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.h"), "w") as f:
        # _SIGNED 讓 nn_main.cc 知道要用 int8_t 還是 uint8_t 解讀張量。
        # 這不能猜：EfficientNet-Lite0 的 I/O 是 uint8（內部才是 int8），
        # 用 int8 去讀會把 128~255 當成負數，top-k 排序直接錯掉。
        signed_flag = 1 if ctype == "signed char" else 0
        f.write(f"""// 由 tools/gen_io.py 自動產生，請勿手改。
#ifndef {stem.upper()}_H_
#define {stem.upper()}_H_

#define {stem.upper()}_SIGNED {signed_flag}

// 這份資料的來源。編進程式，讓每次執行都自己印出來 ——
// 量測結果若不記錄輸入是什麼，不同輸入的數字混在一起只能靠事後比對發現。
// （實際踩過：真實圖片與 seed 0 的指令數差 94,157，一度被誤判成建置問題。）
#define {stem.upper()}_SRC "{src}"

extern const {ctype} g_{stem}[];
extern const unsigned int g_{stem}_len;

#endif  // {stem.upper()}_H_
""")
    with open(os.path.join(out_dir, f"{stem}.cc"), "w") as f:
        f.write("// 由 tools/gen_io.py 自動產生，請勿手改。\n")
        f.write(f'#include "{stem}.h"\n\n')
        f.write(f"alignas(16) const {ctype} g_{stem}[] = {{\n")
        vals = data.astype(np.int64).tolist()
        for i in range(0, len(vals), PER_LINE):
            f.write("  " + ", ".join(str(v) for v in vals[i:i + PER_LINE]) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int g_{stem}_len = {len(vals)};\n")


def build_input(det, args):
    shape = [int(s) for s in det["shape"]]
    dtype = det["dtype"]
    scale, zp = det["quantization"]

    if dtype not in (np.int8, np.uint8):
        sys.exit(f"輸入 dtype 是 {dtype.__name__}，不是量化整數型別。"
                 " 這個模型不適合無 FPU 的目標。")

    info = np.iinfo(dtype)

    if args.image:
        try:
            from PIL import Image
        except ImportError:
            sys.exit("--image 需要 pillow")
        _, h, w, _ = shape
        img = Image.open(args.image).convert("RGB").resize((w, h), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32)          # [0, 255]
        if args.norm == "pm1":
            x = x / 127.5 - 1.0                        # EfficientNet-Lite 的前處理
        elif args.norm == "unit":
            x = x / 255.0
        # raw: 不動
        if scale == 0:
            sys.exit("輸入張量沒有量化參數，無法從圖片量化")
        q = np.round(x / scale + zp)
        q = np.clip(q, info.min, info.max).astype(dtype)
        src = f"image={os.path.basename(args.image)} norm={args.norm}"
    else:
        # 固定種子的隨機位元。目的是驗證位元級一致性，不是看辨識結果，
        # 所以不需要是真實圖片；可重現比真實重要。
        rng = np.random.default_rng(args.seed)
        q = rng.integers(info.min, info.max + 1, size=shape, dtype=np.int64).astype(dtype)
        src = f"random seed={args.seed}"

    return q.reshape(shape), src


def load_labels(n_out):
    """讀 tools/imagenet1k_labels.txt。只有輸出剛好 1000 類才翻譯。

    EfficientNet 有 1000 類與 1001 類（多一個 background）兩種版本，
    索引整體差 1。維度不符就回 None、印原始索引 —— 給一個錯一格的
    名字比不給名字更糟，因為那會讓人以為檢查過了。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "imagenet1k_labels.txt")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t")[0])
    if len(rows) != 1000:
        print(f"!! {path} 有 {len(rows)} 筆，應該是 1000 —— 不翻譯類別名")
        return None
    if n_out != 1000:
        print(f"!! 模型輸出 {n_out} 類，不是 1000 "
              f"（1001 類版多一個 background，索引差 1）—— 不翻譯類別名")
        return None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--norm", choices=["pm1", "unit", "raw"], default="pm1")
    a = ap.parse_args()

    # 關鍵：關掉預設 delegate，走純整數參考實作
    interp = Interpreter(
        model_path=a.model,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    idet = interp.get_input_details()[0]
    odet = interp.get_output_details()[0]

    x, src = build_input(idet, a)
    print(f"輸入: shape={list(x.shape)} dtype={x.dtype} ({src})")

    interp.set_tensor(idet["index"], x)
    interp.invoke()
    y = interp.get_tensor(odet["index"])
    print(f"輸出: shape={list(y.shape)} dtype={y.dtype}")

    xf, yf = x.flatten(), y.flatten()

    # top-5。模型若已含 SOFTMAX，這是機率；否則是 logits。排序一樣有意義。
    #
    # kind="stable" 配上對 -v 排序，是為了讓平手時取「索引較小」的那個。
    # argsort(v)[::-1] 會讓平手時變成取索引較大的，跟 nn_main.cc 的
    # top_k 不一致，兩邊印出來的 class 編號就會差一個，看起來像 bug。
    top = np.argsort(-yf.astype(np.int32), kind="stable")[:5]
    oscale, ozp = odet["quantization"]
    names = load_labels(len(yf))

    print("LiteRT top-5:")
    for r, i in enumerate(top):
        deq = (int(yf[i]) - ozp) * oscale if oscale else float(yf[i])
        nm = f"   {names[int(i)]}" if names else ""
        print(f"  {r+1}. class {int(i):4d}   q={int(yf[i]):4d}   "
              f"dequant={deq:.6f}{nm}")

    if a.image:
        print()
        print("  ^ 這是整個專案裡唯一的**絕對**正確性檢查。")
        print("    其他檢查全是相對的（RV64 ≡ x86、TFLM ≡ LiteRT、")
        print("    新 kernel ≡ 參考 kernel）—— 兩邊讀同一份模型檔，")
        print("    模型本身有問題的話會一致地錯，BITEXACT 照樣成立。")
        print("    top-1 要對得上圖片內容；對不上就先別管效能數字。")

    ctype = {np.dtype("int8"): "signed char",
             np.dtype("uint8"): "unsigned char"}
    emit("input_data", xf, a.out, ctype[np.dtype(x.dtype)], src)
    emit("golden_data", yf, a.out, ctype.get(np.dtype(y.dtype), "signed char"),
         f"LiteRT({src})")

    with open(os.path.join(a.out, "input_int8.bin"), "wb") as f:
        f.write(xf.tobytes())
    with open(os.path.join(a.out, "golden_int8.bin"), "wb") as f:
        f.write(yf.tobytes())

    print(f"\n已寫出 {a.out}/input_data.{{h,cc}}, golden_data.{{h,cc}}, *.bin")
    print(f"input {xf.nbytes} bytes, golden {yf.nbytes} bytes")


if __name__ == "__main__":
    main()