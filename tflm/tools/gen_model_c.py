#!/usr/bin/env python3
"""
gen_model_c.py — .tflite -> C 陣列。

用法:
    python3 gen_model_c.py <model.tflite> [--out <dir>]

產出:
    <dir>/model_data.h
    <dir>/model_data.cc

為什麼不用 xxd -i:
    (1) CentOS 7 不保證有 xxd（在 vim-common 裡）
    (2) xxd 不會加 alignas(16)。flatbuffer 未對齊時 TFLM 會在
        GetModel() 之後拿到垃圾值，症狀是 schema version 看起來正常
        但張量全錯 —— 非常難查。
"""
import argparse
import os

PER_LINE = 16


def emit(name, data, out_dir, decl_type="unsigned char"):
    os.makedirs(out_dir, exist_ok=True)
    stem = name
    with open(os.path.join(out_dir, f"{stem}.h"), "w") as f:
        f.write(f"""// 由 tools/gen_model_c.py 自動產生，請勿手改。
#ifndef {stem.upper()}_H_
#define {stem.upper()}_H_

extern const {decl_type} g_{stem}[];
extern const unsigned int g_{stem}_len;

#endif  // {stem.upper()}_H_
""")
    with open(os.path.join(out_dir, f"{stem}.cc"), "w") as f:
        f.write(f'// 由 tools/gen_model_c.py 自動產生，請勿手改。\n')
        f.write(f'#include "{stem}.h"\n\n')
        # alignas(16)：flatbuffer 必須對齊，否則 TFLM 讀到的張量是垃圾
        f.write(f"alignas(16) const {decl_type} g_{stem}[] = {{\n")
        for i in range(0, len(data), PER_LINE):
            chunk = data[i:i + PER_LINE]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int g_{stem}_len = {len(data)};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        data = f.read()

    emit("model_data", data, a.out)
    print(f"模型 {len(data)} bytes ({len(data)/1024:.1f} KiB)"
          f" -> {a.out}/model_data.{{h,cc}}")
    print(f"提醒：.rodata 會因此增加約 {len(data)/1024:.0f} KiB，"
          f"bare-metal 的 link.ld 要容得下")


if __name__ == "__main__":
    main()
