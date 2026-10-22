#!/usr/bin/env python3
"""
gen_golden_from_bin.py — 把 host 跑出來的 tflm_out.bin 轉成 golden_data.{h,cc}。

用法:
    python3 gen_golden_from_bin.py host/tflm_out.bin --out gen --signed 0

為什麼需要這一步:
    Spike 上要驗的是「同一套 C++ 跑在 RV64 上有沒有跟跑在 x86 上位元相同」。
    那個比較必須是 BITEXACT，沒有容忍空間。
    如果 golden 留著 LiteRT 的版本，bare-metal 會帶著跨框架的那幾個 LSB 差異，
    你就分不出那是正常繼承還是移植寫錯 —— 這兩件事的處理方式完全不同。

    跨框架的比較（gen_io.py 產生的那份）留著當報告材料，不是驗收條件。
"""
import argparse
import os

PER_LINE = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binfile")
    ap.add_argument("--out", default=".")
    ap.add_argument("--signed", type=int, choices=[0, 1], default=None,
                    help="輸出張量是否為有號；省略則沿用 out/golden_data.h 現有的值")
    ap.add_argument("--config", default="?",
                    help="產生這份 golden 的 host 配置（B0/B1/...），只寫進來源字串")
    a = ap.parse_args()

    with open(a.binfile, "rb") as f:
        raw = f.read()
    if not raw:
        raise SystemExit(f"{a.binfile} 是空的")

    signed = a.signed
    if signed is None:
        # 沿用 gen_io.py 寫的值，避免手動指定錯誤
        hdr = os.path.join(a.out, "golden_data.h")
        signed = 0
        if os.path.isfile(hdr):
            for line in open(hdr):
                if "GOLDEN_DATA_SIGNED" in line:
                    signed = int(line.strip().split()[-1])
                    break
            print(f"（沿用 {hdr} 的 GOLDEN_DATA_SIGNED = {signed}）")
        else:
            print("（找不到現有的 golden_data.h，預設無號；可用 --signed 指定）")

    # golden 的來源字串不能沿用 gen_io.py 寫的那個 —— 那份是 LiteRT 算的，
    # 這份是 TFLM@x86 算的，兩者位元不同。沿用會讓 binary 自報的來源說謊。
    # 輸入是哪一份則要從 input_data.h 讀：golden 必須標明它配的是哪個輸入，
    # 不然「同一個 binary、不同輸入」的指令數差異又只能靠事後比對發現。
    in_src = "unknown"
    ihdr = os.path.join(a.out, "input_data.h")
    if os.path.isfile(ihdr):
        for line in open(ihdr):
            if "INPUT_DATA_SRC" in line and '"' in line:
                in_src = line.split('"')[1]
                break
    else:
        print(f"!! 找不到 {ihdr}，golden 的輸入來源記成 unknown")
    src = f"TFLM@x86(CONFIG={a.config}) in={in_src}"
    print(f"（來源字串: {src}）")

    ctype = "signed char" if signed else "unsigned char"
    vals = [b - 256 if (signed and b > 127) else b for b in raw]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "golden_data.h"), "w") as f:
        f.write(f"""// 由 tools/gen_golden_from_bin.py 產生，來源 {os.path.basename(a.binfile)}
// 這是 TFLM 在 x86 上的輸出，bare-metal 必須與它位元完全相同。
#ifndef GOLDEN_DATA_H_
#define GOLDEN_DATA_H_

#define GOLDEN_DATA_SIGNED {signed}

// 這份 golden 是誰、用什麼輸入算出來的。編進程式，讓每次執行自己印出來。
#define GOLDEN_DATA_SRC "{src}"

extern const {ctype} g_golden_data[];
extern const unsigned int g_golden_data_len;

#endif  // GOLDEN_DATA_H_
""")
    with open(os.path.join(a.out, "golden_data.cc"), "w") as f:
        f.write(f"// 由 tools/gen_golden_from_bin.py 產生，"
                f"來源 {os.path.basename(a.binfile)}\n")
        f.write('#include "golden_data.h"\n\n')
        f.write(f"alignas(16) const {ctype} g_golden_data[] = {{\n")
        for i in range(0, len(vals), PER_LINE):
            f.write("  " + ", ".join(str(v) for v in vals[i:i + PER_LINE]) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int g_golden_data_len = {len(vals)};\n")

    print(f"{len(raw)} bytes -> {a.out}/golden_data.{{h,cc}}  ({ctype})")
    print("之後 host 重跑應該要得到 BITEXACT；拿不到就是這一步弄錯了")


if __name__ == "__main__":
    main()