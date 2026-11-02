# RV64IMC 上的量化神經網路推論加速

高等計算機結構期中專題。在 **bare-metal RV64IMC** 上執行完整的
EfficientNet-Lite0（W8A8 量化，0.385 GMAC），用 Spike 量測指令數、
找出瓶頸、加入兩條自訂指令，並驗證每一步的輸出與軟體參考**位元完全相同**。

```
B0  未修改的 TFLM 參考實作         19,575,565,013 條    1.000x   50.82 inst/MAC
B1  軟體優化（1x1 conv + depthwise）  3,682,312,012 條    5.316x    9.56
X1  + xqmac8（8 路 packed int8 MAC）  1,590,153,616 條   12.310x    4.13
X2  + xqrq（再量化）                  1,271,170,850 條   15.400x    3.30
```

全部配置在 x86 與 RV64/Spike 上 **BITEXACT**，包含真實圖片
（LiteRT top-1 = class 652 military uniform）。

---

## 快速開始

```bash
git clone <this-repo> ca-project-repo && cd ca-project-repo/tflm

bash scripts/setup_env.sh all
```

`all` 會依序做五件事，每一步跑完都會自己驗證再往下：

| 階段 | 時間 | 內容 |
|---|---|---|
| `python` | ~2 分 | `ai-edge-litert` / `numpy` / `pillow` |
| `spike` | ~15 分 | clone riscv-isa-sim、套本專案的 patch、建置、安裝 |
| `tflm` | ~10 分 | clone tflite-micro、套 `tflm-patch/`、建 x86 與 riscv64 兩個庫 |
| `gen` | ~5 分 | 產生模型的 C 陣列、固定輸入、交換 golden |
| `verify` | ~5 分 | kernel 等價性 → 自訂指令 → 完整模型 BITEXACT |

**約 35 分鐘。** 階段可以單獨重跑：

```bash
bash scripts/setup_env.sh spike      # 只重建 Spike
bash scripts/setup_env.sh verify     # 只重跑驗證
```

跑完接著是完整量測：

```bash
bash scripts/measure.sh B0 B1 B1C B1D X1 X2    # ~5 分，六個配置
bash scripts/instr_mix.sh X2                   # ~30 秒，指令組成
bash scripts/verify_image.sh B0 B1 X1 X2       # ~5 分，真實圖片
for C in B1 X1 X2; do make -C bare CONFIG=$C bench_run; done
```

數字應該與下面「預期輸出」一致。

---

## 前置需求

```bash
bash scripts/setup_env.sh check
```

秒級，不裝任何東西 —— 逐項列出有沒有、缺的怎麼裝。`all` 會在最前面自動跑一次。

| 需求 | 版本 | `setup_env.sh` 會裝嗎 |
|---|---|---|
| 平台 | Linux x86_64 或 macOS | — |
| `git` / `make` / `g++`（C++17）/ `python3` / `dtc` | | ❌ **自己裝** |
| RISC-V toolchain | `riscv64-unknown-elf-*`，**標準版即可** | ❌ **自己裝** |
| Spike | riscv-isa-sim `f11c7b6` + `patches/spike-xq.patch` | ✅ |
| TFLM | tflite-micro `64b3de8` + `tflm-patch/` | ✅ |
| Python 套件 | ai-edge-litert 1.4.0 / numpy 2.2.6 / pillow 12.2.0 | ✅ `pip --user` |
| 模型 | `model/eff-lite0/model.tflite` | ✅ 已在倉庫內 |

`setup_env.sh` 全程**不需要 root**，東西都裝在你指定的路徑下。
要自己裝的那兩列多半需要 root，所以才不代勞。

### 安裝那兩列

**基本工具**

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y \
     git make g++ python3-pip device-tree-compiler autoconf automake libtool

# Fedora / RHEL 9+
sudo dnf install -y git make gcc-c++ python3-pip dtc autoconf automake libtool

# macOS
brew install dtc autoconf automake libtool
```

**RISC-V toolchain** —— 要的是 **bare-metal 的 `riscv64-unknown-elf` 三元組**。

```bash
# Debian / Ubuntu
sudo apt install -y gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf
```

其他平台（含 Fedora / RHEL / macOS）下載預編 release，解壓後把 `bin/` 加進
`PATH`，這條到哪都通：

<https://github.com/riscv-collab/riscv-gnu-toolchain/releases>

> **不要用 `riscv64-linux-gnu`。** Fedora 主倉庫只有
> `gcc-riscv64-linux-gnu`，那是給 Linux 使用者空間的三元組，不是
> bare-metal 的 `-unknown-elf`；`bare/Makefile` 找的是後者。

裝完重跑 `bash scripts/setup_env.sh check`，全綠就可以 `all`。
套件名稱因版本而異，`check` 會直接告訴你還缺哪個。

### 三件要知道的事

1. **不需要特製 toolchain。** 標準版的預設 code model 是 `medlow`，連結不到
   `0x80000000`；倉庫附了一份 medany 建的 libgcc，`bare/Makefile` 偵測到
   `medlow` 就**自動**用它，不必設任何變數。省掉數小時的 toolchain 建置。

2. **兩個 commit 是重現條件，不是建議。**
   `MultiplyByQuantizedMultiplier` 的捨入模式會隨 TFLM 版本變，而 `xqrq`
   必須與它位元相同。詳見下面「捨入模式」。

3. **改了編譯器或 libgcc 要先 `rm -rf bare/build`。** `make` 只比時間戳，
   沿用舊物件檔不會有任何警告，但連結出來的東西不對應你記錄的設定。

路徑可以覆寫（`TFLM_ROOT` 會自動跟著 `WORK`）：

```bash
WORK=/path/to/src RISCV_PREFIX=/path/to/install bash scripts/setup_env.sh all
```

---

## 沒套 Spike patch 的失敗症狀（重要）

`xqmac8` / `xqrq` 是本專案自己加的指令，任何標準 Spike 都不認得它們。
沒套 patch 就跑 X1/X2，Spike 遇到未知指令會 illegal instruction trap 到
`mtvec`（bare-metal 沒裝 trap handler，預設 0），從位址 0 取指令再次 fault，
變成**無窮 trap 迴圈**。

症狀是：**印完開頭幾行就停住，沒有任何錯誤訊息**，一直跑到 timeout。
開發過程中實際踩過一次，空轉了 16 分鐘才發現用到的是舊版 spike。

確認 Spike 認得這三條指令：

```bash
ls <spike-build>/xqtest.o <spike-build>/xqmac8.o <spike-build>/xqrq.o
which spike && ls -l $(which spike)      # 確認是剛建的那個
```

`setup_env.sh spike` 會自動做這個檢查。

---

## 分階段手動執行

`setup_env.sh` 做的就是下面這些，列出來是為了讓每一步都可查。

### 1. Spike

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim && git checkout f11c7b6
git apply <repo>/tflm/patches/spike-xq.patch
mkdir -p build && cd build
../configure --prefix=$HOME/ca-project/riscv --enable-commitlog --without-boost-asio
make -j8 && make install
```

> `--enable-commitlog` 不能省 —— `scripts/instr_mix.sh` 用 Spike 的 `-g`
> PC histogram 做指令組成分析。

patch 新增三條指令到 custom-0 的 opcode 空間，並改動 `riscv/encoding.h`
與 `riscv/riscv.mk.in`。原始碼也直接放在 `patches/spike/insns/`。

**檢查點**：`build/` 下出現 `xqtest.o`、`xqmac8.o`、`xqrq.o`。

### 2. TFLM

```bash
git clone https://github.com/tensorflow/tflite-micro.git
cd tflite-micro && git checkout 64b3de8
MK=tensorflow/lite/micro/tools/make/Makefile
make -f $MK third_party_downloads

# 套 tflm-patch —— 兩個檔，放置路徑由 TFLM 的 Makefile 決定
mkdir -p tensorflow/lite/micro/riscv64_generic
cp <repo>/tflm/tflm-patch/debug_log.cc tensorflow/lite/micro/riscv64_generic/
cp <repo>/tflm/tflm-patch/riscv64_generic_makefile.inc \
   tensorflow/lite/micro/tools/make/targets/

make -f $MK TARGET=linux TARGET_ARCH=x86_64 microlite -j8   # x86，host 驗證用
make -f $MK TARGET=riscv64_generic TARGET_ARCH=riscv64 microlite -j8   # riscv64

export TFLM_ROOT=$PWD
```

`tflm-patch` 的兩個檔分別解決：

- **`debug_log.cc`** —— 原版用 `vfprintf(stderr, …)`，那需要 newlib 的 stdio
  與一整套 syscall。bare-metal 上只要能吐字元，所以改成
  「eyalroz 的 `vsnprintf_` 格式化 → `plat_puts` 送 HTIF」。
  `plat_puts()` 由 `bare/main_bare.cc` 提供。
- **`riscv64_generic_makefile.inc`** —— TFLM 內建的 riscv target 是 RV32 的。
  這份以它為基礎改成 `rv64imc_zicsr` / `lp64` / `medany`，不加
  `--specs=nano.specs`（我們自己做 bare-metal 連結）。

> **`TARGET_ARCH=riscv64` 不能省。** `.inc` 裡雖然寫了
> `TARGET_ARCH := riscv64`，但 TFLM 的 Makefile 在 include 它**之前**
> 就算好了 `GENDIR := gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)`，
> 所以那個賦值對產物路徑無效 —— 漏掉就會建到
> `gen/riscv64_generic_x86_64_default_gcc/`。庫的內容一樣（旗標來自 `.inc`），
> 但目錄名與本專案記錄的不一致。

**檢查點**：

```bash
find gen -name 'libtensorflow-microlite*.a'
# 要有兩個：
#   gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a
#   gen/riscv64_generic_riscv64_default_gcc/lib/libtensorflow-microlite.a
```

### 3. 產生 `gen/`

```bash
cd <repo>/tflm
MODEL=../model/eff-lite0/model.tflite

python3 tools/model_info.py  $MODEL --out gen    # operator 清單 -> resolver_gen.h
python3 tools/gen_model_c.py $MODEL --out gen    # 模型 -> C 陣列（約 30 MB 文字）
python3 tools/gen_io.py      $MODEL --out gen --seed 0   # 固定輸入 + LiteRT golden

make -C host CONFIG=B0 run                               # 產生 host/tflm_out.bin
python3 tools/gen_golden_from_bin.py host/tflm_out.bin --out gen --config B0
find gen -type f -exec touch {} +
```

最後兩步是**把 golden 從 LiteRT 換成 TFLM@x86 的輸出** —— 理由見下面
「兩種比較」。`touch` 不能省：產生/還原之後 `gen/` 必須比物件檔新，
否則 `make` 會認為不用重編（實際踩過，而且沒有任何警告）。

**檢查點**：

```bash
grep INPUT_DATA_SRC  gen/input_data.h     # random seed=0
grep GOLDEN_DATA_SRC gen/golden_data.h    # TFLM@x86(CONFIG=B0) in=random seed=0
```

這兩個字串會**編進執行檔**，每次執行自己印出來 —— 量測結果若不記錄輸入
是什麼，不同輸入的數字混在一起只能靠事後比對發現（實際踩過：真實圖片與
種子 0 的指令數差 94,157，一度被誤判成建置問題）。

### 4. 驗證

```bash
make -C bare paths            # 環境自檢，每一行都不能有 !!
make -C host paths

bash tests/kernel/run.sh                  # kernel 等價性，x86，秒級
make -C bare t_xqrq_run                   # xqrq：pass=10037 fail=0
make -C bare CONFIG=X2 run                # 完整模型：BITEXACT
```

`t_xqmac8` 是獨立的單檔測試（不連 TFLM，標準 toolchain 就能編）：

```bash
riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
  -O2 -nostdlib -nostartfiles \
  -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns \
  -T ../tests/env/link.ld \
  bare/t_xqmac8.c -o /tmp/t_xqmac8
spike --isa=rv64imc_zicsr -m64 /tmp/t_xqmac8 </dev/null    # pass=10024 fail=0
```

> `-fno-tree-loop-distribute-patterns` 不能省。`splat()` 裡有個填值迴圈，
> GCC 會把它認成 `memset` 的 pattern 而產生對 `memset` 的呼叫 —— 但
> `-nostdlib` 沒有 libc，於是連結時 `undefined reference to memset`。

---

## 預期輸出

### 配置對照（`results/summary.md`）

輸入為固定種子 0，單位 `instret`（已退休指令數），arena 1713 KiB。

| config | Invoke | 相對 B0 | inst/MAC | AllocateTensors | 框架 overhead | `.text` | 判定 |
|:--|---:|---:|---:|---:|---:|---:|:--|
| B0 | 19,575,565,013 | 1.000x | 50.82 | 15,058,298 | 4,473 | 121,760 | BITEXACT |
| B1C | 5,281,442,410 | 3.706x | 13.71 | 15,058,298 | 4,473 | 117,440 | BITEXACT |
| B1D | 17,976,434,615 | 1.089x | 46.67 | 15,058,298 | 4,473 | 121,560 | BITEXACT |
| B1 | 3,682,312,012 | 5.316x | 9.56 | 15,058,298 | 4,473 | 117,240 | BITEXACT |
| X1 | 1,590,153,616 | 12.310x | 4.13 | 15,058,298 | 4,473 | 117,436 | BITEXACT |
| X2 | 1,271,170,850 | 15.400x | 3.30 | 15,058,298 | 4,473 | 116,856 | BITEXACT |

- **B1C / B1D** 把 B1 的兩項改動拆開（只開 1×1 / 只開 depthwise），用於歸因。
  加速比不可相乘（3.706 × 1.089 ≠ 5.316），但**節省量可以相加**，
  因為兩項動的是互不相交的 operator：
  `14,294,122,603 + 1,599,130,398 = 15,893,253,001`，與 B1 實際的節省量**差 0**。
- **`AllocateTensors` 與框架 overhead 在六個配置中完全相同** ——
  B1/X1/X2 都沒有改動 Prepare，本來就該一樣。這是個健全性檢查。
- **`Invoke` 裡沒有任何浮點運算。** rv64imc 沒有 FPU，而
  `QuantizeMultiplier` 用 `double`，所以有軟體模擬：`__muldf3` 4,385,796 +
  `__divdf3` 2,118,476 = 6,504,272 條（全程的 0.50%）。**全部落在
  `AllocateTensors`**（占它的 43%），`Invoke` 內為 0 —— 乘數在 Prepare
  階段就算成整數 `multiplier`/`shift` 存起來了。
- **`.text` 不含模型**（模型在 `.rodata`，約 5.3 MB）。
  X2 的 `.text` 比 X1 小 580 bytes、比 B0 小 4,904 bytes（−4.0%）——
  兩條自訂指令讓程式碼**變小**，因為 requantize 的呼叫點從
  「call + 加 offset + 兩個 clamp」塌縮成「`ld` + `xqrq` + 存」。

### 指令組成（X2，`results/X2.mix.txt`）

| 類別 | 指令數 | 占比 |
|---|---:|---:|
| load | 283,551,893 | 21.60% |
| store | 27,207,345 | 2.07% |
| mul/div | 78,225,289 | 5.96% |
| alu | 518,188,400 | 39.47% |
| branch | 352,213,721 | 26.83% |
| jump | 4,870,788 | 0.37% |
| **custom** | **48,653,472** | **3.71%** |
| 合計 | 1,312,935,133 | |

壓縮指令（RVC）482,327,552（36.74%）、`instruction / MAC` 3.41、
`load+store / MAC` 0.81（histogram 涵蓋整個程式含 `AllocateTensors`，
所以比 Invoke 的 3.30 略高）。

**自訂指令利用率**，兩條獨立的路徑算出同一個數字：

```
X2 的 custom   48,653,472
X1 的 custom   42,317,184   (= xqmac8，X2 沒改動 MAC 路徑)
─────────────────────────
xqrq            6,336,288

模型結構算出的輸出元素數（tools/check_packed.py）    6,737,696
扣掉 stem 3x3（112x112x32，退回參考實作）            −401,408
                                                  ──────────
預期的 xqrq 次數                                    6,336,288     差 0
```

### 指令測試

| 測試 | 預期 |
|---|---|
| `t_xqmac8` | `pass=10024  fail=0` |
| `t_xqrq` | `pass=10037 fail=0`（37 directed + 10,000 randomized） |
| `tests/kernel/run.sh` | 全部與參考實作位元相同 |
| `make bench_run` | 離開碼 0（所有 case 位元相同） |

### microbenchmark（X2，`條/MAC`）

同一個執行檔內，相同輸入下量測 TFLM 參考實作與本專案 kernel：

| case | 參考 | 本專案 | 加速 |
|---|---:|---:|---:|
| 1×1 conv，K=192 | 10.414 | **0.923** | 11.27x |
| 1×1 conv，K=48 | 11.659 | **1.134** | 10.28x |
| depthwise 14×14×240 3×3 | 29.124 | **11.005** | 2.64x |
| depthwise 28×28×96 3×3 | 29.475 | **9.059** | 3.25x |

> **這裡的「參考」與 B0 不是同一個基線。** B0 的參考 kernel 來自
> `libtensorflow-microlite.a`（TFLM 自己的建置旗標）；microbenchmark 裡
> 它被實例化在我們的 translation unit、用 `-O2` 完全內聯。實測差 2.4–4.8
> 倍（1×1 conv 是 50.06 vs 10.41 條/MAC），差別主要在
> `RuntimeShape::Dims()` 有沒有被內聯 —— 沒內聯的話 `Offset()` 的每次
> 呼叫都是真的函式呼叫，而內層迴圈每個 MAC 要付兩次。
>
> 兩個數字都有意義，用途不同：**15.400x** 是對未修改的 TFLM（規格書
> 定義的 B0），**11.27x** 是對同樣編譯情境下的同一份參考演算法。

### 真實圖片（`results/image_check.txt`）

```
LiteRT top-1 = class 652 military uniform    ← 絕對正確性檢查
B0 / B1 / X1 / X2 在 RV64/Spike 上全部 BITEXACT
跨框架（TFLM@x86 vs LiteRT）：CLOSE，1000 個輸出中 6 個差 1 LSB
```

真實圖片輸入下的 `Invoke`：

| config | 真實圖片 | 種子 0 | 差 |
|:--|---:|---:|---:|
| B0 | 19,575,470,856 | 19,575,565,013 | 94,157 |
| B1 | 3,682,217,855 | 3,682,312,012 | 94,157 |
| X1 | 1,590,059,459 | 1,590,153,616 | 94,157 |
| **X2** | 1,271,159,507 | 1,271,170,850 | **11,343** |

前三個配置的差異**完全相同**，因為飽和（clamp）的程式碼在三者中一模一樣
—— 那個輸入相依性來自 clamp 的**資料相依分支**。`xqrq` 把 clamp 收進
指令內部變成無分支，所以 X2 消除了 **88%** 的輸入相依變異。
附帶的效果是執行時間更可預測。

---

## 兩種比較，不要混淆

| 比較 | 容忍度 | 意義 |
|---|---|---|
| **TFLM vs LiteRT**（跨框架） | `CLOSE`（少數 1 LSB） | 不同框架對同一個量化模型的實作差異。**不是驗收條件**，留著當報告材料 |
| **RV64/Spike vs x86**（同一套 C++） | **`BITEXACT`，零容忍** | 這是驗收條件。同一份程式碼在兩個 ISA 上，差一個 bit 就是移植或 kernel 寫錯了 |

所以 `verify_image.sh` 的流程是：先用 LiteRT 確認模型本身正確（唯一的
**絕對**檢查），記錄跨框架差異，然後**把 golden 換成 TFLM@x86 的輸出**，
之後所有比較都是零容忍。

---

## 捨入模式（換 TFLM 版本時必讀）

`tflite::MultiplyByQuantizedMultiplier` 有**兩套實作**，差 1 個 LSB：

- **單次捨入**（`TFLITE_SINGLE_ROUNDING=1`）：64 位元乘、加 round、算術右移
- **雙次捨入**（gemmlowp）：`SaturatingRoundingDoublingHighMul` 在 2^31 處
  捨入，再 `RoundingDivideByPOT` 捨入一次

**本專案的 TFLM（`64b3de8`）是雙次捨入**，`patches/spike/insns/xqrq.h`
照它實作。換 TFLM 版本時如果捨入模式變了，`xqrq` 必須跟著改，否則端到端
會得到 `CLOSE` 而不是 `BITEXACT`。

`t_xqrq` 啟動時會印出 TFLM 被編譯成哪一種，所以不需要靠讀原始碼判斷。
它的 golden **直接呼叫 TFLM 本體**而不是自己轉錄一份 —— 這樣捨入模式的
風險從測試裡消失，Spike 的實作對得上 TFLM，端到端 BITEXACT 就是結構上的
結果而不是運氣。

---

## 交件項目對照

| 項目 | 位置 |
|---|---|
| 1. Source code（C/C++，bare-metal workload） | `tflm/app/`、`tflm/bare/`、`tflm/host/` |
| 2. Model / conversion scripts（Python） | `model/eff-lite0/model.tflite`、`tflm/tools/*.py`、`model/*.py` |
| 3. Spike / custom ISA 實作 | `tflm/patches/spike-xq.patch`、`tflm/patches/spike/insns/` |
| 4. Verification tests（測自訂指令） | `tflm/bare/t_xqmac8.c`、`tflm/bare/t_xqrq.cc` |
| 5. Reproducible benchmark scripts | `tflm/scripts/*.sh`、`make -C bare bench_run` |
| 6. Slides | 另附 |
| 7. README + 預期數字 | 本檔 |

---

## 目錄

```
model/              phase-0 的量化數學參考（Python）與模型檔
  eff-lite0/model.tflite    sha256 bc2ffe19c111...
tests/env/          Spike 環境自檢（tflm/bare/link.ld 衍生自此）
tflm/
  app/              與平台無關的程式
    nn_main.cc        推論主程式（host 與 bare-metal 共用）
    conv_xq.cc        1x1 conv 的特化 kernel（B1 純量 / X1 xqmac8）
    dwconv_xq.cc      depthwise 的特化 kernel（channel 分塊）
    xqrq.h            X2 的再量化指令：打包、呼叫、host 模擬
    prof.cc           per-operator 的指令數收集
  bare/             bare-metal RV64 的平台層與 Makefile
    t_xqmac8.c        xqmac8 的 directed + randomized 測試
    t_xqrq.cc         xqrq 的測試（golden 用 TFLM 本體）
    bench_main.cc     kernel 層的 microbenchmark
  host/             x86 的平台層與 Makefile（快速驗正確性）
  tools/            Python：模型盤點、C 陣列產生、指令組成分析、熱點定位
  scripts/          setup_env.sh 與可重現的量測／驗證腳本
  tests/kernel/     kernel 層的等價性測試（x86，秒級）
  patches/          Spike 的改動（patch + insn 原始碼）
  prebuilt/         medany 建的 libgcc.a（省掉自建 toolchain）
  tflm-patch/       對 TFLM 的兩個改動（debug_log + riscv64 target）
  results/          量測結果（本 README 預期數字的來源）
  gen/              由工具產生，不入版本控制（見 .gitignore）
```

---

## 已知限制

- **效能數字全部是指令數，不是 cycle 數。** Spike 是功能模擬器，沒有
  pipeline 模型，所以 load-use 停頓與記憶體頻寬量不到 —— 那些屬於期末的
  CV-Wally RTL 階段。
- **stem 的 3×3 convolution 沒有優化。** 它是 X2 的最大單一項目
  （521,054,588 條 = 40.9% 的指令，卻只有 2.81% 的 MAC），因為
  `conv_xq` 對非 1×1 一律退回參考
  實作。這是軟體問題不是 ISA 問題：結構上它其實可以用 `xqmac8`。
- **depthwise 的邊界慢路徑沒有優化。** 後段層的邊界像素占到 49%~82%，
  而每個邊界像素比快路徑貴 8 倍。
- 快取未命中未量測（需確認 Spike 是否編入 `--ic/--dc/--l2`）。
- `DW_XQ_BLK` 的分塊寬度未做 sweep（1/4/8/16 都驗證過位元一致，
  但沒有系統地比較）。
- `xqrq` 的 zero-point 相加刻意在 int32 裡環繞，與軟體參考位元相同；
  真實的硬體再量化單元會選飽和。兩者的差異只在再量化結果接近
  `INT32_MAX` 時出現，真實模型碰不到（`t_xqrq` 的 `OFFWRAP1..3` 守住這個行為）。