# 環境細節

README 的「前置需求」照著做就能跑起來。這份是給想知道**為什麼**、
或是遇到連結錯誤要排查的人。

---

## 為什麼只需要一個 5.8 MB 的檔

載入位址是 `0x80000000`（Spike 的預設 `DRAM_BASE`）。標準
`riscv64-unknown-elf` 附的 **libgcc 是 medlow code model 建的**，絕對定址
只到 `0x7fffffff`，連結會得到一堆

```
relocation truncated to fit: R_RISCV_HI20
```

`libm` 可以靠自己實作繞過（`bare/libc_min.cc`），但 **libgcc 繞不開** ——
rv64imc 沒有 FPU，而 `QuantizeMultiplier` 的 `double` 除法一定要 `__divdf3`。

**但缺的只有 libgcc 這一個檔。** `-mcmodel=medany` 本身是編譯器旗標，
標準 toolchain 就支援，而且兩邊都明確帶了它：

- `bare/Makefile`：`-mcmodel=medany`
- TFLM 的 riscv64 target（`tflm-patch/riscv64_generic_makefile.inc`）：
  `PLATFORM_FLAGS` 含 `-mcmodel=$(RISCV_CODE_MODEL)`，而
  `RISCV_CODE_MODEL := medany`

所以倉庫附了一份 medany 建的 libgcc：

```
tflm/prebuilt/libgcc-medany-rv64imc-lp64.a      5.8 MB
```

---

## LIBGCC 的自動偵測

`bare/Makefile` 探測 `$(CXX)` 的**預設 code model**：

| `$(CXX)` 預設 | `prebuilt/` | 結果 |
|---|---|---|
| `medany` | — | `-lgcc`（用 toolchain 自己的） |
| `medlow` | 有 | 自動換成 `prebuilt/` |
| `medlow` | 無 | `-lgcc`，連結前就擋下來並印出補救方式 |
| 探測不出來 | — | 同上，不猜 |

```
$ make -C bare paths | grep -A1 LIBGCC
LIBGCC     = /path/to/repo/tflm/prebuilt/libgcc-medany-rv64imc-lp64.a
             (預設 code model 是 medlow，自動改用 prebuilt/)
```

**為什麼要自動**：`measure.sh`、`instr_mix.sh`、`verify_image.sh` 都是直接
呼叫 `make`，不帶這個變數。靠「記得 `export LIBGCC`」是個會被忘記的步驟，
而忘記的症狀是一串看不懂的重定位錯誤。

要覆寫就明確給（給了之後探測完全不介入）：

```bash
make -C bare CONFIG=X2 LIBGCC=/path/to/your/libgcc.a run
```

---

## 編譯器的解析順序

既不能單純寫死路徑，也不能單純靠 PATH：開發機的 PATH 上還有一套 medlow
的，靠 PATH 會抓錯；但寫死 `$HOME/ca-project/riscv-medany` 的話，任何乾淨
clone 的人（toolchain 在 `/usr/bin`）都會拿到 `Command not found`。

所以用解析順序，第一個命中就用：

| | 條件 | 用 |
|---|---|---|
| 1 | 有給 `RISCV_MEDANY=` | `$(RISCV_MEDANY)/bin/riscv64-unknown-elf-*` |
| 2 | `~/ca-project/riscv-medany` 存在 | 它（本專案開發機） |
| 3 | PATH 上有 `riscv64-unknown-elf-g++` | PATH |
| 4 | 都沒有 | 報錯並指回 README |

`make -C bare paths` 會印出命中哪一條。`setup_env.sh` 的 `check` 階段用
`make -C bare print-cxx` 取得同一個答案 —— 它刻意**不**自己查 PATH，否則會
出現「check 報 ok、建置卻 Command not found」。

---

## 已驗證的範圍

**一、medlow 編譯器 + 附的 medany libgcc**

用預設 code model 為 `medlow` 的標準 toolchain（GCC 16.1.0）編譯、連結本
專案附的 medany libgcc：

```
X2  Invoke 1,271,170,850   BITEXACT
```

與 medany toolchain 編出來的結果**指令數一條不差**（兩邊都明確帶
`-mcmodel=medany`，產生的碼實質相同）。

**二、從 GitHub 乾淨 clone 完整還原**

`git clone` 到一個全新目錄，用獨立的 `WORK` / `RISCV_PREFIX`，完全不碰
開發機原本的環境：

```bash
git clone <repo> /tmp/restoretest
cd /tmp/restoretest/tflm
WORK=/tmp/restoretest-work RISCV_PREFIX=/tmp/restoretest-riscv \
  bash scripts/setup_env.sh all          # exit 0
```

`all` 從零建出 Spike（含 `xqmac8` / `xqrq`）與 TFLM、產生 `gen/`、通過
kernel 等價性與兩個指令測試與完整模型。接著蓋掉 `HOME` 讓解析順序落到
第 3 條（PATH），模擬只有標準 toolchain 的情況：

```
CXX        = riscv64-unknown-elf-g++          （不帶路徑）
             來源：PATH
LIBGCC     = .../prebuilt/libgcc-medany-rv64imc-lp64.a
             (預設 code model 是 medlow，自動改用 prebuilt/)
  Invoke                         1271170850
BITEXACT  1000 個輸出全部位元一致
```

與開發機記錄的數字一條不差。

## 未驗證的範圍

**跨 GCC 版本。** 附的 libgcc 是 GCC 16.1.0 建的；配其他主版本的編譯器
理論上可行（用到的只有 `__divdf3` / `__muldf3` / `__adddf3` / `__clz_tab`
這類由 RISC-V calling convention 定義的常式），但沒有測過。上面兩項驗證
用的 medlow 編譯器也是 16.1.0。

連結出現符號或 ABI 問題時，走下面的備案。

---

## 備案：自建 medany toolchain

```bash
bash scripts/setup_env.sh toolchain       # 數小時
```

參數與本專案完全相同：

```bash
./configure --prefix=$RISCV_MEDANY \
            --with-arch=rv64imc --with-abi=lp64 \
            --with-cmodel=medany --disable-gdb
```

建好之後 `export RISCV_MEDANY=...`，**不要**再設 `LIBGCC`。

`bare/Makefile` 的 `check-cmodel` 會在**連結之前**確認 code model —— 用錯的
話會一路編譯成功，到最後連結才爆出一堆看不出原因的 relocation 錯誤。

---

## 不限 CentOS

專案本身不依賴任何發行版。開發機的 CentOS 7 其實是**最難的那個環境**
（glibc 2.17、系統 g++ 4.8.5 連 C++17 都不支援），換到現代的 Ubuntu /
Debian 會更順。bare-metal 那一側完全不碰 OS —— 它跑在 Spike 裡，連 libc
都是自己實作的（`bare/libc_min.cc`）。

唯一的平台限制是 **`ai-edge-litert` 只有 Linux x86_64 與 macOS 的 wheel**，
所以 `gen` 階段（產生模型的 C 陣列與 LiteRT 的參考輸出）需要這兩種平台之一。

host 編譯器不是 PATH 上的 `g++` 時從命令列指定：

```bash
make -C host CXX=/usr/local/bin/g++ paths
```

`make -C host paths` 會**實際編一個 C++17 的小程式**來檢查，不是只看版本字串。
