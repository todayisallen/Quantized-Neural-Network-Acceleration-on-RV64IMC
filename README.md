# RV64IMC 上的量化神經網路推論加速

高等計算機結構期中專題。  
目標：在 **bare-metal RV64IMC** 環境上執行完整的 EfficientNet-Lite0（W8A8 量化，0.385 GMAC），透過 Spike 模擬器量測指令數、分析瓶頸，並設計兩條自訂指令來加速。最終驗證了每一步的輸出都與軟體參考結果達到**位元完全相同（BITEXACT）**。

### 專題重點成果一覽

1. **B0 未修改 TFLM 參考實作**：19,575,565,013 條指令（1.000x / 50.82 inst/MAC）
2. **B1 軟體優化（1x1 conv + depthwise）**：3,682,312,012 條指令（5.316x / 9.56 inst/MAC）
3. **X1 + xqmac8（8 路 packed int8 MAC）**：1,590,153,616 條指令（12.310x / 4.13 inst/MAC）
4. **X2 + xqrq（再量化自訂指令）**：1,271,170,850 條指令（15.400x / 3.30 inst/MAC）

全部配置在 x86 與 RV64/Spike 上皆通過 **BITEXACT** 驗證，包含真實圖片測試（LiteRT top-1 輸出為 class 652 military uniform）。

---

## 1. 前置需求與環境準備

先把倉庫抓下來，再執行環境檢查：

```bash
git clone <this-repo> ca-project-repo && cd ca-project-repo/tflm
```

檢查指令：

```bash
bash scripts/setup_env.sh check
```

這個檢查幾秒內就會完成，不會主動安裝任何軟體，只是幫忙確認系統缺哪些工具。

### 環境需求清單

1. **作業系統**：Linux x86_64 或 macOS。
2. **系統基本工具**：`git` / `make` / `g++`（需支援 C++17）/ `python3` / `dtc`（需自行安裝）。
3. **RISC-V Toolchain**：`riscv64-unknown-elf-*` 標準 bare-metal 版本即可（需自行安裝）。
4. **Spike**：`riscv-isa-sim` (commit `f11c7b6`) + `patches/spike-xq.patch`（腳本自動安裝）。
5. **TFLM**：`tflite-micro` (commit `64b3de8`) + `tflm-patch/`（腳本自動安裝）。
6. **Python 套件**：`ai-edge-litert 1.4.0` / `numpy 2.2.6` / `pillow 12.2.0`（腳本自動安裝）。
7. **模型檔**：`model/eff-lite0/model.tflite`（已內建於倉庫中）。

`setup_env.sh` 全程不需要 root 權限，安裝的內容都在使用者指定的路徑下；需自行安裝的第 2、3 項多半需要 root，因此腳本不代為處理。

### 系統套件與 Toolchain 安裝說明

基本工具可透過以下命令安裝：

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y git make g++ python3-pip device-tree-compiler autoconf automake libtool

# Fedora / RHEL 9+
sudo dnf install -y git make gcc-c++ python3-pip dtc autoconf automake libtool

# macOS
brew install dtc autoconf automake libtool
```

至於 RISC-V toolchain，**請特別注意一定要用 bare-metal 的 `riscv64-unknown-elf`**：

```bash
# Debian / Ubuntu 直接安裝
sudo apt install -y gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf
```

其他平台可直接下載官方預編譯 Release，解壓後將 `bin/` 目錄加入 PATH 即可：
<https://github.com/riscv-collab/riscv-gnu-toolchain/releases>

> **請不要使用 `riscv64-linux-gnu`。** 那是給 Linux user-space 的三元組，而 `bare/Makefile` 需要的是 `-unknown-elf`。Fedora 主倉庫只提供前者，請改用上面的預編譯 Release。

`bare/Makefile` 的編譯器解析順序如下（第一個命中者即採用，`make -C bare paths` 會印出命中哪一條）：

1. 有指定 `RISCV_MEDANY=` 時，使用 `$(RISCV_MEDANY)/bin/riscv64-unknown-elf-*`。
2. `~/ca-project/riscv-medany` 存在時使用它（本專案開發機）。
3. PATH 上有 `riscv64-unknown-elf-g++` 時使用 PATH。
4. 都沒有則報錯並指回本節。

因此 toolchain 安裝在 `/usr/bin` 即可直接使用，不需要設定任何環境變數。

### 注意事項

1. **不用特地自己編譯 Toolchain**：標準版預設 code model 是 `medlow`，無法連結到 `0x80000000`。我們在倉庫裡附了一份用 `medany` 編譯的 `libgcc`（`tflm/prebuilt/libgcc-medany-rv64imc-lp64.a`），`bare/Makefile` 只要偵測到 `medlow` 就會自動載入它，省下好幾個小時編譯 toolchain 的時間。
2. **特定的 Commit 是重現關鍵**：`MultiplyByQuantizedMultiplier` 的捨入邏輯會隨著 TFLM 版本更新而變動，而我們的自訂指令 `xqrq` 必須與它保持 bit-level 對齊，因此我們鎖定了此 commit。
3. **更換編譯器時記得清空快取**：如果更換了 Toolchain 或 `libgcc`，請務必先執行 `rm -rf bare/build`。`make` 只比對檔案時間戳記，沿用舊的物件檔連結出來的結果不會對應當前的建置設定，而且不會有任何警告。

---

## 2. 快速開始流程

要重現我們的實驗結果，可以直接執行環境建置腳本：

```bash
cd ca-project-repo/tflm          # 若尚未切進去
bash scripts/setup_env.sh all
```

執行 `setup_env.sh all` 時，系統會依序完成以下 5 個步驟（每步執行完後都會進行自我驗證）：

1. **`python`（約 2 分鐘）**：安裝 `ai-edge-litert`、`numpy` 與 `pillow`。
2. **`spike`（約 15 分鐘）**：複製 `riscv-isa-sim`、套用本專案特製 patch、編譯並完成安裝。
3. **`tflm`（約 10 分鐘）**：複製 `tflite-micro`、套用 `tflm-patch/`，建置 x86 與 riscv64 兩個函式庫。
4. **`gen`（約 5 分鐘）**：產生模型的 C 語言陣列、固定輸入資料與校正 golden 值。
5. **`verify`（約 5 分鐘）**：執行 kernel 等價性測試、自訂指令測試以及完整模型的 BITEXACT 驗證。

全部流程大約需要 **35 分鐘**。如果你只想重跑特定階段，可以使用單獨指令：

```bash
bash scripts/setup_env.sh spike      # 只重建 Spike
bash scripts/setup_env.sh verify     # 只重跑驗證
```

環境準備好後即可進行量測。所有結果都會寫進 `results/`，本檔的數字全部出自那裡。

### 量測腳本

**指令數對照（`measure.sh`）**

```bash
bash scripts/measure.sh B0 B1 B1C B1D X1 X2   # 六種配置，約 5 分鐘
bash scripts/measure.sh X2                    # 只量單一配置
bash scripts/measure.sh --summary-only         # 不重跑，只從既有結果重產對照表
```

產出 `results/<CONFIG>.txt`（完整輸出，含每個 operator 的指令數）、
`results/<CONFIG>.size.txt`（section 大小）與 `results/summary.md`（§6.1 的對照表）。

**指令組成（`instr_mix.sh`）**

```bash
bash scripts/instr_mix.sh X2
```

產出 `results/<CONFIG>.mix.txt`，內容是指令分類、使用頻率前 15 名、各函式的指令組成。

這個腳本靠 Spike 的 `-g` PC histogram，所以 Spike 必須是用 `--enable-commitlog`
編的；另外需要 `tests/env/build/t01_plain`，第一次請先跑 `bash tests/env/run.sh`（應得 21/21）。
執行時間與指令數成正比，實測：X1 約 30 秒、B1 約 1 分鐘、**B0 約 6 分鐘**。

**真實圖片驗證（`verify_image.sh`）**

```bash
bash scripts/verify_image.sh B0 B1 X1 X2      # 約 5 分鐘
```

產出 `results/image_check.txt`：LiteRT 的 top-1、跨框架比對，以及各配置在 x86 與
RV64 上的 BITEXACT 結果與真實圖片下的指令數。

**Kernel microbenchmark**

```bash
for C in B1 X1 X2; do make -C bare CONFIG=$C bench_run; done
```

只輸出到畫面，不寫檔。比較 TFLM 參考 kernel 與本專案 kernel 在同一個執行檔內的指令數。

**單獨跑一個配置**

```bash
make -C bare CONFIG=X2 run        # 完整模型，印出 Invoke 與每層指令數
make -C bare paths                # 印出實際使用的編譯器、Spike、TFLM 路徑
```

### 配置代號

| CONFIG | 內容 |
|:--|:--|
| `B0` | 未修改的 TFLM 參考實作 |
| `B1C` | 只開 1×1 conv 的優化 |
| `B1D` | 只開 depthwise 的優化 |
| `B1` | 兩項軟體優化都開 |
| `X1` | B1 + `xqmac8` |
| `X2` | X1 + `xqrq` |

---

## 3. 常見問題：沒套用 Spike Patch 的狀況

`xqmac8` 與 `xqrq` 是我們自訂的新指令，標準版的 Spike 並不認識它們。

如果使用沒有套用 patch 的 Spike 執行 X1/X2，Spike 遇到未知指令時會觸發 illegal instruction trap。由於 bare-metal 沒有掛載 trap handler，預設會跳回 `0x0`，進而引發無窮 trap 迴圈。

* **典型症狀**：畫面上印出前幾行文字後就完全卡住、沒有任何錯誤訊息，一直跑到 timeout。
* **確認方式**：

```bash
# 確認使用的是哪一支 Spike
which spike

# 確認 patch 有套上 —— 這兩個檔案只會被 patch 修改
grep -c MATCH_XQRQ <riscv-isa-sim>/riscv/encoding.h
grep -c xqrq       <riscv-isa-sim>/riscv/riscv.mk.in

# 最直接的方式：執行自訂指令測試
make -C bare t_xqrq_run
```

> `spike --help` 列出的是命令列選項，不會列出支援的指令，因此無法用它判斷 patch 是否套上。

---

## 4. 分階段手動建置細節

如果你想逐步了解專案建置過程，可以按照以下 4 個步驟進行：

### 步驟 1：建立並 Patch Spike

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim && git checkout f11c7b6
git apply <repo>/tflm/patches/spike-xq.patch
mkdir -p build && cd build
../configure --prefix=$HOME/ca-project/riscv --enable-commitlog --without-boost-asio
make -j8 && make install
```

> **注意**：`--enable-commitlog` 絕對不能省，因為我們的 `scripts/instr_mix.sh` 需要利用 Spike 的 `-g` PC histogram 功能來進行指令組成分析。

`spike-xq.patch` 的改動主要分為三個部分：

1. **新增指令實作檔**：加入 `riscv/insns/xqtest.h`、`xqmac8.h` 與 `xqrq.h`，作為三條自訂指令的核心行為邏輯。
2. **補充編碼定義**：修改 `riscv/encoding.h`，定義 `MATCH_XQMAC8` 與 `MATCH_XQRQ` 等指令 Opcode 及 Mask 匹配規則。
3. **註冊至 Spike 解碼器**：修改 `riscv/riscv.mk.in`，將新指令納入編譯清單，確保 Spike 能正確解碼並執行。

另外，`tflm/patches/spike/insns/` 存放了這三個 `.h` 的獨立副本，程式碼與 Patch 新增的內容完全一致。這樣設計主要有兩個好處：

* **方便閱讀與審查**：即便不套用 Patch，也能直接查閱自訂指令的實作細節（對應交件第 3 項）。
* **容錯後備機制**：當 `git apply` 因環境差異失敗時，可以作為備用檔案直接手動複製覆蓋；`setup_env.sh` 在 `spike` 建置階段即採用此備援邏輯。

### 步驟 2：建立並 Patch TFLM

```bash
git clone https://github.com/tensorflow/tflite-micro.git
cd tflite-micro && git checkout 64b3de8
MK=tensorflow/lite/micro/tools/make/Makefile
make -f $MK third_party_downloads

# 套用專案特製的 tflm-patch
mkdir -p tensorflow/lite/micro/riscv64_generic
cp <repo>/tflm/tflm-patch/debug_log.cc tensorflow/lite/micro/riscv64_generic/
cp <repo>/tflm/tflm-patch/riscv64_generic_makefile.inc tensorflow/lite/micro/tools/make/targets/

make -f $MK TARGET=linux TARGET_ARCH=x86_64 microlite -j8
make -f $MK TARGET=riscv64_generic TARGET_ARCH=riscv64 microlite -j8

export TFLM_ROOT=$PWD
```

我們在這裡處理了兩個關鍵問題：

1. **`debug_log.cc`**：原版程式依賴 `vfprintf(stderr, ...)`，這在 bare-metal 上會拉入龐大的 C 函式庫。我們將其替換為輕量化的 vsnprintf 格式化後直接輸出至 HTIF。
2. **`riscv64_generic_makefile.inc`**：修正 TFLM 預設只有 RV32 設定的問題，將目標調整為 `rv64imc_zicsr` / `lp64` / `medany`。

> **小提醒**：`TARGET_ARCH=riscv64` 必須從命令列傳入。Makefile 內部在載入標的設定檔之前就已經決定了輸出目錄路徑，不傳會生成到錯誤的資料夾名稱（`..._x86_64_...` 而非 `..._riscv64_...`）。

### 步驟 3：產生模型資料庫 (gen/ 目錄)

```bash
cd <repo>/tflm
MODEL=../model/eff-lite0/model.tflite

python3 tools/model_info.py  $MODEL --out gen
python3 tools/gen_model_c.py $MODEL --out gen
python3 tools/gen_io.py      $MODEL --out gen --seed 0

make -C host CONFIG=B0 run
python3 tools/gen_golden_from_bin.py host/tflm_out.bin --out gen --config B0
find gen -type f -exec touch {} +
```

後續的腳本會把 golden 資料從 LiteRT 轉為 TFLM@x86 的輸出。請記得執行 `touch`，讓生成檔案的時間戳記更新，避免 `make` 判斷檔案沒變而跳過編譯。

### 步驟 4：執行驗證測試

```bash
make -C bare paths
make -C host paths

bash tests/kernel/run.sh
make -C bare t_xqrq_run
make -C bare CONFIG=X2 run
```

---

## 5. 模型架構與算子統計

我們使用的是 `model/eff-lite0/model.tflite` (W8A8 全整數量化模型)，其基本參數如下：

1. **輸入規格**：`images` [1, 224, 224, 3] `uint8`（scale 0.012566016986966133，zero_point 131）
2. **輸出規格**：`Softmax` [1, 1000] `uint8`（scale 0.00390625，zero_point 0）
3. **運算量**：385,187,552 MACs（共 64 個 op，8 種 operator 種類）
4. **浮點張量數**：0（推論全程皆為整數運算）
5. **量化型別**：所有 weight 與 activation 皆為 `int8`，bias 與 per-channel 量化參數為 `int32`；僅輸入輸出兩個張量是 `uint8`，由 op 0 與 op 63 的 `QUANTIZE` 橋接。

前四點可以藉由執行 `python3 tools/model_info.py model/eff-lite0/model.tflite` 直接從模型讀出；第 5 點是逐一列出全部張量的 dtype 得到的結果（全部張量中只有上述兩個 I/O 不是 `int8` 或 `int32`）。

### 模型架構

| 階段 | op idx | 輸出 shape | 說明 |
|:--|:--|:--|:--|
| 輸入再量化 | 0 | 224×224×3 | uint8 → int8 |
| **stem 3×3 conv** | 1 | 112×112×32 | stride 2，**`CONV_2D` 裡唯一的非 1×1** |
| MBConv ×1 | 2–3 | 112×112×16 | DW 3×3，無展開，只有 DW → 投影 |
| MBConv ×2 | 4–10 | 56×56×24 | DW 3×3，stride 2，1 個殘差 |
| MBConv ×2 | 11–17 | 28×28×40 | **DW 5×5**，stride 2，1 個殘差 |
| MBConv ×3 | 18–28 | 14×14×80 | DW 3×3，stride 2，2 個殘差 |
| MBConv ×3 | 29–39 | 14×14×112 | **DW 5×5**，2 個殘差 |
| MBConv ×4 | 40–54 | 7×7×192 | **DW 5×5**，stride 2，3 個殘差 |
| MBConv ×1 | 55–57 | 7×7×320 | DW 3×3 |
| head 1×1 conv | 58 | 7×7×1280 | |
| global average pool | 59 | 1×1×1280 | |
| reshape | 60 | 1280 | |
| fully connected | 61 | 1000 | |
| softmax | 62 | 1000 | |
| 輸出再量化 | 63 | 1000 | int8 → uint8 |

每個 MBConv 區塊的結構是「1×1 展開 → depthwise → 1×1 投影」，當輸入輸出形狀相同時會多一個 `ADD` 殘差。depthwise 的 kernel 尺寸依 stage 而異：16 層裡有 7 層 3×3、9 層 5×5，順序為 3,3,5,3,5,5,3（與 EfficientNet-B0 相同）。解析度總共下降五次：224 → 112 → 56 → 28 → 14 → 7（發生在 op 1、5、12、19、41）；通道走向為 32 → 16 → 24 → 40 → 80 → 112 → 192 → 320 → 1280。

### Operator 種類統計與優化策略

我們針對模型中的算子進行了統計分析：

1. **`CONV_2D` (共 33 次)**：我們發現這 33 個卷積中，有 32 個都是 **1x1 conv**（16 個區塊的展開與投影，扣掉第一個區塊沒有展開，再加上 head 的 1x1）。因此我們特別撰寫了 `conv_xq` 來全面加速 1x1 卷積；唯一一個非 1x1 卷積是第一層的 3x3 stem conv，這部分我們不進行優化。
2. **`DEPTHWISE_CONV_2D` (共 16 次)**：全部導向我們優化的 `dwconv_xq` 核心處理。
3. **`ADD` (9 次)** 與 **其他運算子 (6 次)**：包含兩端的 `QUANTIZE`、pool、reshape、FC 與 softmax，這些算子佔總執行時間低於 3%，因此維持原樣不另外修改。

---

## 6. 實驗結果與性能數據

### 6.1 各配置執行指令數對照

測試條件為固定隨機種子 0，量測單位為已退休指令數 (`instret`)：

| 配置 | Invoke 指令數 | 相對 B0 加速比 | inst/MAC | AllocateTensors | 框架 Overhead | `.text` 區段大小 | 結果判定 |
|:--|---:|---:|---:|---:|---:|---:|:--|
| **B0** | 19,575,565,013 | 1.000x | 50.82 | 15,058,298 | 4,473 | 121,760 | BITEXACT |
| **B1C** | 5,281,442,410 | 3.706x | 13.71 | 15,058,298 | 4,473 | 117,440 | BITEXACT |
| **B1D** | 17,976,434,615 | 1.089x | 46.67 | 15,058,298 | 4,473 | 121,560 | BITEXACT |
| **B1** | 3,682,312,012 | 5.316x | 9.56 | 15,058,298 | 4,473 | 117,240 | BITEXACT |
| **X1** | 1,590,153,616 | 12.310x | 4.13 | 15,058,298 | 4,473 | 117,436 | BITEXACT |
| **X2** | 1,271,170,850 | 15.400x | 3.30 | 15,058,298 | 4,473 | 116,856 | BITEXACT |

#### 數據分析：

1. **獨立優化可加性**：B1C（僅開 1x1）與 B1D（僅開 depthwise）節省的指令數分別為 `14,294,122,603` 與 `1,599,130,398`。兩者相加精準等於 B1 節省的總指令數，證明兩項優化處理的運算完全沒有重疊。
2. **無浮點運算**：`Invoke` 執行期間浮點指令數為 0。浮點運算僅出現在 `AllocateTensors` 階段（`__muldf3` 4,385,796 條 + `__divdf3` 2,118,476 條，佔該階段 43%），用於初始化時計算乘數，後續推論的時候都是整數運算。
3. **程式碼體積縮減**：X2 配置的 `.text` 區段比 B0 還小了 4,904 bytes（4.9 KB，−4.0%）。因為自訂指令 `xqrq` 把原本需要多道指令組合的 clamp 與 offset 運算塌縮成單一指令，不僅變快，程式碼也變輕量了。
4. **健全性檢查**：`AllocateTensors` 與框架 Overhead 在六種配置中完全相同 —— B1/X1/X2 都沒有改動 Prepare 階段，本來就應該一致。

### 6.2 X2 配置的指令組成分析

在最終的 X2 配置中，指令分布如下：

1. **Load 指令**：283,551,893 (21.60%)
2. **Store 指令**：27,207,345 (2.07%)
3. **Mul/Div 指令**：78,225,289 (5.96%)
4. **ALU 指令**：518,188,400 (39.47%)
5. **Branch 分支指令**：352,213,721 (26.83%)
6. **Jump 跳躍指令**：4,870,788 (0.37%)
7. **Custom 自訂指令**：48,653,472 (3.71%)

總計 1,312,935,133 條（上列七類之外另有 CSR 264 條與未分類 23,961 條，主要是 `zext.b`），其中壓縮指令 (RVC) 482,327,552 條（36.74%）。此處的 histogram 涵蓋整個程式（含 `AllocateTensors` 與啟動碼），因此總數略高於 `Invoke`。

衍生指標（同樣出自 `results/<CONFIG>.mix.txt`）：

| 指標 | B0 | B1 | X1 | X2 |
|:--|---:|---:|---:|---:|
| MAC / instruction | 0.0196 | 0.1034 | 0.2360 | **0.2934** |
| instruction / MAC | 50.93 | 9.67 | 4.24 | **3.41** |
| load+store / MAC | 10.98 | 2.42 | 0.87 | **0.81** |
| RVC 比例 | 26.40% | 41.10% | 43.02% | 36.74% |

`load+store / MAC` 從 10.98 降到 0.81（13.6 倍）是記憶體指令開銷的直接指標：
B0 每個 MAC 要將近 11 次記憶體存取，因為 `Offset()` 每個 MAC 都重算兩次位址、
每次重新載入；改成指標行走後降到 2.42，兩條自訂指令再把它壓到 0.81。

RVC 比例在 X2 反而下降，是因為自訂指令以 `.insn` 編碼、固定 32-bit 不可壓縮，
而它取代掉的是一串可壓縮的指令。

> **自訂指令驗證**：X2 觸發的自訂指令中，`xqmac8` 占 42,317,184 次，`xqrq` 占 6,336,288 次。我們比對了模型結構理論計算出的輸出元素數量（6,737,696 個），扣除掉走參考實作的 stem 層（401,408 個）後，理論值剛好也是 `6,336,288` 次，落差為 **0**。

### 6.3 各主要 operator 的執行貢獻

X2 配置下，把 64 個 op 依類型加總（資料出自 `results/X2.txt` 的 per-operator 表）：

| operator | 次數 | 指令數 | 占 `Invoke` |
|:--|--:|---:|---:|
| `CONV_2D` | 33 | 900,744,234 | 70.9% |
| `DEPTHWISE_CONV_2D` | 16 | 331,268,350 | 26.1% |
| `ADD` | 9 | 25,117,928 | 2.0% |
| `FULLY_CONNECTED` | 1 | 11,581,083 | 0.9% |
| `AVERAGE_POOL_2D` | 1 | 1,160,393 | 0.1% |
| `QUANTIZE` | 2 | 909,599 | 0.1% |
| `SOFTMAX` | 1 | 383,749 | 0.0% |
| `RESHAPE` | 1 | 1,041 | 0.0% |
| 合計 | 64 | 1,271,166,377 | 100.0% |

逐層加總與 `Invoke` 1,271,170,850 相差 **4,473 條**，正好等於 §6.1 的框架 overhead。

如果把 `CONV_2D` 拆開來看，會發現一個非常不直覺的對比：

| 算子類別 | 執行指令數 | 占 `Invoke` 比例 | 運算量 (MAC) 占比 |
|:--|---:|---:|---:|
| **op 1 的 3×3 stem**（走參考實作） | 521,054,588 | **41.0%** | 2.81% |
| **其餘 32 個 1×1 conv**（走 `conv_xq` + 自訂指令） | 379,689,646 | 29.9% | **87.9%** |

**單獨一個完全沒優化的 3×3 卷積，產生的指令數竟然比全部 32 個優化過的 1×1 卷積加起來還要多。**

這個數據呈現出極度強烈的對比：
* **3×3 stem** 雖然只占了全模型 2.81% 的 MAC，卻因為走 TFLM 的預設軟體參考實作（包含了大量未內聯的通道尋址、邊界檢查與純量迴圈），吃掉了高達 41% 的總指令數。
* **32 個 1×1 卷積** 承載了接近九成的總運算量（338,537,472 MAC，87.9%），但在打包記憶體存取以及 `xqmac8` / `xqrq` 自訂指令的雙重加速下，總指令數直接被壓到 30% 以下。

這也直接點出了我們目前架構的下一個最大瓶頸——這並非 ISA 或硬體的極限，純粹是軟體層面的問題。第一層 stem 在結構上同樣能套用打包矩陣乘法與自訂指令加速，是後續最佳化最顯著的潛力點。

### 6.4 三大瓶頸

依量測結果排序：

1. **記憶體指令開銷** —— B0 的 `load+store / MAC` 是 10.98，因為 TFLM 參考實作的
   `Offset()` 每個 MAC 要重算兩次位址、每次重新載入。改成指標行走後降到 2.42，
   兩條自訂指令再壓到 **0.81**（13.6 倍）。**已解決。**
2. **再量化開銷** —— 每個輸出元素都要「定點乘 → 捨入 → 位移 → 加 zero-point →
   飽和」。X2 用 `xqrq` 把這一串收成一條指令，省下 318,982,766 條（X1 → X2）。
   **已解決。**
3. **未特化的 3×3 卷積** —— 上表的 41.0%。`conv_xq` 只特化 1×1，其餘一律退回參考
   實作。這是軟體問題不是 ISA 問題：因為結構上它也可以改寫成用 `xqmac8`。而這也是之後可以優化的部分

另外 B0 的分支指令占 38.07%（7,468,861,913 條），比 ALU 的 26.58% 還多 —— 一個
MAC 密集的 workload 裡分支比算術多，來自參考 kernel 的迴圈與邊界檢查。B1 把它
降到 684,650,331 條（10.9 倍）。

### 6.5 Microbenchmark 與真實圖片測試

在單獨 Kernel 效能測試上（比較參考 kernel 與本專案優化 kernel）：

1. **1x1 Conv (K=192)**：從 10.414 inst/MAC 降至 **0.923 inst/MAC** (11.27x 加速)
2. **1x1 Conv (K=48)**：從 11.659 inst/MAC 降至 **1.134 inst/MAC** (10.28x 加速)
3. **Depthwise (14x14x240)**：從 29.124 inst/MAC 降至 **11.005 inst/MAC** (2.64x 加速)
4. **Depthwise (28x28x96)**：從 29.475 inst/MAC 降至 **9.059 inst/MAC** (3.25x 加速)

> **此處的「參考」與 B0 不是同一個基線。** B0 的參考 kernel 來自 `libtensorflow-microlite.a`（使用 TFLM 自己的建置旗標）；microbenchmark 裡則是把同一份參考演算法實例化在我們的 translation unit、以 `-O2` 完全內聯，因此快了 2.4–4.8 倍。兩個數字各有意義：**15.400x** 是相對於未修改的 TFLM（規格書定義的 B0），**11.27x** 則是相對於同樣編譯情境下的同一份參考演算法。

#### 真實圖片數據：

使用軍服照片測試時，各配置的 `Invoke` 與固定種子輸入的差異如下：

| 配置 | 真實圖片 | 種子 0 | 差異(指令數) |
|:--|---:|---:|---:|
| B0 | 19,575,470,856 | 19,575,565,013 | 94,157 |
| B1 | 3,682,217,855 | 3,682,312,012 | 94,157 |
| X1 | 1,590,059,459 | 1,590,153,616 | 94,157 |
| **X2** | 1,271,159,507 | 1,271,170,850 | **11,343** |

實測發現指令數會隨輸入改變，主因是傳統 clamp 帶有條件分支：數值超出範圍被夾住時才會多跑指令，被夾幾次完全看資料。前三個配置的變動量完全一樣，是因為輸出 bit 完全對齊、夾住的次數相同，這剛好能拿來當 sanity check。

X2 直接把 clamp 併入 xqrq 指令內部，不管輸入什麼資料都是單一指令完成。我們逐層核對 64 個 operator，發現 32 個 1×1 conv 與 16 個 depthwise 在兩種輸入下的指令數完全相同（差距為 0）。全模型唯一還有變動的卷積，只有走參考實作的 op 1（3×3 stem，少了 34,700 條）；其餘微幅變動則來自 9 個 ADD（合計 +23,000）、pool（+513）、FC（+8）與 softmax（−164）。正負相抵後淨差 11,343 條，成功消除了 88% 的輸入資料波動。換句話說，只要經過自訂指令優化的卷積層，執行指令數已經做到完全決定性（deterministic），不再受輸入資料影響。

---

## 7. 比較基準與捨入模式說明

在進行驗證時，我們定義了兩種不同嚴格度的比較：

1. **TFLM vs LiteRT（跨框架）**：允許極少數結果有 1 LSB 的微小落差（實測 1000 個輸出中有 6 個，不一致率 0.6%），這是由於不同框架底層浮點轉整數的實作細節不同所致。
2. **RV64/Spike vs x86（同一套 C++）**：**BITEXACT**。相同的 C++ 原始碼在不同的 ISA 上執行，結果必須完全一致。

另外，TFLM 內部支援「單次捨入」與「雙次捨入」兩種模式。我們採用的 TFLM預設使用 **gemmlowp 雙次捨入**，因此我們在 `spike-xq.patch` 中的 `xqrq` 邏輯也是完全對齊雙次捨入的行為。若未來要更換 TFLM 版本，需要特別留意這一點。

### 驗證測試預期結果

驗證分成三個層級：**單一指令**（`t_xqmac8` / `t_xqrq`）、**kernel**（`tests/kernel/run.sh`、`bench_run`）、**完整模型**（`CONFIG=X2 run`、`verify_image.sh`）。

| 測試 | 測什麼 | 預期輸出 |
|:--|:--|:--|
| `t_xqmac8` | **`xqmac8` 的指令語意**：8 個 lane 各自的 sign-extend、累加結果、以及 ±127 / −128 這類邊界值。directed 案例加上大量隨機案例，golden 由 C 直接算出。 | `pass=10024  fail=0` |
| `t_xqrq` | **`xqrq` 的指令語意**：捨入模式是否與 TFLM 一致、gemmlowp 內部朝零截斷的除法、offset 在 int32 相加時的環繞、descriptor 的打包與解包、int32 飽和、clamp 上下界。golden 直接呼叫 TFLM 本體的 `MultiplyByQuantizedMultiplier`。 | `pass=10037  fail=0`（37 directed + 10,000 randomized） |
| `tests/kernel/run.sh` | **kernel 等價性**：`conv_xq` 與 `dwconv_xq` 兩個改寫過的 kernel，輸出是否與 TFLM 參考實作位元相同。跑在 x86、秒級，不需要 Spike，是改 kernel 之後最快的回歸測試。 | 全部與參考實作位元相同 |
| `make bench_run` | **microbenchmark 的正確性**：量指令數之外，每個 case 也把兩邊的輸出逐 byte 比對，所以它同時是效能量測與正確性檢查。 | 離開碼 0（所有 case 位元相同） |
| `make -C bare CONFIG=X2 run` | **完整模型**：整個 EfficientNet-Lite0 在 RV64/Spike 上跑完，1000 個輸出與 x86 的 golden 比對。 | `BITEXACT  1000 個輸出全部位元一致` |
| `bash scripts/verify_image.sh B0 B1 X1 X2` | **真實圖片的絕對正確性**：LiteRT 的 top-1 是否符合該圖的已知標籤（不依賴另一個實作的唯一檢查）、跨框架差異、四個配置在 x86 與 RV64 兩個 ISA 上的一致性。 | `結果: 通過` |

前兩項對應規格 §3 的「Verify results against a software reference」與 §5 的
「Xqmac passes directed and randomized verification tests」。

這兩個指令測試可以透過 `bash scripts/setup_env.sh verify` 一併完成，若想針對單一指令除錯，也可以獨立執行：

```bash
cd <repo>/tflm

# 1. 執行 t_xqrq 測試
make -C bare t_xqrq_run

# 2. 執行 t_xqmac8 測試（未包進 Makefile，需直接編譯並交給 Spike 執行）
riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
  -O2 -nostdlib -nostartfiles \
  -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns \
  -T ../tests/env/link.ld \
  bare/t_xqmac8.c -o /tmp/t_xqmac8
spike --isa=rv64imc_zicsr -m64 /tmp/t_xqmac8 </dev/null | tail -3
```

`-fno-tree-loop-distribute-patterns` 不能省：`t_xqmac8.c` 裡有一個填值迴圈，GCC 會把它認成 memset 的 pattern 而產生對 `memset` 的呼叫，但 `-nostdlib` 沒有 libc，連結時就會 `undefined reference to 'memset'`。

---

## 8. 專案目錄結構與交件對照

### 交件項目對照表

1. **Source Code (bare-metal)**：`tflm/app/`、`tflm/bare/`、`tflm/host/`
2. **Model & Conversion Tools**：`model/eff-lite0/model.tflite`、`tflm/tools/*.py`
3. **Spike / Custom ISA 實作**：`tflm/patches/spike-xq.patch`、`tflm/patches/spike/insns/`（指令編碼與語意見 §9）
4. **Verification Tests**：`tflm/bare/t_xqmac8.c`、`tflm/bare/t_xqrq.cc`
5. **Benchmark Scripts**：`tflm/scripts/*.sh`、`make -C bare bench_run`
6. **README 與預期效能數據**：`README.md`

### 目錄結構

```
model/             模型檔案與 Phase-0 驗證腳本
tests/env/         Spike 環境自檢（tflm/bare/link.ld 衍生自此）
tflm/
├── app/           與平台無關的核心演算法 (conv_xq.cc, dwconv_xq.cc, xqrq.h)
├── bare/          RV64 Bare-metal 平台層與測試 Makefile
├── host/          x86 平台層（用於快速開發驗證）
├── tools/         模型分析、C 陣列生成與指令分析工具
├── scripts/       自動化建置與測試腳本 (setup_env.sh, measure.sh 等)
├── tests/kernel/  x86 Kernel 等價性測試
├── patches/       Spike 自訂指令 Patch 檔與 insn 原始碼
├── prebuilt/      medany 編譯的 libgcc.a（省去自建 toolchain）
├── tflm-patch/    對 TFLM 的兩項修改（debug_log + riscv64 target）
├── results/       詳細測試數據與紀錄檔（本檔所有數字的來源）
└── gen/           由工具產生，不納入版本控制
```

---

## 9. 自訂指令規格（編碼與語意）

### 9.1 編碼

三條指令都放在 **custom-0**（opcode `0x0b` = `0001011`），使用標準 R-type 欄位：

```
 31      25 24   20 19   15 14  12 11   7 6        0
┌──────────┬───────┬───────┬──────┬──────┬──────────┐
│  funct7  │  rs2  │  rs1  │funct3│  rd  │  opcode  │
│    0     │       │       │      │      │   0x0b   │
└──────────┴───────┴───────┴──────┴──────┴──────────┘
```

| 指令 | funct3 | MATCH | MASK |
|:--|--:|:--|:--|
| `xqtest` | 0 | `0x0000000b` | `0xfe00707f` |
| `xqmac8` | 1 | `0x0000100b` | `0xfe00707f` |
| `xqrq` | 2 | `0x0000200b` | `0xfe00707f` |

MASK `0xfe00707f` 蓋住 funct7(31:25)、funct3(14:12) 與 opcode(6:0)，因此 funct7
固定為 0，同一個 opcode 下還有 127 個 funct7 編碼空間可用。`xqtest` 是建置初期
用來確認解碼器接線正確的 smoke test，不參與推論。

kernel 端用 `.insn` 指令組合發出，不需要修改 binutils：

```c
__asm__(".insn r 0x0b, 1, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(w));  // xqmac8
__asm__(".insn r 0x0b, 2, 0, %0, %1, %2" : "=r"(out) : "r"(acc), "r"(desc)); // xqrq
```

### 9.2 `xqmac8 rd, rs1, rs2`

8 路 packed int8 乘加：

```
rd += Σ(i=0..7) sext8(rs1[8i+7:8i]) × sext8(rs2[8i+7:8i])
```

- **`rs1`**：8 個 int8 activation，packed 在 64 bit 裡
- **`rs2`**：8 個 int8 weight，同樣的 packing
- **`rd`**：累加器，**read-modify-write** —— 讀取舊值、加上這 8 項、寫回
- **累加器寬度：XLEN（64 bit）**

選 64 bit 的理由：`rd` 本來就是整數暫存器，寬度是免費的，而且徹底排除溢位
考量。本模型最深的 1×1 卷積 K = 1152，int8×int8 的最大絕對值是 127 × 128 =
16,256，乘上 1152 只有 18,726,912，連 int32 都用不滿 —— 但 64 bit 讓這條指令
不必對 K 設上限。

### 9.3 `xqrq rd, rs1, rs2`

把再量化的五個步驟收成一條指令。

- **`rs1`**：int32 累加器（取低 32 bit）
- **`rs2`**：64 bit descriptor，欄位配置如下
- **`rd`**：int8 結果，sign-extend 到 XLEN。**write-only** —— 這點與 `xqmac8`
  相反，是設計上刻意的區別

| 位元 | 欄位 | 寬度 | 型別 |
|:--|:--|--:|:--|
| 63:32 | `multiplier` | 32 | int32 |
| 31:24 | `shift` | 8 | int8（有號，可為負） |
| 23:16 | `offset` | 8 | int8（輸出 zero-point） |
| 15:8 | `min` | 8 | int8（飽和下界） |
| 7:0 | `max` | 8 | int8（飽和上界） |

運算順序：

```
1. 定點乘：SaturatingRoundingDoublingHighMul(acc, multiplier)
2. 捨入右移（shift 為負）或左移（shift 為正）
3. += offset          ← 在 int32 裡，溢位時環繞
4. clamp 到 [min, max]
5. sign-extend 到 XLEN 寫入 rd
```