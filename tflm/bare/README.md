# bare-metal RV64IMC（B0）

`nn_main.cc` 跟 x86 完全共用，這裡只有平台層。

> **先說清楚：這一包我沒有實機驗過。** 我手上沒有 RV64 toolchain 也沒有
> Spike。能驗的我驗了 —— 型別檢查全過、HTIF 協定是對照 `riscv-isa-sim/fesvr`
> 原始碼寫的而不是憑記憶、libm 的需求是用 `nm -u` 從 TFLM 的 object 實際列出來的。
> 沒驗的是 RV64 專屬的部分：relocation、ISA attribute 相容性、實際執行。
> 所以下面刻意分成四階段，每一階失敗都能指向明確的原因。

---

## 階段 0：單獨驗 HTIF console（先做這個）

```bash
cd ~/ca-project/repo/tflm/bare
riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
  -O2 -nostdlib -nostartfiles -T ../../tests/env/link.ld \
  t_console.c -o /tmp/t_console
timeout 10 spike --isa=rv64imc_zicsr -m64 /tmp/t_console </dev/null
echo "exit=$?"
```

預期：印出 `HTIF console OK`，離開碼 0。

**為什麼先做這個**：console 是唯一一個我只能從原始碼推、沒辦法實測的東西。
協定的關鍵是 `bcd_t::handle_write` **不呼叫 `respond()`**，所以 console 寫入
不會設 `fromhost` —— 照直覺去 `while (fromhost == 0)` 等回應會永遠卡住。
這個測試 30 秒就能確認，不要把它跟 5 MB 的 TFLM 移植綁在一起除錯。

結果對應：

| 結果 | 意思 |
|---|---|
| 印出訊息、離開碼 0 | 協定正確，往下走 |
| 逾時、沒有輸出 | 協定錯或 sp 沒設好 —— 貼給我 |
| 沒輸出但離開碼 0 | console 不通、結束可用 → 先只靠離開碼，不靠 print |

---

## 階段 1：把兩個檔案放進 TFLM 樹

```bash
cd ~/ca-project/work/tflite-micro
cp ~/ca-project/repo/tflm/tflm-patch/riscv64_generic_makefile.inc \
   tensorflow/lite/micro/tools/make/targets/
mkdir -p tensorflow/lite/micro/riscv64_generic
cp ~/ca-project/repo/tflm/tflm-patch/debug_log.cc \
   tensorflow/lite/micro/riscv64_generic/
```

目錄名不能改。TFLM 的 Makefile 有

```make
PATH_TO_TARGET_SRCS := tensorflow/lite/micro/$(TARGET)
```

會拿那個目錄裡同名的檔案去替換 base 版本。所以 `TARGET=riscv64_generic`
就會自動用我們的 `debug_log.cc` 蓋掉原本用 `vfprintf(stderr, ...)` 的那份
（那份需要 newlib 的整套 stdio）。

**記得把這兩個檔案留在 `repo/tflm/tflm-patch/`** —— 它們是對第三方樹的修改，
報告的可重現性那 5% 要靠這個。

---

## 階段 2：建 riscv64 版的 microlite

```bash
cd ~/ca-project/work/tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
     TARGET=riscv64_generic TARGET_ARCH=riscv64 microlite -j8
find gen -path '*riscv64*' -name 'libtensorflow-microlite*.a'
```

產物會在 `gen/riscv64_generic_riscv64_micro/lib/`，跟 x86 那份並存不衝突
（路徑由 `GENDIR := gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)` 決定）。

---

## 階段 3：編我們的程式並執行

```bash
cd ~/ca-project/repo/tflm/bare
make paths          # 每一行都不能有 !!
make                # 會順便印 size
make run            # 可能要跑幾分鐘
```

`make run` 預期印出跟 x86 一樣的 `BITEXACT`，離開碼 0。

`make` 之後先看 `size` 的輸出，應該大致是：

| section | 約略大小 | 內容 |
|---|---|---|
| `.rodata` | 5.3 MB | 模型 |
| `.bss` | 1.7 MB | tensor arena |
| `.text` | 數百 KB | TFLM + 我們的程式 |

---

## 可能會卡的地方，依可能性排序

**1. ISA attribute 不相容**

```
ld: ... : can't link rv64imc_zicsr object with rv64imc object
```

newlib 的 libm/libc 是用 `rv64imc` 建的（你 configure 時 `--with-arch=rv64imc`），
我們的程式是 `rv64imc_zicsr`。連結器通常會取聯集，但不保證。

真的擋住的話：把 `bare/Makefile` 的 `ARCH` 改成 `rv64imc`，CSR 讀取改用
`.insn` 繞過組譯器的擴充檢查 —— 跟你 `xqtest` 已經在用的手法一樣。

`rdinstret` 就是 `csrrs rd, 0xc02, x0`：

```c
#define RDINSTRET(x) __asm__ volatile(".insn i 0x73, 2, %0, x0, -1022" : "=r"(x))
```

`-1022` 不是打錯：`.insn i` 的立即值欄位是**有號** 12 bit，而 CSR 編號
`0xc02` = 3074 超出 +2047 的上限，要寫成它的二補數 `3074 - 4096 = -1022`。
寫 `0xc02` 組譯器會報 value out of range。

驗證方式：編出來反組譯應該看到 `c0202573`（rd=a0）這類字樣，
而且 `objdump` 會直接標成 `rdinstret`。你之前的 `t_g` 反組譯裡
出現過 `c0202673  rdinstret a2`，就是同一個編碼（rd=a2）。

Phase 3 要量指令數才會用到，階段 3 本身不需要。

**2. 還缺符號**

```bash
make undef     # 列出所有還沒定義的外部符號
```

我是用 x86 的 object 去抓需求清單的，glibc 跟 newlib 的內部符號不同
（例如 `__errno_location` vs `__errno`），所以可能會有幾個漏的。
把 `make undef` 的輸出貼給我。

**3. `relocation truncated to fit: R_RISCV_HI20`**

某處漏了 `-mcmodel=medany`。TFLM 那邊和我們這邊都要有 —— 兩個 Makefile 都設了，
除非你改過。

**4. 逾時（離開碼 124）**

- 完全沒輸出 → 卡在啟動。用 `make dis` 看 `_start` 有沒有在 `0x80000000`、
  `la sp, _stack_top` 有沒有被編出來。
- 有輸出才卡住 → 卡在推論中。可能只是慢，先把 `TIMEOUT` 加大再試
  （`make run TIMEOUT=3600`）。

**5. 跑完但不是 BITEXACT**

這個最有意思，而且不是「哪裡寫錯」那種問題。可能的原因是
**libm 的 `pow`/`exp` 在 newlib 與 glibc 之間最後一個 bit 不同**，
而那會讓 requantize 的 multiplier 差 1，進而改變輸出。

`QuantizeMultiplier` 用的是 `std::frexp`，而 `frexp`/`floor`/`TfLiteRound`
都是精確運算、不產生捨入誤差，軟浮點的 IEEE-754 四則運算也是確定性的 ——
所以理論上應該要 BITEXACT。真的不是的話把差異分布貼給我，
那一行能看出是系統性偏移還是零星的平手差。

---

## 之後 B1 可以動的地方

這些都是量得出 code size / 指令數差異的具體槓桿，不是口頭上的「精簡」：

- `-DTF_LITE_STRIP_ERROR_STRINGS` —— 砍掉所有錯誤訊息字串
- `-Wl,--gc-sections` 配 `-ffunction-sections -fdata-sections` —— 砍掉用不到的程式碼
  （B0 刻意不開，少一個變數）
- op resolver 從 TFLM 完整的 118 個砍到我們實際用的 8 個
- arena 從 8 MB 收到實測的 1649 KiB（已經做了）
