# bare-metal RV64IMC build。
#
# 跟 host/Makefile 的差別只有工具鏈與平台層，nn_main.cc / gen/ 完全共用。
#
#   make paths        先跑這個，確認路徑與工具都在
#   make              編譯
#   make run          在 Spike 上跑（會花幾分鐘，見下面說明）
#   make size dis     看 section 大小 / 反組譯
#   make ARENA_KB=... 改 arena 大小

TFLM_ROOT ?= $(HOME)/ca-project/work/tflite-micro
GEN       ?= ../gen
APP       ?= ../app
ARENA_KB  ?= 1713

# 必須用 medany 的那套 toolchain。
#
# 原本 $RISCV 下那套是預設的 medlow，它附的 libgcc 用 lui+addi 絕對定址
# （__clz_tab 的查表、__divdf3 的 jump table），連在 0x80000000 會
#     relocation truncated to fit: R_RISCV_HI20
# 而 __divdf3 繞不開 —— rv64imc 沒有 FPU，QuantizeMultiplier 的 double
# 除法一定要它。所以另外建了一套 --with-cmodel=medany 的。
#
# 寫死路徑而不是靠 PATH，是因為 PATH 上還有舊的那套，
# 用錯了只會在連結時才爆，而且錯誤訊息看起來跟 code model 無關。
RISCV_MEDANY ?= $(HOME)/ca-project/riscv-medany
CROSS   := $(RISCV_MEDANY)/bin/riscv64-unknown-elf-
CXX     := $(CROSS)g++
OBJDUMP := $(CROSS)objdump
NM      := $(CROSS)nm
SIZE    := $(CROSS)size

ARCH := rv64imc_zicsr
ABI  := lp64

# 載入位址。跟 tests/env 一致，也是 Spike 的預設 DRAM_BASE。
#
# 之所以能留在 0x80000000，是因為我們**完全不連 newlib**（見 libc_min.cc）。
# newlib 的 libm 是 medlow 建的，絕對定址只到 0x7fffffff，連在這個位址
# 會得到一堆 "relocation truncated to fit: R_RISCV_HI20"。
# 自己提供那幾個函式之後就沒有這個限制了。
# 注意當時的連結錯誤裡只有 libm、沒有 libgcc，所以軟浮點那組是位置無關的。
LOAD_BASE ?= 0x80000000

DL  := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads
# 只找 riscv64 那一份，不要抓到 x86 的
LIB := $(shell find $(TFLM_ROOT)/gen -path '*riscv64*' \
         -name 'libtensorflow-microlite*.a' 2>/dev/null | head -1)

INC := -I$(TFLM_ROOT) \
       -I$(DL)/flatbuffers/include \
       -I$(DL) \
       -I$(DL)/eyalroz_printf/src \
       -I$(GEN) -I$(APP)

# 這些旗標必須跟 TFLM 的 riscv64_generic target 一致。
# TF_LITE_STATIC_MEMORY 尤其不能漏 —— 它會改變 TfLiteTensor 的結構佈局，
# 兩邊不一致的話張量指標會指到奇怪的地方，而且編譯完全沒有警告。
CXXFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany \
            -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
            -fno-exceptions -fno-rtti -fno-threadsafe-statics \
            -fno-use-cxa-atexit -fomit-frame-pointer \
            -funsigned-char -fno-delete-null-pointer-checks \
            -DTF_LITE_STATIC_MEMORY \
            -DTFLM_REMOVE_VIRTUAL_DELETE \
            -DTF_LITE_MCU_DEBUG_LOG \
            -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
            -DTF_LITE_USE_GLOBAL_MIN -DTF_LITE_USE_GLOBAL_MAX \
            -DARENA_KB=$(ARENA_KB) \
            $(INC)

# -nostartfiles  我們自己有 _start
# -nostdlib      不要預設的連結群組，下面手動把需要的庫加回來
# -mno-relax     TFLM 自己的 riscv target 註明連結器 relaxation 會產生錯誤的碼
#                （tools/make/targets/riscv32_generic_makefile.inc 的註解）
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany \
           -nostdlib -nostartfiles -mno-relax -T link.ld \
           -Wl,--defsym=LOAD_BASE=$(LOAD_BASE) \
           -Wl,--no-warn-rwx-segments

# 只連 libgcc：rv64imc 沒有 FPU，double 的四則運算要靠它的軟浮點
# （__adddf3 / __muldf3 / __divdf3 ...）。
#
# **不連 -lm -lc** —— string 與 math 由 libc_min.cc 自己提供。
# 原因見 libc_min.cc 開頭：newlib 的 libm 是 medlow 建的，
# 連在 0x80000000 會 relocation truncated to fit。
LDLIBS := -lgcc

# ======================================================================
# 配置：四個 configuration 共用同一份原始碼，差別只在編譯期開關。
# 這樣「唯一的變數」這件事是由 build 系統保證的，不是靠紀律。
#
#   B0   TFLM 原樣（移植好的 runtime）
#   B1   1x1 conv 換成重寫過的 kernel，內積用純 RV64IMC scalar
#   X1   同一份 kernel，內積換成 xqmac8
#
# 產物分開放在 build/<CONFIG>/，可以並存，不用每次 clean。
#
#   make CONFIG=B1 run
# ======================================================================
CONFIG ?= B0
CONFIG_CXXFLAGS :=
CONFIG_SRC :=
ifeq ($(CONFIG),B0)
else ifeq ($(CONFIG),B1)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()' \
                     -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/conv_xq.cc $(APP)/dwconv_xq.cc
# B1C / B1D：把 B1 的兩項改動拆開，用來分別歸因加速來源。
# 報告的 §4 要講「哪個改動貢獻多少」，不能只給一個合併數字。
else ifeq ($(CONFIG),B1C)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/conv_xq.cc
else ifeq ($(CONFIG),B1D)
  CONFIG_CXXFLAGS := -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/dwconv_xq.cc
else ifeq ($(CONFIG),X1)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()' \
                     -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()' \
                     -DUSE_XQMAC8
  CONFIG_SRC := $(APP)/conv_xq.cc $(APP)/dwconv_xq.cc
# X2：在 X1 之上再加 xqrq（再量化指令）。
# 規格書 §3 要求「Evaluation of instruction effectivness should be based
# incremental implementation of one instruction on the other」——
# 所以 X2 是 X1 的超集，不是另一個分支。
else ifeq ($(CONFIG),X2)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()' \
                     -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()' \
                     -DUSE_XQMAC8 -DUSE_XQRQ
  CONFIG_SRC := $(APP)/conv_xq.cc $(APP)/dwconv_xq.cc
else
  $(error CONFIG 必須是 B0 / B1 / B1C / B1D / X1 / X2，收到 '$(CONFIG)')
endif

# depthwise 的 channel 分塊寬度。空的話用 dwconv_xq.cc 裡的預設（8）。
#
#   make CONFIG=B1 BLK=4 run
#
# 為什麼要能從命令列掃：8 個累加器放不放得進暫存器，取決於整個函式的
# 暫存器壓力，看原始碼猜不出來。BLK 小一點累加器少，但每個 block
# 攤提到的位址算術變多；最佳值只能量。
# 不同 BLK 進不同的 build 目錄，掃的時候不用 clean。
BLK ?=
ifneq ($(BLK),)
  CONFIG_CXXFLAGS += -DDW_XQ_BLK=$(BLK)
  BLKSUF := -blk$(BLK)
else
  BLKSUF :=
endif

# REPORT=1 開啟 conv_xq.cc 的 XQ_REPORT 診斷：每個走特化路徑的 1x1 層
# 印一行，並累計真正走 packed 路徑的 MAC 數。
#
#   make CONFIG=X1 REPORT=1 run
#
# 用途是在相信任何 X1 數字之前先釘住因果 —— 「數字變好了」不能證明
# 「指令生效了」。之前在 x86 上踩過一次：-Igen 的順序讓自訂 kernel
# 根本沒被註冊，程式跑得完全正常，量到一個 1.13x 的假數字。
#
# 這個診斷會印 33 行、改變指令數，所以**不能**用來量效能。
# 進獨立的 build 目錄，不會污染 results/ 裡的量測。
REPORT ?=
ifneq ($(REPORT),)
  CONFIG_CXXFLAGS += -DXQ_REPORT
  RPTSUF := -rpt
else
  RPTSUF :=
endif

SRC := $(APP)/nn_main.cc $(APP)/prof.cc $(CONFIG_SRC) \
       main_bare.cc syscalls.cc libc_min.cc \
       $(GEN)/model_data.cc $(GEN)/input_data.cc $(GEN)/golden_data.cc
BUILD := build/$(CONFIG)$(BLKSUF)$(RPTSUF)
OBJ := $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(SRC)))
BIN := $(BUILD)/nn_bare

# ======================================================================
# 標頭檔相依性追蹤
# ======================================================================
# 沒有這個的話，改了 .h 不會觸發重編 —— 增量建置的執行檔會跟乾淨建置的
# 不同。這不是理論風險：某次量測中 B0 與 B1 各比乾淨建置少 94,157 條
# 指令，而 B1C / B1D 不變，使得「B1C 與 B1D 的節省量相加應等於 B1 的
# 節省量」這項檢查從 0 變成 2x94,157。清空 build 重跑後才回到一致。
#
# -MMD 產生 .d（只含使用者標頭，不含系統標頭），-MP 替每個標頭產生一個
# 空目標，這樣刪掉標頭時不會出現 "No rule to make target"。
DEPFLAGS := -MMD -MP
DEPS := $(OBJ:.o=.d)
-include $(DEPS)

# 用哪一個 spike。預設走 PATH，但可以明確指定 ——
#
#   make -C bare CONFIG=X2 run SPIKE=~/ca-project/work/riscv-isa-sim/build/spike
#
# 為什麼需要這個：當初刻意不覆蓋原有的 spike（好讓舊量測能重現），
# 所以樹裡會有多個版本。改了 A 樹、跑的是 B 樹編出來的執行檔，
# 症狀是「自訂指令永遠不被認得」，而且看起來跟改動完全無關。
SPIKE ?= spike
SPIKE_FLAGS := --isa=$(ARCH) -m64
# 0.385 GMAC 在 Spike 上大概是數十億條指令，第一次跑預留久一點
TIMEOUT ?= 1800

.PHONY: all run paths size dis hotloop undef clean
all: $(BIN)

paths: check-cmodel
	@echo "CXX        = $(CXX) ($(shell $(CXX) --version 2>/dev/null | head -1))"
	@command -v $(CXX) >/dev/null || echo "  !! 找不到 $(CXX)"
	@command -v $(SPIKE) >/dev/null || echo "  !! 找不到 $(SPIKE)"
	@echo "SPIKE      = $(SPIKE) -> $(shell command -v $(SPIKE) 2>/dev/null)"
	@echo "TFLM_ROOT  = $(TFLM_ROOT)"
	@echo "libmicrolite(riscv64) = $(if $(LIB),$(LIB),<找不到>)"
	@test -n "$(LIB)" || echo "  !! 請先建 riscv64 版的 microlite，見 README"
	@echo "CONFIG     = $(CONFIG)"
	@echo "  旗標     = $(if $(CONFIG_CXXFLAGS),$(CONFIG_CXXFLAGS),<無>)"
	@echo "ARENA_KB   = $(ARENA_KB)"
	@echo "ARCH/ABI   = $(ARCH) / $(ABI)"
	@for f in $(SRC); do test -f $$f && echo "  ok   $$f" || echo "  MISS $$f"; done

# 連結前就確認用的是 medany 的 toolchain。
# 不檢查的話，用錯 compiler 會一路編譯成功，到最後連結才爆出一堆
# relocation truncated to fit —— 而那個訊息完全看不出是 code model 的問題。
.PHONY: check-cmodel
check-cmodel:
	@m=$$($(CXX) -Q --help=target 2>/dev/null | \
	      sed -n 's/^[[:space:]]*-mcmodel=[[:space:]]*\(.*\)$$/\1/p' | head -1); \
	 if [ "$$m" != "medany" ]; then \
	   echo "!! $(CXX)"; \
	   echo "!! 的預設 code model 是 '$$m'，不是 medany。"; \
	   echo "!! 它附的 libgcc 無法連在 0x80000000（__divdf3 / __clz_tab）。"; \
	   echo "!! 請確認 RISCV_MEDANY 指向 --with-cmodel=medany 建的那套："; \
	   echo "!!   目前 RISCV_MEDANY = $(RISCV_MEDANY)"; \
	   exit 1; \
	 fi

$(OBJ): | check-cmodel

$(BUILD)/%.o: $(APP)/%.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: $(GEN)/%.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# libc_min.cc 自己實作 memcpy / memset，所以這個 TU 必須關掉 builtin。
#
# 不關的話 GCC 會把 memset 內部那個 byte 填值迴圈認成 memset 的 pattern，
# 直接編成「呼叫 memset」—— 也就是自己呼叫自己，無窮遞迴。
# 這不是理論風險：在 x86 上拿 glibc 當基準測試時實際撞到了，
# 堆疊 522,718 層之後 segfault。-fno-tree-loop-distribute-patterns
# 就是專門關掉那個轉換的旗標。
# 這條是顯式規則，會蓋掉上面的 pattern rule。
$(BUILD)/libc_min.o: libc_min.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -ffreestanding -fno-builtin \
	  -fno-tree-loop-distribute-patterns -c $< -o $@

$(BIN): $(OBJ) link.ld
	@test -n "$(LIB)" || { echo "找不到 riscv64 的 libtensorflow-microlite.a，先跑 make paths"; exit 1; }
	$(CXX) $(LDFLAGS) $(OBJ) $(LIB) $(LDLIBS) -o $@
	@echo "-> $@"
	@$(SIZE) $@

# </dev/null 不能省：stdout 是 pipe 時 Spike 的 HTIF 會輪詢 stdin，
# 不關掉就會永遠等下去（ENV_NOTES §4.1）。
run: $(BIN)
	@echo "$(SPIKE) $(SPIKE_FLAGS) $(BIN)   （可能要跑幾分鐘）"
	@timeout $(TIMEOUT) $(SPIKE) $(SPIKE_FLAGS) $(BIN) </dev/null; rc=$$?; \
	 echo "--- spike 離開碼 $$rc ---"; \
	 case $$rc in \
	   0)   echo "BITEXACT：bare-metal 與 x86 位元完全相同" ;; \
	   2)   echo "CLOSE：top-1 相同但有 LSB 差異 —— 對 x86 比對來說這是失敗，要查" ;; \
	   124) echo "逾時。先確認 console 有沒有輸出；完全沒有就是卡在啟動" ;; \
	   *)   echo "失敗" ;; \
	 esac; \
	 [ $$rc -eq 0 ]

size: $(BIN)
	$(SIZE) -A $(BIN)

dis: $(BIN)
	$(OBJDUMP) -d $(BIN) > $(BUILD)/nn_bare.dis
	@echo "-> $(BUILD)/nn_bare.dis"
	@$(OBJDUMP) -d $(BIN) | sed -n '/<_start>:/,/^$$/p'

# 把某個 kernel 的熱點迴圈挖出來看。
#
# 為什麼需要這個：指令組成分析告訴你「哪個函式花了多少」，
# 但要知道「為什麼是這個數字」就得看生成的碼。
# 第一版 depthwise 實測 17.9 條指令/MAC，看了碼才發現累加器每個 tap
# 都往返記憶體、而且全域陣列位址在 medany 下每輪重算（auipc+addi）。
# 在 x86 上看不準 —— x86-64 只有 16 個通用暫存器，RV64 有 32，
# 暫存器配置的結果完全不同。
#
# 不從函式開頭印：Eval 函式前面幾百行全是取張量的 boilerplate。
# tools/hotloop.py 用滑動視窗找「關鍵指令最密的那一段」。
#
#   make CONFIG=B1 hotloop                      # 預設看 depthwise 的 mul 熱點
#   make CONFIG=B1 hotloop FN=XqConvEval        # 看 1x1
#   make CONFIG=B1 hotloop KEY=custom           # X1：確認 xqmac8 真的編出來了
#   make CONFIG=B1 hotloop W=96                 # 視窗開大一點
# 有 results/<CONFIG>.hist（instr_mix.sh 的產物）就用執行次數找熱點，
# 那是量到的；沒有就退回靜態猜測，工具會自己大聲警告。
#
# .dis 一律現場重新產生 —— 不能用 results/ 裡那份舊的，
# 位址必須跟 .hist 來自同一個 binary，否則執行次數全對不上。
FN  ?= XqDwConvEval
KEY ?= mul
W   ?= 56
HIST = ../results/$(CONFIG)$(BLKSUF)$(RPTSUF).hist
hotloop: $(BIN)
	@$(OBJDUMP) -d --no-show-raw-insn $(BIN) > $(BUILD)/hotloop.dis
	@if [ -f "$(HIST)" ]; then \
	   echo "用 $(HIST) 的執行次數找熱點"; \
	   python3 ../tools/hotloop.py --func '$(FN)' --window $(W) \
	     --objdump $(BUILD)/hotloop.dis --hist "$(HIST)"; \
	 else \
	   echo "找不到 $(HIST)，退回靜態猜測（先跑 scripts/instr_mix.sh $(CONFIG) 會準得多）"; \
	   python3 ../tools/hotloop.py --func '$(FN)' --key '$(KEY)' --window $(W) \
	     --objdump $(BUILD)/hotloop.dis; \
	 fi

# ======================================================================
# t_xqrq：xqrq 指令本身的驗證（規格書 §5 的 directed + randomized）
# ======================================================================
# 跟 nn_bare 分開建：它不跑模型，只測一條指令。
#
# golden 用的是 TFLM 本體的 MultiplyByQuantizedMultiplier，所以要連
# libtensorflow-microlite.a（也順便拿到 MicroPrintf / DebugLog）。
# 刻意**不**定義 XQ_SIM_INSN —— 要測的是真的指令，不是模擬。
#
#   make -C bare t_xqrq        編
#   make -C bare t_xqrq_run    編 + 在 Spike 上跑（期望最後一行 pass=N fail=0）
TXQ_BUILD := build/t_xqrq
TXQ_SRC   := t_xqrq.cc main_bare.cc syscalls.cc libc_min.cc
TXQ_OBJ   := $(patsubst %.cc,$(TXQ_BUILD)/%.o,$(TXQ_SRC))
TXQ_BIN   := $(TXQ_BUILD)/t_xqrq
-include $(TXQ_OBJ:.o=.d)

.PHONY: t_xqrq t_xqrq_run
t_xqrq: $(TXQ_BIN)

$(TXQ_OBJ): | check-cmodel

$(TXQ_BUILD)/%.o: %.cc
	@mkdir -p $(TXQ_BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# 同 build/<CFG>/libc_min.o 的理由：自己實作 memset 的 TU 必須關掉
# builtin，否則 GCC 會把 memset 內部的填值迴圈認成 memset 而自我遞迴。
$(TXQ_BUILD)/libc_min.o: libc_min.cc
	@mkdir -p $(TXQ_BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -ffreestanding -fno-builtin \
	  -fno-tree-loop-distribute-patterns -c $< -o $@

$(TXQ_BIN): $(TXQ_OBJ) link.ld
	@test -n "$(LIB)" || { echo "找不到 riscv64 的 libtensorflow-microlite.a，先跑 make paths"; exit 1; }
	$(CXX) $(LDFLAGS) $(TXQ_OBJ) $(LIB) $(LDLIBS) -o $@
	@echo "-> $@"

t_xqrq_run: $(TXQ_BIN)
	@timeout 300 $(SPIKE) $(SPIKE_FLAGS) $(TXQ_BIN) </dev/null; rc=$$?; \
	 echo "--- spike 離開碼 $$rc ---"; \
	 case $$rc in \
	   0) echo "xqrq 與 TFLM 的 MultiplyByQuantizedMultiplier 位元一致" ;; \
	   1) echo "有案例不符 —— 看上面印出的案例名" ;; \
	   *) echo "指令沒被認得或程式沒跑起來（檢查 Spike 的四處改動）" ;; \
	 esac; \
	 [ $$rc -eq 0 ]

# ======================================================================
# bench：kernel 層的 microbenchmark（規格書 §5 Evaluation Coverage）
# ======================================================================
# 跟 nn_bare 分開建，但**用同一組 CONFIG 旗標** —— 這樣 B1/X1/X2 的
# kernel 跟完整模型跑的是同一份碼，數字可以互相對照。
#
# bench_main.cc 把 conv_xq.cc 與 dwconv_xq.cc #include 進自己的 TU
# （KERNEL_ONLY 模式），所以**不能**把那兩個檔也列進 SRC ——
# conv_xq.cc 的 g_wbuf 在全域範圍，重複編譯會是重複定義。
#
#   make -C bare CONFIG=X2 bench_run
#   make -C bare CONFIG=B1 bench_run    # 對照：沒有自訂指令的版本
BN_BUILD := build/bench-$(CONFIG)
BN_SRC   := bench_main.cc main_bare.cc syscalls.cc libc_min.cc
BN_OBJ   := $(patsubst %.cc,$(BN_BUILD)/%.o,$(BN_SRC))
BN_BIN   := $(BN_BUILD)/bench
-include $(BN_OBJ:.o=.d)

.PHONY: bench bench_run
bench: $(BN_BIN)

$(BN_OBJ): | check-cmodel

$(BN_BUILD)/%.o: %.cc
	@mkdir -p $(BN_BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# 同 build/<CFG>/libc_min.o：自己實作 memset 的 TU 要關掉 builtin
$(BN_BUILD)/libc_min.o: libc_min.cc
	@mkdir -p $(BN_BUILD)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -ffreestanding -fno-builtin \
	  -fno-tree-loop-distribute-patterns -c $< -o $@

$(BN_BIN): $(BN_OBJ) link.ld
	@test -n "$(LIB)" || { echo "找不到 riscv64 的 libtensorflow-microlite.a，先跑 make paths"; exit 1; }
	$(CXX) $(LDFLAGS) $(BN_OBJ) $(LIB) $(LDLIBS) -o $@
	@echo "-> $@"

bench_run: $(BN_BIN)
	@timeout 900 $(SPIKE) $(SPIKE_FLAGS) $(BN_BIN) </dev/null; rc=$$?; \
	 echo "--- spike 離開碼 $$rc ---"; \
	 case $$rc in \
	   0) echo "所有 case 與參考實作位元一致" ;; \
	   1) echo "有 case 算錯 —— 效能數字在修好之前沒有意義" ;; \
	   *) echo "沒跑起來（自訂指令沒被認得？確認 $(SPIKE)）" ;; \
	 esac; \
	 [ $$rc -eq 0 ]

# 連結失敗時先看這個：還缺哪些符號
undef: $(OBJ)
	@$(NM) -u $(OBJ) $(LIB) 2>/dev/null | grep -E '^ +U ' | \
	  sed 's/^ *U //' | sort -u

clean:
	rm -rf build