# x86 host build：把 TFLM 當靜態庫連，自己的程式單獨編。
#
# 刻意不用 TFLM 的 Makefile 來編我們的程式：
# 那會把專案塞進 TFLM 的目錄結構裡，之後換 TFLM 版本、
# 或是要跟 bare-metal 共用同一份 nn_main.cc 都會很痛。
#
#   make paths     印出找到的路徑（先跑這個確認）
#   make           編譯
#   make run       編譯並執行
#   make ARENA_KB=2048   改 arena 大小重編

TFLM_ROOT ?= $(HOME)/ca-project/work/tflite-micro
GEN       ?= ../gen
APP       ?= ../app
ARENA_KB  ?= 8192

# 編譯器。這裡**刻意不設預設值**：
#
# 原本寫的是 `CXX ?= /usr/local/bin/g++`（為了避開 CentOS 7 內建的 4.8.5），
# 但那行是**無效的** —— make 有內建的 CXX = g++（origin 'default'），而 ?=
# 把「有 default 值」當成「已設定」，所以根本不會賦值。實測：
#
#   CXX ?= /usr/local/bin/g++   ->   CXX = g++  (origin: default)
#
# 在開發機上建置成功是因為 PATH 裡 /usr/local/bin 排在 /usr/bin 前面，
# `g++` 剛好解析到新版 —— 與那行 Makefile 無關。
#
# 寫死路徑也不對（不可攜）。所以改成：用 PATH 上的 g++，並在 `make paths`
# 實際編一個 C++17 的小程式來檢查。要指定別的編譯器就從命令列傳：
#
#   make CXX=/usr/local/bin/g++ ...
#   make CXX=clang++ ...

DL   := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads
# 庫名在 TFLM 版本之間改過：舊的叫 libtensorflow-microlite.a，
# 新的叫 libtflm.a。兩種都找，不要寫死。
LIB  := $(shell find $(TFLM_ROOT)/gen \
          \( -name 'libtflm*.a' -o -name 'libtensorflow-microlite*.a' \) \
          2>/dev/null | head -1)

INC := -I$(TFLM_ROOT) \
       -I$(DL)/flatbuffers/include \
       -I$(DL)/gemmlowp \
       -I$(DL)/ruy \
       -I$(DL)/kissfft \
       -I$(GEN) -I$(APP)

# TF_LITE_STATIC_MEMORY 不能漏：它會改變 TfLiteTensor 的結構佈局。
# 我們這邊沒定義、libtflm.a 有定義的話，兩邊對同一個結構的認知不同，
# 症狀是張量指標莫名其妙指到奇怪的地方，而且編譯完全沒有警告。
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
            -DTF_LITE_STATIC_MEMORY \
            -DARENA_KB=$(ARENA_KB) \
            -fno-exceptions -fno-rtti -fno-threadsafe-statics \
            $(INC)

# 配置切換，跟 bare/Makefile 一致。
#   B0  TFLM 原樣
#   B1  1x1 conv 換成重寫過的 kernel
# X1 在 host 上編不了（xqmac8 是 RISC-V 的 .insn），要驗 X1 只能用 Spike。
#
# host 的用途是**快速驗正確性**：B1 的 kernel 改完先在這裡確認 BITEXACT，
# 再去 Spike 上花 90 秒量指令數。
CONFIG ?= B0
CONFIG_CXXFLAGS :=
CONFIG_SRC :=
ifeq ($(CONFIG),B0)
else ifeq ($(CONFIG),B1)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()' \
                     -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/conv_xq.cc $(APP)/dwconv_xq.cc
# B1C：只開 1x1，不開 depthwise。用來把 B1 的兩項改動拆開歸因。
else ifeq ($(CONFIG),B1C)
  CONFIG_CXXFLAGS := -DUSE_CONV_XQ -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/conv_xq.cc
# B1D：只開 depthwise，不開 1x1。同上。
else ifeq ($(CONFIG),B1D)
  CONFIG_CXXFLAGS := -DUSE_DWCONV_XQ \
                     -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()'
  CONFIG_SRC := $(APP)/dwconv_xq.cc
else ifeq ($(CONFIG),X1)
  $(error X1 用到 RISC-V 的 .insn，host 編不了。請用 bare/ 加 Spike)
else ifeq ($(CONFIG),X2)
  $(error X2 用到 RISC-V 的 .insn，host 編不了。kernel 層的等價性用 \
          tests/kernel/run.sh 的 XQ_SIM_INSN 純量模擬驗，見 docs/X2_xqrq_design.md)
else
  $(error CONFIG 必須是 B0 / B1 / B1C / B1D，收到 '$(CONFIG)')
endif

SRC := $(APP)/nn_main.cc $(APP)/prof.cc $(CONFIG_SRC) main_host.cc \
       $(GEN)/model_data.cc $(GEN)/input_data.cc $(GEN)/golden_data.cc
BUILD := build/$(CONFIG)
OBJ := $(patsubst %.cc,$(BUILD)/%.o,$(notdir $(SRC)))
BIN := $(BUILD)/nn_host

# 標頭檔相依性追蹤。沒有這個的話改了 .h 不會觸發重編，
# 增量建置與乾淨建置會產生不同的執行檔（bare/Makefile 有詳細說明）。
DEPFLAGS := -MMD -MP
DEPS := $(OBJ:.o=.d)
-include $(DEPS)

.PHONY: all run paths clean
all: $(BIN)

paths:
	@echo "TFLM_ROOT = $(TFLM_ROOT)"
	@test -d $(TFLM_ROOT) || echo "  !! 目錄不存在"
	@echo "downloads = $(DL)"
	@test -d $(DL)/flatbuffers/include || \
	  echo "  !! 沒有 flatbuffers，請先跑 third_party_downloads"
	@echo "libtflm   = $(if $(LIB),$(LIB),<找不到>)"
	@test -n "$(LIB)" || echo "  !! 請先在 TFLM_ROOT 下建 microlite（見 RUNBOOK 步驟 1）"
	@echo "CXX       = $(CXX) ($(shell $(CXX) --version 2>/dev/null | head -1))"
	@command -v $(CXX) >/dev/null 2>&1 || echo "  !! 找不到 $(CXX)"
	@t=$$(mktemp /tmp/cxx17probe.XXXXXX.cc); \
	 printf '#include <cstdint>\n#include <type_traits>\nint main(){ if constexpr(true){} return 0; }\n' > $$t; \
	 $(CXX) -std=c++17 -fsyntax-only $$t 2>/dev/null \
	   || { echo "  !! $(CXX) 不支援 -std=c++17"; \
	        echo "  !!   TFLM 需要 C++17。CentOS 7 內建的 g++ 4.8.5 不夠。"; \
	        echo "  !!   用 make CXX=/path/to/newer/g++ 指定"; }; \
	 rm -f $$t
	@echo "CONFIG    = $(CONFIG)"
	@echo "ARENA_KB  = $(ARENA_KB)"
	@for f in $(SRC); do test -f $$f && echo "  ok   $$f" || echo "  MISS $$f"; done

$(BUILD)/%.o: $(APP)/%.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: $(GEN)/%.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(CONFIG_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	@test -n "$(LIB)" || { echo "找不到 libtflm.a，先跑 make paths"; exit 1; }
	$(CXX) $(OBJ) $(LIB) -lm -o $@
	@echo "-> $@"

# 離開碼 0 = BITEXACT、2 = CLOSE，兩者都算通過；只有其他值才是真的失敗。
# 不做這個轉換的話 make 會把 CLOSE 當成 Error 2。
run: $(BIN)
	@./$(BIN); rc=$$?; \
	 if [ $$rc -eq 0 ] || [ $$rc -eq 2 ]; then exit 0; else exit $$rc; fi

clean:
	rm -rf build