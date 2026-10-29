# RV64IMC bare-metal target。
# 放到 tensorflow/lite/micro/tools/make/targets/riscv64_generic_makefile.inc
#
# 以 TFLM 內附的 riscv32_generic_makefile.inc 為基礎，四處不同：
#   1. rv64imc_zicsr / lp64 —— 規格書指定的 ISA
#   2. 不下載 toolchain，用 $RISCV 下自己建好的那套（PATH 上就有）
#   3. 不加 --specs=nano.specs —— 我們自己做 bare-metal 連結，不要 newlib 的 crt
#   4. 不設 TEST_SCRIPT / SIZE_SCRIPT —— 我們不跑 TFLM 自己的 test target
#
# TARGET 取名 riscv64_generic 不是隨意的：Makefile 的
#   PATH_TO_TARGET_SRCS := tensorflow/lite/micro/$(TARGET)
# 會去那個目錄找同名檔案做特化，所以我們的 debug_log.cc 要放
#   tensorflow/lite/micro/riscv64_generic/debug_log.cc
#
# 產物路徑由 GENDIR := gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE) 決定。
#
# 注意：下面的 TARGET_ARCH 與 BUILD_TYPE 賦值對 GENDIR **無效** ——
# TFLM 的 Makefile 在 include 本檔之前就算好了 GENDIR，所以只有命令列
# 傳進去的變數蓋得過。實際的建置指令必須是
#
#   make -f tensorflow/lite/micro/tools/make/Makefile \
#        TARGET=riscv64_generic TARGET_ARCH=riscv64 microlite -j8
#
# 產物在 gen/riscv64_generic_riscv64_default_gcc/lib/libtensorflow-microlite.a
# （不是 _riscv64_micro —— BUILD_TYPE 的賦值同樣太晚）。
# 漏掉 TARGET_ARCH=riscv64 的話會變成 gen/riscv64_generic_x86_64_default_gcc/，
# 庫的內容一樣（旗標來自本檔），但目錄名不同。

TARGET_ARCH := riscv64
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

RISCV_ARCH := rv64imc_zicsr
RISCV_ABI := lp64
RISCV_CODE_MODEL := medany

RISCV_EXTRA_CFLAGS :=

# 留空表示直接吃 PATH 上的 riscv64-unknown-elf-*，不要去 downloads 抓
TARGET_TOOLCHAIN_ROOT :=

PLATFORM_FLAGS = \
  -march=$(RISCV_ARCH) \
  -mabi=$(RISCV_ABI) \
  -mcmodel=$(RISCV_CODE_MODEL) \
  -mexplicit-relocs \
  -fno-builtin-printf \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fpermissive \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX

CCFLAGS += $(PLATFORM_FLAGS)

BUILD_TYPE := micro

CCFLAGS  += $(RISCV_EXTRA_CFLAGS)
CXXFLAGS += $(RISCV_EXTRA_CFLAGS)

# 提供 vsnprintf_()，不依賴 newlib 的 stdio。我們的 debug_log.cc 會用它。
include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc
