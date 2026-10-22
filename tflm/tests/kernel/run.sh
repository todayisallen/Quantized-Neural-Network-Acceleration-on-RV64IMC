#!/usr/bin/env bash
# run.sh — kernel 層的等價性測試（x86，不需要 Spike）。
#
#   bash tests/kernel/run.sh           # 預設案例數
#   bash tests/kernel/run.sh 20000     # 跑久一點
#
# 兩個 kernel 各自跟 TFLM 的參考實作對打，要求輸出**位元完全相同**：
#   conv_equiv.cc  1x1 conv      （占全模型 88.18% 的 MAC）
#   dw_equiv.cc    depthwise conv
#
# conv 的部分會跑兩次：B1 的純量路徑與 X1 的 xqmac8 路徑。
# x86 上沒有 xqmac8，所以用 XQ_SIM_INSN 以純量模擬那條指令 ——
# 語意與 bare/t_xqmac8.c 在 Spike 上驗過的實作一致。這樣
# 「對齊判斷、K 的尾巴、權重複製」這些純軟體邏輯可以在秒級驗證，
# 不必每次上 Spike。
#
# 只連 TFLM 的 kernels/internal/common.cc（MultiplyByQuantizedMultiplier），
# 不連 libtflm.a —— 測的是純數值，不該依賴 interpreter。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_DIR="$(cd "$HERE/../.." && pwd)"
TFLM_ROOT="${TFLM_ROOT:-$HOME/ca-project/work/tflite-micro}"
CXX="${CXX:-/usr/local/bin/g++}"
N="${1:-}"

DL="$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads"
COMMON="$TFLM_ROOT/tensorflow/lite/micro/kernels/internal/common.cc"

for f in "$TFLM_ROOT" "$DL/flatbuffers/include" "$COMMON"; do
  [ -e "$f" ] || { echo "找不到 $f"; echo "設 TFLM_ROOT 指到 tflite-micro 的根目錄"; exit 1; }
done

OUT="$HERE/build"; mkdir -p "$OUT"
rc=0

build_and_run() {   # $1=名稱 $2=原始碼 $3=額外的 -D $4=案例數
  local name="$1" src="$2" defs="$3" n="$4"
  printf "\n\033[1m=== %s ===\033[0m\n" "$name"
  # shellcheck disable=SC2086
  "$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
    -DTF_LITE_STATIC_MEMORY $defs \
    -fno-exceptions -fno-rtti -fno-threadsafe-statics \
    -I"$TFLM_ROOT" -I"$DL/flatbuffers/include" -I"$TFLM_DIR/app" \
    "$src" "$COMMON" -o "$OUT/$name" || { echo "編譯失敗"; rc=1; return; }
  "$OUT/$name" $n || rc=1
}

build_and_run conv_b1 "$HERE/conv_equiv.cc" ""                            "${N:-600}"
build_and_run conv_x1 "$HERE/conv_equiv.cc" "-DUSE_XQMAC8 -DXQ_SIM_INSN"  "${N:-600}"
build_and_run dw      "$HERE/dw_equiv.cc"   ""                            "${N:-2000}"

echo
if [ "$rc" -eq 0 ]; then
  printf "\033[32mkernel 等價性測試全部通過\033[0m\n"
else
  printf "\033[31m有測試失敗\033[0m\n"
fi
exit $rc