#!/usr/bin/env bash
# measure.sh — 建置並量測一個或多個 configuration，結果一律落地。
#
# 用法:
#   bash scripts/measure.sh              # B0 B1（預設）
#   bash scripts/measure.sh B0 B1 X1
#   bash scripts/measure.sh --summary-only   # 只重產對照表，不重跑
#   ARENA_KB=1713 bash scripts/measure.sh B0
#
# 產出:
#   results/<CONFIG>.txt        完整執行輸出（分段、每 op、每層）
#   results/<CONFIG>.size.txt   section 大小
#   results/summary.md          所有已量測配置的對照表
#
# 為什麼要有這個腳本：量測結果留在終端機上，關掉視窗就沒了。
# 四個配置、每個 90 秒，重跑一次是幾分鐘；但如果是報告寫到一半
# 才發現數字不見，那個代價不是幾分鐘。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM="$(cd "$HERE/.." && pwd)"
ARENA_KB="${ARENA_KB:-1713}"
SPIKE_ISA="${SPIKE_ISA:-rv64imc_zicsr}"
# 用哪一個 spike。樹裡可能有多個版本（刻意不覆蓋舊的，好讓舊量測能重現），
# 所以讓它可覆寫：SPIKE=/path/to/spike bash scripts/...
SPIKE="${SPIKE:-spike}"
TIMEOUT="${TIMEOUT:-1800}"
RES="$TFLM/results"
# 用 medany 那套 toolchain 的 size，跟 bare/Makefile 的 CROSS 一致
RISCV_MEDANY="${RISCV_MEDANY:-$HOME/ca-project/riscv-medany}"
SIZE_BIN="$RISCV_MEDANY/bin/riscv64-unknown-elf-size"
command -v "$SIZE_BIN" >/dev/null 2>&1 || SIZE_BIN=riscv64-unknown-elf-size

# --summary-only：不重跑任何配置，只從現有的 results/*.txt 重新產生對照表。
# 用途：對照表的產生邏輯改了（例如之前漏列 B1C/B1D），不值得為了一張表
# 重跑整輪 sweep（六個配置約五分鐘）。
# 注意：bash 4.2（CentOS 7）在 set -u 下展開空陣列會報 unbound variable，
# 所以下面的迴圈用 ${CFGS[@]+"${CFGS[@]}"} 這個寫法，不能寫成 "${CFGS[@]}"。
SUMMARY_ONLY=0
if [ "${1:-}" = "--summary-only" ]; then
  SUMMARY_ONLY=1
  shift
fi

CFGS=("$@")
if [ "$SUMMARY_ONLY" = "1" ]; then
  CFGS=()
elif [ ${#CFGS[@]} -eq 0 ]; then
  CFGS=(B0 B1)
fi

mkdir -p "$RES"
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }

for C in ${CFGS[@]+"${CFGS[@]}"}; do
  printf "\n\033[1m=== %s ===\033[0m\n" "$C"
  t0=$(date +%s)

  if ! make -C "$TFLM/bare" CONFIG="$C" ARENA_KB="$ARENA_KB" \
       > /tmp/ms_build.$$ 2>&1; then
    red "  編譯失敗"; tail -20 /tmp/ms_build.$$ | sed 's/^/    /'
    continue
  fi
  BIN="$TFLM/bare/build/$C/nn_bare"
  [ -f "$BIN" ] || { red "  找不到 $BIN"; continue; }

  # section 大小。.text 要跟 .rodata 分開 —— .rodata 幾乎都是模型（5.3 MB），
  # 把它算進 code size 會讓 B1 的改善被稀釋到看不見。
  "$SIZE_BIN" -A "$BIN" > "$RES/$C.size.txt" 2>/dev/null \
    || echo "(size 取得失敗：$SIZE_BIN)" > "$RES/$C.size.txt"

  echo "  執行中（約 90 秒）..."
  # </dev/null 不能省：stdout 是 pipe 時 Spike 的 HTIF 會輪詢 stdin
  out="$(cd "$TFLM/bare" && timeout "$TIMEOUT" "$SPIKE" --isa="$SPIKE_ISA" -m64 \
         "build/$C/nn_bare" </dev/null 2>&1)"
  rc=$?
  {
    echo "# configuration: $C"
    echo "# 日期: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# arena: ${ARENA_KB} KiB"
    echo "# spike: --isa=$SPIKE_ISA -m64"
    echo "# 離開碼: $rc"
    echo
    printf '%s\n' "$out"
    echo
    echo "# --- section 大小 ---"
    cat "$RES/$C.size.txt"
  } > "$RES/$C.txt"

  verd="$(grep -oE 'BITEXACT|CLOSE|DIVERGENT' <<<"$out" | head -1)"
  inv="$(sed -n 's/^  Invoke  *\([0-9]*\).*/\1/p' <<<"$out" | head -1)"
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ]; then
    red "  逾時"
  elif [ "$verd" = "BITEXACT" ]; then
    green "  BITEXACT   Invoke ${inv:-?}   (${dt}s)"
  else
    red "  ${verd:-失敗}   Invoke ${inv:-?}"
  fi
  echo "  -> $RES/$C.txt"
done

# ---------------------------------------------------------------- 對照表
{
  echo "# 各配置量測結果"
  echo
  echo "產生時間 $(date '+%Y-%m-%d %H:%M')，arena ${ARENA_KB} KiB，"
  echo "Spike \`--isa=$SPIKE_ISA -m64\`，單位 instret（已退休指令數）。"
  echo
  echo "| config | Invoke | 相對 B0 | AllocateTensors | kernel 合計 | 框架 overhead | .text | 判定 |"
  echo "|:--|---:|---:|---:|---:|---:|---:|:--|"

  # 順序刻意是 B0 -> B1C -> B1D -> B1 -> X1 -> X2：
  # B1C/B1D 是把 B1 的兩項改動拆開的歸因配置，列在 B1 前面才讀得通
  # （報告 §6.4 的加成性檢查就是用這三行）。
  #
  # 之前這裡寫死 B0/B1/X1/X2 四個，所以 B1C/B1D 量了卻不會出現在
  # summary.md 裡 —— 那是個真實踩到的坑：報告要引用時才發現表上沒有。
  # 現在改成「偏好順序 + 其餘自動補上」，量了什麼就一定列得出來。
  ORDER="B0 B1C B1D B1 X1 X2"
  FILES=""
  for C in $ORDER; do
    [ -f "$RES/$C.txt" ] && FILES="$FILES $RES/$C.txt"
  done
  for f in "$RES"/*.txt; do
    case "$(basename "${f%.txt}")" in
      summary|image_check) continue ;;
    esac
    # 已經在偏好順序裡的就不要重複
    case " $FILES " in *" $f "*) continue ;; esac
    # 只收真正的量測檔（有 Invoke 那一行的）
    grep -q '^  Invoke ' "$f" 2>/dev/null && FILES="$FILES $f"
  done

  b0inv=""
  for f in $FILES; do
    [ -f "$f" ] || continue
    C="$(basename "${f%.txt}")"
    inv="$(sed -n 's/^  Invoke  *\([0-9]*\).*/\1/p' "$f" | head -1)"
    alc="$(sed -n 's/^  AllocateTensors(含Prepare)  *\([0-9]*\).*/\1/p' "$f" | head -1)"
    ker="$(sed -n 's/^    其中 kernel 合計  *\([0-9]*\).*/\1/p' "$f" | head -1)"
    ovh="$(sed -n 's/^    框架 overhead  *\([0-9]*\).*/\1/p' "$f" | head -1)"
    txt="$(sed -n 's/^\.text  *\([0-9]*\).*/\1/p' "$f" | head -1)"
    vd="$(grep -oE 'BITEXACT|CLOSE|DIVERGENT' "$f" | head -1)"
    [ "$C" = "B0" ] && b0inv="$inv"
    sp="-"
    if [ -n "$b0inv" ] && [ -n "$inv" ] && [ "$inv" -gt 0 ] 2>/dev/null; then
      sp="$(awk -v a="$b0inv" -v b="$inv" 'BEGIN{printf "%.2fx", a/b}')"
    fi
    echo "| $C | ${inv:-?} | $sp | ${alc:-?} | ${ker:-?} | ${ovh:-?} | ${txt:-?} | ${vd:-?} |"
  done

  echo
  echo "## 讀法"
  echo
  echo "- **Invoke** 是推論本身，各配置要比的就是這一欄。"
  echo "- **B1C / B1D** 把 B1 的兩項改動拆開（只開 1x1 / 只開 depthwise），"
  echo "  用來歸因加速來源。加速比不可相乘，但**節省量可以相加** ——"
  echo "  因為兩項動的是互不相交的 operator。"
  echo "- **AllocateTensors** 含 Prepare 的浮點工作（\`frexp\` + double 除法），"
  echo "  一次性，不該算進加速比。"
  echo "- **框架 overhead** = Invoke 減掉所有 kernel 的和。B0 量到只有數千條"
  echo "  指令（約 0.00002%），所以框架精簡在速度上沒有空間 —— B1 的價值"
  echo "  在 code size。"
  echo "- **.text** 不含模型（模型在 .rodata，約 5.3 MB）。B1 的 code size"
  echo "  改善要看這一欄。"
  echo "- **判定** 必須是 BITEXACT：每個配置都要跟 x86 的 B0 輸出位元相同。"
  echo "  B1/X1 只是換實作，數學上完全等價，差一個 bit 就是寫錯了。"
} > "$RES/summary.md"

printf "\n\033[1m=== 對照表 ===\033[0m\n"
sed -n '/^| config/,/^$/p' "$RES/summary.md"
echo "-> $RES/summary.md"
rm -f /tmp/ms_build.$$