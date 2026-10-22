#!/usr/bin/env bash
# instr_mix.sh — 用 Spike 的 PC histogram 產生指令組成分析。
#
# 用法:
#   bash scripts/instr_mix.sh            # B0 B1
#   bash scripts/instr_mix.sh B0 B1 X1
#
# 產出:
#   results/<CONFIG>.hist       Spike 的 PC histogram（原始）
#   results/<CONFIG>.dis        反組譯
#   results/<CONFIG>.mix.txt    指令組成報告
#
# 為什麼用 histogram 而不是 --log-commits:
#   B0 的 Invoke 有 19.58 B 條指令，commit log 會是 TB 級。
#   histogram 只記「每個 PC 執行幾次」，檔案大小跟程式碼量成正比
#   （我們的 .text 約 120 KB，所以幾萬行），跟執行次數無關，
#   而且結果是精確的，不是取樣估計。
#
# ** 時間 **：-g 會讓 Spike 每條指令多做一次 hash map 更新，
# 實測前預估單一配置要 10–20 分鐘（不開是 90 秒）。兩個配置可能半小時。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM="$(cd "$HERE/.." && pwd)"
ARENA_KB="${ARENA_KB:-1713}"
SPIKE_ISA="${SPIKE_ISA:-rv64imc_zicsr}"
TIMEOUT="${TIMEOUT:-3600}"
MACS="${MACS:-385187552}"
RES="$TFLM/results"

RISCV_MEDANY="${RISCV_MEDANY:-$HOME/ca-project/riscv-medany}"
OBJDUMP="$RISCV_MEDANY/bin/riscv64-unknown-elf-objdump"
NM="$RISCV_MEDANY/bin/riscv64-unknown-elf-nm"
command -v "$OBJDUMP" >/dev/null 2>&1 || OBJDUMP=riscv64-unknown-elf-objdump
command -v "$NM" >/dev/null 2>&1 || NM=riscv64-unknown-elf-nm

CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(B0 B1)
mkdir -p "$RES"
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }

# ---------------------------------------------------------------- 先確認 -g 可用
# 探測必須用「會正常結束」的程式。
# histogram 是在 processor_t 的解構子裡印的（riscv/processor.cc），
# 所以把 Spike 中途 kill 掉就什麼都不會印 —— 那會被誤判成「不支援」。
# tests/env/build/t01_plain 跑完就結束，是正確的探測對象。
PROBE="$TFLM/../tests/env/build/t01_plain"
if [ ! -f "$PROBE" ]; then
  echo "找不到 $PROBE"
  echo "先跑一次 bash tests/env/run.sh 產生它（那套自檢的副產物）。"
  exit 1
fi
echo "確認 Spike 的 -g 是否可用..."
spike -g --isa="$SPIKE_ISA" -m64 "$PROBE" </dev/null >/dev/null 2>/tmp/ix_probe.$$
if ! grep -q 'PC Histogram' /tmp/ix_probe.$$; then
  red "這版 Spike 沒有輸出 PC Histogram。"
  echo
  echo "先確認原始碼有沒有這個功能："
  echo "  grep -n 'PC Histogram' ~/ca-project/work/riscv-isa-sim/riscv/processor.cc"
  echo
  echo "有的話是編譯選項問題，重建（不會動到現有的 spike）："
  echo "  cd ~/ca-project/work/riscv-isa-sim && mkdir -p build-hist && cd build-hist"
  echo "  ../configure --prefix=\$RISCV --enable-commitlog --without-boost-asio"
  echo "  make -j16"
  echo "  ./spike -g --isa=$SPIKE_ISA -m64 $PROBE </dev/null 2>&1 | grep -c Histogram"
  echo "  make install          # 確認上一行印 1 之後再做"
  echo "  cd ~/ca-project/repo/tests/env && bash run.sh    # 要 21/21"
  rm -f /tmp/ix_probe.$$
  exit 1
fi
green "可用（$(grep -c . /tmp/ix_probe.$$) 行 histogram）"
rm -f /tmp/ix_probe.$$

for C in "${CFGS[@]}"; do
  printf "\n\033[1m=== %s ===\033[0m\n" "$C"
  BIN="$TFLM/bare/build/$C/nn_bare"
  if [ ! -f "$BIN" ]; then
    echo "  建置 $C ..."
    make -C "$TFLM/bare" CONFIG="$C" ARENA_KB="$ARENA_KB" >/tmp/ix_mk.$$ 2>&1 || {
      red "  編譯失敗"; tail -15 /tmp/ix_mk.$$ | sed 's/^/    /'; continue; }
  fi

  echo "  反組譯與符號表..."
  "$OBJDUMP" -d "$BIN" > "$RES/$C.dis" 2>/dev/null
  "$NM" -n "$BIN" > "$RES/$C.nm" 2>/dev/null

  echo "  Spike -g 執行中（可能 10–20 分鐘）..."
  t0=$(date +%s)
  # histogram 走 stderr，程式自己的輸出走 stdout，兩邊分開存。
  # </dev/null 照例不能省。
  timeout "$TIMEOUT" spike -g --isa="$SPIKE_ISA" -m64 "$BIN" \
      </dev/null > "$RES/$C.ghist.stdout" 2> "$RES/$C.hist"
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ]; then
    red "  逾時（${TIMEOUT}s）。用 TIMEOUT=7200 再試。"
    continue
  fi
  if ! grep -q 'PC Histogram' "$RES/$C.hist"; then
    red "  沒有 histogram 輸出"; continue
  fi
  green "  完成 (${dt}s)，histogram $(wc -l < "$RES/$C.hist") 行"

  python3 "$TFLM/tools/instr_mix.py" \
      --hist "$RES/$C.hist" --objdump "$RES/$C.dis" --nm "$RES/$C.nm" \
      --macs "$MACS" > "$RES/$C.mix.txt" 2>&1
  if [ $? -ne 0 ]; then
    red "  分析失敗"; head -10 "$RES/$C.mix.txt" | sed 's/^/    /'; continue
  fi
  {
    echo "# configuration: $C"
    echo "# 日期: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# MAC 總數: $MACS"
    echo "# 注意：histogram 涵蓋整個程式（含 AllocateTensors 與啟動碼），"
    echo "#       不只 Invoke。B0 實測 Invoke 占 99.92%，污染可忽略。"
    echo
    cat "$RES/$C.mix.txt"
  } > "$RES/$C.mix.tmp" && mv "$RES/$C.mix.tmp" "$RES/$C.mix.txt"

  echo "  -> $RES/$C.mix.txt"
  sed -n '/整體指令組成/,/^$/p' "$RES/$C.mix.txt" | head -20
done

rm -f /tmp/ix_mk.$$
echo
echo "完整報告在 $RES/*.mix.txt"