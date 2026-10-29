#!/usr/bin/env bash
# verify_seeds.sh — 用多組輸入驗證移植正確性，並順便收集跨框架一致性數據。
#
# 每一個種子跑三件事：
#   1. TFLM-x86  vs  LiteRT        跨框架差異（報告要用的數字）
#   2. TFLM-x86  vs  TFLM-x86      確認換 golden 的工具沒錯（必須 BITEXACT）
#   3. TFLM-RV64 vs  TFLM-x86      移植正確性（必須 BITEXACT）
#
# 用法:
#   bash scripts/verify_seeds.sh            # 種子 0 1 2 3
#   bash scripts/verify_seeds.sh 0 5 7      # 指定種子
#   SKIP_BARE=1 bash scripts/verify_seeds.sh    # 只跑 x86（快，除錯用）
#
# 產出:
#   verify_seeds.md    markdown 表格，報告可直接引用
#
# 每個種子約 100 秒（bare-metal 推論佔 91 秒），四個約 7 分鐘。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM="$(cd "$HERE/.." && pwd)"
# 預設用 repo 內的那份模型，不是家目錄下的開發用副本。
#
# 為什麼重要：README 記錄了 repo 內那份的 sha256，而所有效能數字必須是
# 用「被記錄的那一份」量出來的。預設指向 repo 外的副本會讓兩者漂移 ——
# 兩份現在位元相同，但改到任一邊就分岔，而這種分岔不會有任何錯誤訊息。
#
# 本專案所有的 BITEXACT 都是相對檢查（RV64 ≡ x86、TFLM ≡ LiteRT），
# 兩邊讀同一份模型檔的話，模型本身有問題會一致地錯 —— 所以「是哪一份
# 模型」必須是確定的，不能靠慣例。
MODEL="${MODEL:-$TFLM/../model/eff-lite0/model.tflite}"
ARENA_KB="${ARENA_KB:-1713}"
SPIKE_ISA="${SPIKE_ISA:-rv64imc_zicsr}"
# 用哪一個 spike。樹裡可能有多個版本（刻意不覆蓋舊的，好讓舊量測能重現），
# 所以讓它可覆寫：SPIKE=/path/to/spike bash scripts/...
SPIKE="${SPIKE:-spike}"
OUT="$TFLM/verify_seeds.md"

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3)

[ -f "$MODEL" ] || { echo "找不到模型: $MODEL"; exit 1; }

red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
head2() { printf "\n\033[1m=== %s ===\033[0m\n" "$1"; }

# model_data.cc 是 30 MB 的文字檔，編譯要一兩分鐘。
# 種子換掉時它不會變，所以只刪會變的那幾個 object，不要 make clean。
# 讓 gen/ 的 mtime 比任何物件檔新，強制下一次 make 重編。
#
# 以前這裡是刪除特定的 .o 路徑，但那些路徑是 build/ 還沒分配置子目錄
# 時寫的（現在是 build/<CONFIG>/），所以它刪不到任何東西 ——
# 一個無聲失效的防護措施比沒有防護更危險，因為你以為被保護了。
# touch 與 build 的目錄結構無關，不會再因為改目錄佈局而失效。
#
# 參數保留（呼叫端傳 host/bare 的路徑），但實際不需要用到。
refresh() {
  : "${1:-}"
  find "$TFLM/gen" -type f -exec touch {} + 2>/dev/null || true
}

# 從 nn_main 的輸出抓數字
f_arena()  { sed -n 's/.*arena_used_bytes = \([0-9]*\).*/\1/p'      <<<"$1" | head -1; }
f_mism()   { sed -n 's/.*不一致 \([0-9]*\) .*/\1/p'                  <<<"$1" | head -1; }
f_maxd()   { sed -n 's/.*最大差 \([0-9-]*\).*/\1/p'                  <<<"$1" | head -1; }
f_order()  { sed -n 's/.*同序 \([0-9]*\)\/\([0-9]*\).*/\1\/\2/p'     <<<"$1" | head -1; }
f_set()    { sed -n 's/.*同集合 \([0-9]*\)\/\([0-9]*\).*/\1\/\2/p'   <<<"$1" | head -1; }
f_verd()   { grep -oE 'BITEXACT|CLOSE|DIVERGENT'                     <<<"$1" | head -1; }
f_top1()   { sed -n 's/^top-1 \(.*\)$/\1/p'                          <<<"$1" | head -1; }

rows=()
fails=0
t_all=$(date +%s)

for s in "${SEEDS[@]}"; do
  head2 "seed $s"
  t0=$(date +%s)

  # ---------------------------------------------------------------- 1. 跨框架
  echo "[1/3] 產生輸入與 LiteRT 參考輸出"
  if ! python3 "$TFLM/tools/gen_io.py" "$MODEL" --out "$TFLM/gen" \
         --seed "$s" > /tmp/vs_genio.$$ 2>&1; then
    red "  gen_io.py 失敗"; sed 's/^/    /' /tmp/vs_genio.$$ | head -10
    fails=$((fails+1)); continue
  fi

  refresh "$TFLM/host"
  if ! make -C "$TFLM/host" ARENA_KB="$ARENA_KB" > /tmp/vs_mk.$$ 2>&1; then
    red "  host 編譯失敗"; tail -15 /tmp/vs_mk.$$ | sed 's/^/    /'
    fails=$((fails+1)); continue
  fi
  # cd 進 host 是因為 plat_dump_output 把 tflm_out.bin 寫在 cwd
  x1="$(cd "$TFLM/host" && ./build/nn_host 2>&1)"
  arena="$(f_arena "$x1")"; xmism="$(f_mism "$x1")"; xmaxd="$(f_maxd "$x1")"
  xord="$(f_order "$x1")";  xset="$(f_set "$x1")";   xverd="$(f_verd "$x1")"
  xtop1="$(f_top1 "$x1")"
  mkdir -p "$TFLM/logs"
  printf '%s\n' "$x1" > "$TFLM/logs/x86_vs_litert_seed${s}.txt"
  echo "      vs LiteRT: 不一致 ${xmism:-?}，最大差 ${xmaxd:-?}，top-1 ${xtop1:-?}，top-5 同序 ${xord:-?} → ${xverd:-?}"

  if [ ! -s "$TFLM/host/tflm_out.bin" ]; then
    red "  沒有產生 tflm_out.bin，無法繼續"; fails=$((fails+1)); continue
  fi

  # ---------------------------------------------------------------- 2. 換 golden
  echo "[2/3] 把 golden 換成 TFLM-x86 的輸出，x86 自我比對"
  python3 "$TFLM/tools/gen_golden_from_bin.py" "$TFLM/host/tflm_out.bin" \
          --out "$TFLM/gen" --config B0 > /dev/null 2>&1
  refresh "$TFLM/host"
  make -C "$TFLM/host" ARENA_KB="$ARENA_KB" > /dev/null 2>&1
  x2="$(cd "$TFLM/host" && ./build/nn_host 2>&1)"
  sverd="$(f_verd "$x2")"
  if [ "$sverd" = "BITEXACT" ]; then
    green "      x86 自我比對 BITEXACT"
  else
    red "      x86 自我比對不是 BITEXACT（$sverd）—— 換 golden 的工具有問題"
    printf '%s\n' "$x2" | grep -E '不一致|分布' | sed 's/^/        /'
    fails=$((fails+1))
  fi

  # ---------------------------------------------------------------- 3. RV64
  bverd="skip"; bmism="-"
  if [ -z "${SKIP_BARE:-}" ]; then
    echo "[3/3] RV64 bare-metal vs x86（約 90 秒）"
    refresh "$TFLM/bare"
    if ! make -C "$TFLM/bare" ARENA_KB="$ARENA_KB" > /tmp/vs_mkb.$$ 2>&1; then
      red "  bare 編譯失敗"; tail -15 /tmp/vs_mkb.$$ | sed 's/^/    /'
      fails=$((fails+1)); continue
    fi
    # </dev/null 不能省：stdout 是 pipe 時 Spike 的 HTIF 會輪詢 stdin（ENV_NOTES §4.1）
    b="$(cd "$TFLM/bare" && timeout 1800 "$SPIKE" --isa="$SPIKE_ISA" -m64 \
           build/nn_bare </dev/null 2>&1)"
    brc=$?
    bverd="$(f_verd "$b")"; bmism="$(f_mism "$b")"
    barena="$(f_arena "$b")"
    # 完整輸出留檔：裡面有每層的 instret，是 B1/X1/X2 的對照基準
    mkdir -p "$TFLM/logs"
    printf '%s\n' "$b" > "$TFLM/logs/bare_seed${s}.txt"
    if [ "$brc" -eq 124 ]; then
      red "      逾時"; bverd="TIMEOUT"; fails=$((fails+1))
    elif [ "$bverd" = "BITEXACT" ]; then
      green "      RV64 BITEXACT（arena ${barena:-?}）"
      [ "$barena" = "$arena" ] || red "      !! arena 與 x86 不同：$barena vs $arena"
    else
      red "      RV64 不是 BITEXACT（$bverd）"
      printf '%s\n' "$b" | grep -E '不一致|分布|未實作' | sed 's/^/        /'
      fails=$((fails+1))
    fi
  else
    echo "[3/3] 跳過（SKIP_BARE=1）"
  fi

  dt=$(( $(date +%s) - t0 ))
  rows+=("| $s | ${xmism:-?} | ${xmaxd:-?} | ${xtop1:-?} | ${xord:-?} | ${xset:-?} | ${xverd:-?} | ${sverd:-?} | ${bverd} | ${dt}s |")
done

# ---------------------------------------------------------------- 回到種子 0
if [ -z "${NO_RESTORE:-}" ]; then
  head2 "把 gen/ 復原成種子 0 的狀態"
  python3 "$TFLM/tools/gen_io.py" "$MODEL" --out "$TFLM/gen" --seed 0 >/dev/null 2>&1
  refresh "$TFLM/host"
  make -C "$TFLM/host" ARENA_KB="$ARENA_KB" >/dev/null 2>&1
  (cd "$TFLM/host" && ./build/nn_host >/dev/null 2>&1)
  python3 "$TFLM/tools/gen_golden_from_bin.py" "$TFLM/host/tflm_out.bin" \
          --out "$TFLM/gen" --config B0 >/dev/null 2>&1
  refresh "$TFLM/host"; refresh "$TFLM/bare"
  echo "gen/ 的 golden 現在是種子 0 的 TFLM-x86 輸出"
fi

# ---------------------------------------------------------------- 報表
{
  echo "# 移植正確性驗證（多輸入）"
  echo
  echo "模型 \`$(basename "$MODEL")\`，arena ${ARENA_KB} KiB，產生時間 $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "輸入是固定種子的隨機 uint8（\`[1,224,224,3]\`，150,528 bytes）。"
  echo "刻意不用真實照片：這一步驗的是整數運算的等價性，而隨機資料會讓每層的"
  echo "激活值鋪滿動態範圍，比平滑的照片更容易踩到 requantize 的平手點。"
  echo
  echo "| seed | vs LiteRT 不一致 | 最大差 | top-1 | top-5 同序 | 同集合 | 判定 | x86 自比 | RV64 vs x86 | 耗時 |"
  echo "|---:|---:|---:|:--:|:--:|:--:|:--|:--|:--|---:|"
  printf '%s\n' "${rows[@]}"
  echo
  echo "## 怎麼讀"
  echo
  echo "- **vs LiteRT**：兩套獨立實作的差異，來自 requantize 平手值的進位策略不同，"
  echo "  隨網路深度累積。判定看的是 **top-1 是否保持 + 不一致的比例**，不是最大差 ——"
  echo "  最大差會隨輸入在 2~4 之間跳動，拿它當門檻只是在對單一樣本過擬合。"
  echo "- **top-5 同序**：隨機輸入會讓分數幾乎打平（例如 53/21/19/17/13），"
  echo "  1 個 LSB 就足以讓相鄰名次對調。所以同序偶爾不是 5/5 屬正常，"
  echo "  同集合才是穩定的指標。真實圖片的 top-1 分離度高得多。"
  echo "- **x86 自比**：把 TFLM-x86 的輸出存下來當 golden 再跟自己比，必須 BITEXACT。"
  echo "  這一欄只確認換 golden 的工具沒寫錯，不是獨立的結果。"
  echo "- **RV64 vs x86**：同一套 C++ 在兩個架構上的結果，**必須 BITEXACT**。"
  echo "  這是唯一有原則的驗收條件，零容忍。"
  echo
  echo "每次 bare-metal 執行的完整輸出（含每層 instret）留在 \`logs/bare_seed<N>.txt\`。"
  echo
} > "$OUT"

head2 "結果"
cat "$OUT" | sed -n '/^| seed/,/^$/p'
echo "報表 -> $OUT"
echo "總耗時 $(( $(date +%s) - t_all )) 秒"
rm -f /tmp/vs_genio.$$ /tmp/vs_mk.$$ /tmp/vs_mkb.$$

if [ "$fails" -eq 0 ]; then
  green "全部通過"
  exit 0
fi
red "有 $fails 項失敗"
exit 1