#!/usr/bin/env bash
# verify_image.sh — 用一張「答案已知」的真實圖片做絕對正確性檢查。
#
# 用法:
#   bash scripts/verify_image.sh                 # B1
#   bash scripts/verify_image.sh B0 B1 X1
#   EXPECT=281 IMAGE=my_cat.jpg bash scripts/verify_image.sh B1
#
# 產出:
#   results/image_check.txt
#
# ======================================================================
# 為什麼需要這個
# ======================================================================
# 專案裡其他所有正確性檢查都是**相對的**：
#   RV64 ≡ x86          證明移植對
#   TFLM ≡ LiteRT       證明 kernel 對
#   新 kernel ≡ 參考     證明重寫對
#
# 三個都成立，模型檔本身還是可能是錯的 —— 被截斷、版本不對、
# 或根本不是我們以為的那個模型。因為 x86 與 RV64 讀同一份 bytes，
# 錯的模型會讓兩邊**一致地錯**，BITEXACT 照樣成立。
#
# 而所有效能數字（B0 的 19.6 B、B1 的 3.7 B、385,187,552 個 MAC）
# 都是在那份模型檔上量的。模型不是我們以為的東西，數字就在描述別的東西。
#
# 測試圖用 TFLite label_image 範例的 grace_hopper：期望答案有文件記載
# （範例 README 列 top-1 = 653 military uniform，那是 1001 類索引；
# 我們的模型是 1000 類，所以是 652）。這是已知答案測試，不是
# 「自己找一張照片然後覺得結果看起來對」。
#
# ======================================================================
# 三種比較，三種容忍度 —— 這是本腳本的核心，不能混
# ======================================================================
#   (1) LiteRT 的 top-1 vs 圖片的已知標籤
#       -> 絕對檢查。只看 top-1，不看位元。
#
#   (2) TFLM@x86 vs LiteRT
#       -> **跨框架**比較，沒有先驗的容忍門檻。差幾個 LSB 是正常的
#          （不同的 requantize 捨入路徑）。只記錄，**不是驗收條件**。
#
#   (3) TFLM@RV64 vs TFLM@x86
#       -> 同一套 C++ 跑在兩個 ISA 上，必須 BITEXACT，零容忍。
#          這才是驗收條件。
#
# 所以中間必須有一步「golden 交換」：gen_io.py 產生的 golden 是 LiteRT 的
# 輸出，拿它去驗 RV64 等於讓 RV64 跟 LiteRT 比，那是 (2) 不是 (3)，
# RV64 會帶著跨框架的 LSB 差異，於是分不出「正常繼承」和「移植寫錯」——
# 這兩件事的處理方式完全不同。
#
# （本腳本的第一版就漏了這一步，結果 Spike 被判成失敗，
#   實際上那是 TFLM 與 LiteRT 的正常差異。）
#
# ======================================================================
# gen/ 的還原
# ======================================================================
# 這個腳本會覆寫 gen/，跑完再換回去。用 trap 還原，所以 Ctrl-C 或
# 中途失敗也會還原 —— 忘記還原的話 verify_seeds.sh 會失敗、
# 已落地的量測數字跟 gen/ 不一致，而那種不一致很難追。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM="$(cd "$HERE/.." && pwd)"
cd "$TFLM"

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
IMAGE="${IMAGE:-tests/data/grace_hopper.bmp}"
IMAGE_URL="${IMAGE_URL:-https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp}"
EXPECT="${EXPECT:-652}"
NORM="${NORM:-pm1}"
ARENA_KB="${ARENA_KB:-1713}"
RES="$TFLM/results"
OUT="$RES/image_check.txt"

CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(B1)

red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
bold()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

# nn_main 的判定行一律從第 0 欄開始，是 BITEXACT / CLOSE / DIVERGENT 之一。
# 刻意用這個而不是 make 的離開碼：host/Makefile 把 0 與 2 都當成功，
# 分不出 BITEXACT 與 CLOSE，而這裡的差別正是重點。
verdict() { grep -oE '^(BITEXACT|CLOSE|DIVERGENT)' "$1" | head -1; }

[ -f "$MODEL" ] || { red "找不到模型 $MODEL"; echo "用 MODEL=... 指定"; exit 1; }

if [ ! -f "$IMAGE" ]; then
  echo "沒有 $IMAGE，試著抓下來..."
  mkdir -p "$(dirname "$IMAGE")"
  if ! curl -sSfL -o "$IMAGE" "$IMAGE_URL"; then
    red "抓不到測試圖。"
    echo "手動放一張到 $IMAGE，或用 IMAGE=... EXPECT=<類別索引> 指定自己的。"
    exit 1
  fi
  echo "  -> $IMAGE ($(wc -c < "$IMAGE") bytes)"
fi

# ---------------------------------------------------------------- gen/ 備份
BAK=""
restore() {
  if [ -n "$BAK" ] && [ -d "$BAK" ]; then
    rm -rf "$TFLM/gen"; mv "$BAK" "$TFLM/gen"
    # 還原後必須讓 gen/ 比 build/ 裡的物件檔新，否則 make 認為物件是
    # 最新的、不重編 —— 執行檔裡留著本腳本用的真實圖片輸入，而磁碟上
    # 的 gen/ 是還原後的內容，兩者不一致且沒有任何警告。
    #
    # 這是實際踩到的：本腳本跑完後 X1 量到 1,590,059,459（真實圖片的值）
    # 而不是 1,590,153,616（種子 0 的值）。原因是 cp -r 不保留 mtime，
    # 備份的時間戳是腳本「開始」的時間，比中途重建出的物件檔還舊。
    find "$TFLM/gen" -type f -exec touch {} + 2>/dev/null || true
    echo "gen/ 已還原成原本的內容（已更新 mtime 以強制重編）"
  fi
}
# PIPE 與 HUP 不能省。實際踩過：用 `bash scripts/verify_image.sh B0 | head -5`
# 看輸出時，head 提早關閉管線使 bash 收到 SIGPIPE 而直接結束 ——
# SIGPIPE 沒被 trap 的話 EXIT trap 不會執行，gen/ 就留在被覆寫的狀態，
# 而且沒有任何訊息說它沒還原。
trap restore EXIT INT TERM HUP PIPE

if [ -d gen ]; then
  BAK="$TFLM/gen.before_image.$$"
  cp -r gen "$BAK"
  echo "gen/ 已備份到 $(basename "$BAK")（結束時自動還原）"
fi

mkdir -p "$RES"
{
  echo "# 真實圖片的絕對正確性檢查"
  echo "# 日期:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# 模型:   $MODEL"
  echo "# 圖片:   $IMAGE"
  echo "# 前處理: --norm $NORM"
  echo "# 期望:   class $EXPECT"
  echo "# 配置:   ${CFGS[*]}"
  echo
} > "$OUT"

TMP="$RES/.imgchk.$$"; mkdir -p "$TMP"
cleanup_tmp() { rm -rf "$TMP"; }
trap 'cleanup_tmp; restore' EXIT INT TERM HUP PIPE

# ================================================================ (1) LiteRT
bold "=== (1) LiteRT：模型本身對不對 ==="
LOG="$TMP/litert.log"
python3 tools/gen_io.py "$MODEL" --out gen --image "$IMAGE" --norm "$NORM" \
    2>&1 | tee "$LOG"
TOP1=$(grep -m1 '^  1\. class' "$LOG" | awk '{print $3}')
{ echo "=== (1) LiteRT ==="; cat "$LOG"; echo; } >> "$OUT"

if [ "$TOP1" != "$EXPECT" ]; then
  red "LiteRT top-1 = $TOP1，期望 $EXPECT"
  echo "先別管效能數字。兩個方向："
  echo "  1. 前處理：NORM=raw 或 NORM=unit 再試"
  echo "  2. 模型檔本身有問題"
  echo "結果: 失敗 —— LiteRT top-1=$TOP1 期望=$EXPECT" >> "$OUT"
  exit 1
fi
green "LiteRT top-1 = $TOP1，符合已知標籤 -> 模型是可用的"

# ============================================ (2) TFLM@x86 vs LiteRT（只記錄）
#
# golden 的來源配置與「要驗的配置」刻意分開。
# 所有配置的輸出位元相同（那正是要驗的事），所以 golden 來自哪個都行；
# 用 B0 最乾淨 —— 它是未修改的 TFLM，不含任何自訂 kernel。
#
# 不能用 CFGS[0]：X1 的 .insn 是 RISC-V 專屬，host 組不起來。
REF="${REF_CFG:-B0}"
bold "=== (2) TFLM@x86 vs LiteRT（跨框架，只記錄不驗收）==="
echo "  參考配置：$REF（golden 的來源）"
LOG="$TMP/x_${REF}_vs_litert.log"
if ! make -C host CONFIG="$REF" ARENA_KB="$ARENA_KB" run >"$LOG" 2>&1; then
  red "x86 $REF 跑不起來"; tail -20 "$LOG" | sed 's/^/  /'
  { echo "=== (2) x86 $REF ==="; cat "$LOG"; } >> "$OUT"
  exit 1
fi
grep -E '^(BITEXACT|CLOSE|DIVERGENT)|比對|差異分布' "$LOG" | sed 's/^/  /'
XFW=$(verdict "$LOG")
echo "  跨框架判定: ${XFW:-?}（這一項不是驗收條件）"
{ echo "=== (2) TFLM@x86 vs LiteRT（跨框架，僅記錄）==="; cat "$LOG"; echo; } >> "$OUT"

BIN="$TFLM/host/tflm_out.bin"
[ -s "$BIN" ] || { red "找不到 $BIN，無法交換 golden"; exit 1; }

# ============================================================ (3) golden 交換
bold "=== (3) 把 golden 換成 TFLM@x86 的輸出 ==="
python3 tools/gen_golden_from_bin.py "$BIN" --out gen --config "$REF" 2>&1 | sed 's/^/  /'
echo "  之後所有比較都是「同一套 C++ 在兩個 ISA 上是否位元相同」，零容忍"
{ echo "=== (3) golden := TFLM@x86 的輸出（$REF）==="; echo; } >> "$OUT"

# =================================================== (4) 每個配置，兩個平台
fail=0
for C in "${CFGS[@]}"; do
  for PLAT in host bare; do
    name=$([ "$PLAT" = host ] && echo "x86" || echo "RV64/Spike")
    bold "=== (4) $C on $name：必須 BITEXACT ==="
    LOG="$TMP/${C}_${PLAT}.log"
    make -C "$PLAT" CONFIG="$C" ARENA_KB="$ARENA_KB" run >"$LOG" 2>&1
    V=$(verdict "$LOG")
    # 有些配置在 host 上本來就建置不了（X1 的 .insn 是 RISC-V 專屬）。
    # 那不是失敗 —— RV64 對 x86 的位元比較靠 golden 完成，golden 已經
    # 是 TFLM@x86 的輸出了。跳過並註明即可。
    if [ "$PLAT" = host ] && [ -z "$V" ] && grep -q 'host 編不了' "$LOG"; then
      echo "  （$C 在 x86 上無法建置，略過；驗收由 RV64 對 golden 的比較完成）"
      { echo "=== (4) $C on $name：略過（host 無法建置）==="; echo; } >> "$OUT"
      continue
    fi
    grep -E '^(BITEXACT|CLOSE|DIVERGENT)|比對|top-1 相同|Invoke ' "$LOG" | sed 's/^/  /'
    if [ "$V" = "BITEXACT" ]; then
      green "  $C on $name: BITEXACT"
    else
      red "  $C on $name: ${V:-沒有判定行}"
      tail -20 "$LOG" | sed 's/^/    /'
      fail=1
    fi
    { echo "=== (4) $C on $name ==="; cat "$LOG"; echo; } >> "$OUT"
  done
done

# ---------------------------------------------------------------- 結論
echo
if [ "$fail" -eq 0 ]; then
  {
    echo "結果: 通過"
    echo
    echo "  (1) LiteRT top-1 = $EXPECT，符合 grace_hopper 的已知標籤"
    echo "      -> 模型檔是可用的訓練模型，不是損壞或版本不符的檔案。"
    echo "         這是專案裡唯一不依賴「另一個實作」的檢查。"
    echo
    echo "  (2) TFLM@x86 vs LiteRT: ${XFW:-?}"
    echo "      -> 跨框架差異，記錄用。沒有先驗門檻，不是驗收條件。"
    echo
    echo "  (3)(4) golden 換成 TFLM@x86 的輸出之後，"
    echo "      ${CFGS[*]} 在 x86 與 RV64 上全部 BITEXACT。"
    echo "      -> 這比「固定種子的隨機輸入下 BITEXACT」強：同一張"
    echo "         分類正確的真實圖片，所有配置在兩個 ISA 上算出完全"
    echo "         相同的 bytes。"
  } >> "$OUT"
  green "通過 —— 真實圖片分類正確，且所有配置在兩個 ISA 上位元一致"
else
  echo "結果: 有配置沒有達到 BITEXACT，詳見本檔各段" >> "$OUT"
  red "有失敗"
fi
echo "完整紀錄: $OUT"
exit $fail