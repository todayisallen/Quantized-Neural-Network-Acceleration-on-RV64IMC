#!/usr/bin/env bash
# setup_env.sh — 從零建立本專案所需的外部環境。
#
# ======================================================================
# 為什麼要這個腳本
# ======================================================================
# 這個專案依賴三樣 repo 外的東西：改過的 Spike、特製的 TFLM target、
# 以及（可選的）medany toolchain。README 把步驟寫得很清楚，但「照著
# 十幾個指令打一遍」跟「跑一行」的差別，在別人的機器上就是能不能重現。
#
# 每個階段都可以單獨重跑，而且**跑完會自己驗證**再回報成功 ——
# 失敗就停在原地，不會一路壞下去讓你在三步之後才看到莫名的錯誤。
#
# ======================================================================
# 用法
# ======================================================================
#   bash scripts/setup_env.sh check       秒級    盤點缺什麼（不安裝）
#   bash scripts/setup_env.sh python      ~2 分   Python 套件
#   bash scripts/setup_env.sh spike       ~15 分  Spike + 本專案的自訂指令
#   bash scripts/setup_env.sh tflm        ~10 分  TFLM + 特製 target + 兩個庫
#   bash scripts/setup_env.sh gen         ~5 分   產生 gen/ 與 golden
#   bash scripts/setup_env.sh verify      ~5 分   跑完整驗證
#   bash scripts/setup_env.sh all                 以上全部（跳過 toolchain）
#
#   bash scripts/setup_env.sh toolchain   數小時  medany toolchain（**通常不需要**）
#
# toolchain 那階段通常不需要：本專案附了一份 medany 建的 libgcc.a
# （prebuilt/），而那是標準 toolchain 唯一缺的東西。詳見 README。
#
# ======================================================================
# 可覆寫的路徑
# ======================================================================
#   WORK=...            外部原始碼放哪（預設 $HOME/ca-project/work）
#   RISCV_PREFIX=...    Spike 裝到哪（預設 $HOME/ca-project/riscv）
#   RISCV_MEDANY=...    toolchain 裝到哪（預設 $HOME/ca-project/riscv-medany）
#   JOBS=8              平行編譯數

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_DIR="$(cd "$HERE/.." && pwd)"          # repo/tflm
REPO="$(cd "$TFLM_DIR/.." && pwd)"          # repo

WORK="${WORK:-$HOME/ca-project/work}"
RISCV_PREFIX="${RISCV_PREFIX:-$HOME/ca-project/riscv}"
RISCV_MEDANY="${RISCV_MEDANY:-$HOME/ca-project/riscv-medany}"
JOBS="${JOBS:-8}"
MODEL="${MODEL:-$REPO/model/eff-lite0/model.tflite}"

# 釘住的版本。換版本的話捨入模式可能變（見 README 的「捨入模式」一節），
# `xqrq` 就要跟著改 —— 所以這兩個 commit 是重現條件，不是建議。
TFLM_COMMIT="64b3de8"
SPIKE_COMMIT="f11c7b6"
MODEL_SHA256="bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16"

PY_PKGS=("ai-edge-litert==1.4.0" "numpy==2.2.6" "pillow==12.2.0")

# ----------------------------------------------------------------------
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
die()   { red "!! $*"; exit 1; }

step()  { printf "\n"; bold "=== $* ==="; }
ok()    { green "   ok: $*"; }

need() { command -v "$1" >/dev/null 2>&1 || die "找不到 $1，請先安裝"; }

# ======================================================================
# 環境檢查 —— 不安裝任何東西，只盤點缺什麼、該去哪拿。
#
# 刻意放在 all 的最前面：RISC-V toolchain 這個依賴本腳本**不負責安裝**，
# 而缺了它會讓 tflm 階段的 riscv64 庫建不起來、verify 階段的自訂指令與
# 完整模型全部失敗。跑到第三階段才發現，前面已經白花十幾分鐘建 Spike。
# ======================================================================
stage_check() {
  step "環境檢查"
  local hard=0

  _have() {  # $1=指令 $2=說明 $3=取得方式（空=本腳本會處理）
    if command -v "$1" >/dev/null 2>&1; then
      printf "   \033[32mok\033[0m   %-28s %s\n" "$1" "$($1 --version 2>&1 | head -1 | cut -c1-48)"
      return 0
    fi
    printf "   \033[31m缺\033[0m   %-28s %s\n" "$1" "$2"
    [ -n "${3:-}" ] && printf "        取得方式：%s\n" "$3"
    return 1
  }

  _have git    "版本控制" "系統套件管理員" || hard=1
  _have make   "建置"     "系統套件管理員" || hard=1
  _have g++    "host 編譯器（需支援 C++17）" "系統套件管理員" || hard=1
  _have python3 "產生 gen/ 需要" "系統套件管理員" || hard=1

  # host 編譯器要真的支援 C++17（CentOS 7 內建的 4.8.5 不夠）
  if command -v g++ >/dev/null 2>&1; then
    local t
    t="$(mktemp /tmp/cxx17probe.XXXXXX.cc)"
    printf '#include <type_traits>\nint main(){ if constexpr(true){} return 0; }\n' > "$t"
    if g++ -std=c++17 -fsyntax-only "$t" 2>/dev/null; then
      printf "   \033[32mok\033[0m   %-28s\n" "g++ -std=c++17"
    else
      printf "   \033[31m缺\033[0m   %-28s %s\n" "g++ -std=c++17" "這個 g++ 不支援 C++17"
      printf "        用 CXX=/path/to/newer/g++ 指定，或裝新版\n"
      hard=1
    fi
    rm -f "$t"
  fi

  # ---- 本腳本**不安裝**的那一個 ------------------------------------
  #
  # 刻意**不**用 command -v 查 PATH：bare/Makefile 有自己的解析順序
  # （明確指定 > 開發機的 ~/ca-project/riscv-medany > PATH），而它不一定
  # 落在 PATH 上。查 PATH 的話會出現「check 報 ok、建置卻 Command not
  # found」—— 有檢查卻檢查錯東西比沒檢查更糟。
  #
  # 所以問 Makefile 本人要它實際會用的那個 $(CXX)。單一真相來源。
  local rvcxx=""
  rvcxx="$(make -C "$TFLM_DIR/bare" --no-print-directory print-cxx 2>/dev/null | tail -1)"
  if [ -n "$rvcxx" ] && command -v "$rvcxx" >/dev/null 2>&1; then
    printf "   \033[32mok\033[0m   %-28s %s\n" "riscv toolchain" \
           "$("$rvcxx" --version 2>&1 | head -1 | cut -c1-44)"
    printf "        %s\n" "$rvcxx"
  else
    printf "   \033[31m缺\033[0m   %-28s %s\n" "riscv toolchain" \
           "RISC-V 交叉編譯器（本腳本不安裝）"
    [ -n "$rvcxx" ] && printf "        bare/Makefile 會用：%s\n" "$rvcxx"
    hard=1
    cat <<'EOF'

        本腳本不負責安裝 RISC-V toolchain。取得方式（都是分鐘級）：

          Debian / Ubuntu
            sudo apt install gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf

          預編 release
            https://github.com/riscv-collab/riscv-gnu-toolchain/releases
            下載後解壓，把 bin/ 放進 PATH

        **標準版就夠** —— 不需要 --with-cmodel=medany 的特製版本。
        bare/Makefile 會偵測編譯器的預設 code model，是 medlow 就自動
        改用專案附的 prebuilt/libgcc-medany-rv64imc-lp64.a，不必設
        任何環境變數。

        缺了它仍然可以驗證：kernel 等價性（tests/kernel/run.sh）與
        x86 上的完整模型。但 bare-metal 的所有效能數字都量不到。
EOF
  fi

  # toolchain 有的話，順手報一下 code model —— 它決定會不會用到
  # prebuilt/ 的 libgcc，是最容易踩到卻最難看懂的那個失敗點。
  if [ -n "$rvcxx" ] && command -v "$rvcxx" >/dev/null 2>&1; then
    local cm
    cm="$("$rvcxx" -Q --help=target 2>/dev/null \
          | sed -n 's/^[[:space:]]*-mcmodel=[[:space:]]*\(.*\)$/\1/p' | head -1)"
    case "$cm" in
      medany) printf "   \033[32mok\033[0m   %-28s %s\n" "code model" \
                     "medany —— 用 toolchain 自己的 libgcc" ;;
      "")     printf "   \033[33m??\033[0m   %-28s %s\n" "code model" \
                     "探測不出來；連結失敗時見 README 的「關於 medany」" ;;
      *)      if [ -f "$TFLM_DIR/prebuilt/libgcc-medany-rv64imc-lp64.a" ]; then
                printf "   \033[32mok\033[0m   %-28s %s\n" "code model" \
                       "$cm —— 會自動改用 prebuilt/ 的 libgcc"
              else
                printf "   \033[31m缺\033[0m   %-28s %s\n" "code model" \
                       "$cm，而 prebuilt/libgcc-medany-rv64imc-lp64.a 不在"
                printf "        這個組合連結不到 0x80000000。見 README「關於 medany」\n"
                hard=1
              fi ;;
    esac
  fi

  # WORK 被覆寫時，bare/host 的 TFLM_ROOT 預設值就不對了 —— 它寫死
  # $(HOME)/ca-project/work/tflite-micro，跟 WORK 的預設值一致，所以
  # 預設情況沒事，覆寫了才斷。非致命，但一定要講，因為症狀是
  # 「找不到 libmicrolite」，看不出跟 WORK 有關。
  if [ "$WORK" != "$HOME/ca-project/work" ]; then
    printf "   \033[33m注意\033[0m %-26s\n" "WORK 被覆寫成 $WORK"
    printf "        bare/ 與 host/ 的 TFLM_ROOT 預設是 \$HOME/ca-project/work/tflite-micro，\n"
    printf "        所以 make 時要一起帶：TFLM_ROOT=%s/tflite-micro\n" "$WORK"
  fi

  printf "\n"
  if [ "$hard" != "0" ]; then
    red "!! 有缺的依賴（見上）。補齊之後重跑。"
    return 1
  fi
  ok "依賴齊全"
}

# ======================================================================
stage_python() {
  step "Python 套件"
  need python3
  python3 -m pip --version >/dev/null 2>&1 || die "python3 沒有 pip"

  # --only-binary 不能省：ai-edge-litert 從原始碼建會需要 bazel。
  python3 -m pip install --user --only-binary=:all: "${PY_PKGS[@]}" \
    || die "pip 安裝失敗"

  python3 - <<'EOF' || die "套件裝了但 import 不起來"
import numpy, PIL
from ai_edge_litert.interpreter import Interpreter, OpResolverType
print("   ai_edge_litert / numpy %s / pillow %s" % (numpy.__version__, PIL.__version__))
EOF
  ok "Python 環境就緒"
}

# ======================================================================
stage_spike() {
  step "Spike（含本專案的三條自訂指令）"
  need git; need make; need g++
  local src="$WORK/riscv-isa-sim"
  local patch="$TFLM_DIR/patches/spike-xq.patch"

  [ -f "$patch" ] || die "找不到 $patch"

  mkdir -p "$WORK"
  if [ ! -d "$src/.git" ]; then
    git clone https://github.com/riscv-software-src/riscv-isa-sim.git "$src" \
      || die "clone 失敗"
  fi

  cd "$src"
  git checkout "$SPIKE_COMMIT" 2>/dev/null || die "checkout $SPIKE_COMMIT 失敗"

  # ---- 套 patch ----------------------------------------------------
  # 已經套過就不要再套（git apply 不是 idempotent 的）。
  #
  # patch 必須用 `git add -A; git diff --cached` 產生，**不能**用
  # `git add -N; git diff` —— 後者在 git 1.8.3（CentOS 7 內建）上會把新檔
  # 記成「對空檔案的修改」（index e69de29..., 缺 new file mode），
  # git apply 於是回報 "No such file or directory"。實際踩過。
  if git apply --check --reverse "$patch" >/dev/null 2>&1; then
    ok "patch 已套用過，跳過"
  elif git apply "$patch" 2>/dev/null; then
    ok "已套用 spike-xq.patch"
  else
    red "!! git apply 失敗（git $(git --version | awk '{print $3}')）"
    echo "   會改用直接複製 insn 原始碼的方式繼續，並單獨檢查另兩處改動。"
    echo "   patch 本身若要能用，請以 git add -A; git diff --cached 重新產生。"
  fi

  # ---- 確保三個 insn 檔到位 ----------------------------------------
  # 不論 patch 成功與否都檢查。交件項目 3 要的是「source codes」，
  # 所以原始碼也直接收在 repo 裡，patch 只是方便。
  local insns="$TFLM_DIR/patches/spike/insns"
  local i
  for i in xqtest xqmac8 xqrq; do
    if [ ! -f "riscv/insns/$i.h" ]; then
      [ -f "$insns/$i.h" ] \
        || die "缺 riscv/insns/$i.h，而 $insns/$i.h 也不存在"
      cp "$insns/$i.h" "riscv/insns/$i.h" || die "複製 $i.h 失敗"
      ok "複製 $i.h（patch 未建立）"
    fi
  done

  # ---- encoding.h 與 riscv.mk.in 只能靠 patch，確認它們生效 --------
  grep -q 'MATCH_XQRQ'   riscv/encoding.h \
    || die "riscv/encoding.h 缺 MATCH_XQRQ —— patch 沒套成功，見上面的說明"
  grep -q 'MATCH_XQMAC8' riscv/encoding.h \
    || die "riscv/encoding.h 缺 MATCH_XQMAC8 —— patch 沒套成功"
  grep -q 'xqrq'         riscv/riscv.mk.in \
    || die "riscv/riscv.mk.in 缺 xqrq —— patch 沒套成功"
  ok "encoding.h 與 riscv.mk.in 的改動已就位"

  # --enable-commitlog 不能省：scripts/instr_mix.sh 需要 -g 的 PC histogram
  mkdir -p build && cd build
  if [ ! -f Makefile ]; then
    ../configure --prefix="$RISCV_PREFIX" --enable-commitlog --without-boost-asio \
      || die "configure 失敗"
  fi
  make -j"$JOBS" || die "建置失敗"
  make install   || die "安裝失敗"

  # 驗證：三條指令都要編進去
  local miss=""
  for i in xqtest xqmac8 xqrq; do
    [ -f "$i.o" ] || miss="$miss $i"
  done
  [ -z "$miss" ] || die "這幾條指令沒被編進去：$miss（riscv.mk.in 的改動沒生效？）"

  "$RISCV_PREFIX/bin/spike" --help >/dev/null 2>&1 \
    || die "$RISCV_PREFIX/bin/spike 跑不起來"

  ok "spike 已裝到 $RISCV_PREFIX/bin/spike"
  echo "   把它放進 PATH：export PATH=$RISCV_PREFIX/bin:\$PATH"
}

# ======================================================================
stage_tflm() {
  step "TFLM（含特製的 riscv64 target）"
  need git; need make
  local src="$WORK/tflite-micro"
  local mk="tensorflow/lite/micro/tools/make/Makefile"

  mkdir -p "$WORK"
  if [ ! -d "$src/.git" ]; then
    git clone https://github.com/tensorflow/tflite-micro.git "$src" || die "clone 失敗"
  fi
  cd "$src"
  git checkout "$TFLM_COMMIT" 2>/dev/null || die "checkout $TFLM_COMMIT 失敗"

  make -f "$mk" third_party_downloads || die "third_party_downloads 失敗"

  # ---- 套 tflm-patch ------------------------------------------------
  # 兩個檔，放置路徑由 TFLM 的 Makefile 決定：
  #   PATH_TO_TARGET_SRCS := tensorflow/lite/micro/$(TARGET)
  # 所以 TARGET=riscv64_generic 時，debug_log.cc 要放在那個同名目錄下，
  # specialize_files.py 才會用它替換掉原本的版本。
  mkdir -p tensorflow/lite/micro/riscv64_generic
  cp "$TFLM_DIR/tflm-patch/debug_log.cc" \
     tensorflow/lite/micro/riscv64_generic/debug_log.cc || die "複製 debug_log.cc 失敗"
  cp "$TFLM_DIR/tflm-patch/riscv64_generic_makefile.inc" \
     tensorflow/lite/micro/tools/make/targets/ || die "複製 target .inc 失敗"
  ok "已套用 tflm-patch（debug_log.cc + riscv64_generic target）"

  # ---- x86 的庫（host 驗證用）---------------------------------------
  make -f "$mk" TARGET=linux TARGET_ARCH=x86_64 microlite -j"$JOBS" \
    || die "x86 microlite 建置失敗"
  local xlib
  xlib="$(find gen -path '*linux_x86_64*' -name 'libtensorflow-microlite*.a' | head -1)"
  [ -n "$xlib" ] || die "找不到 x86 的 libtensorflow-microlite.a"
  ok "x86 庫：$xlib"

  # ---- riscv64 的庫（bare-metal 用）--------------------------------
  # 這一步需要 PATH 上有 riscv64-unknown-elf-*。
  # 注意 target 的 .inc 明確帶了 -mcmodel=medany，所以**標準 toolchain
  # 也能建**，不需要 medany 版的編譯器 —— 只有最後連結時需要 medany 的
  # libgcc.a，而本專案附了一份（見 README）。
  if ! command -v riscv64-unknown-elf-g++ >/dev/null 2>&1; then
    red "!! PATH 上找不到 riscv64-unknown-elf-g++，跳過 riscv64 的庫"
    echo "   裝好 riscv toolchain 之後重跑：bash scripts/setup_env.sh tflm"
    return 1
  fi
  # TARGET_ARCH=riscv64 不能省。
  #
  # tflm-patch/riscv64_generic_makefile.inc 裡雖然寫了 TARGET_ARCH := riscv64，
  # 但 TFLM 的 Makefile 在 include target 的 .inc **之前**就算好了
  #   GENDIR := gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)
  # 所以那個賦值對產物路徑無效，會得到 gen/riscv64_generic_x86_64_default_gcc/。
  # 命令列傳的變數（origin 'command line'）才蓋得過去。
  # 同理 .inc 的 BUILD_TYPE := micro 也沒生效 —— 實際是 default_gcc，
  # 這裡刻意不傳 BUILD_TYPE，以重現本專案實際使用的那組設定。
  make -f "$mk" TARGET=riscv64_generic TARGET_ARCH=riscv64 microlite -j"$JOBS" \
    || die "riscv64 microlite 建置失敗"
  local rlib
  rlib="$(find gen -path '*riscv64*' -name 'libtensorflow-microlite*.a' | head -1)"
  [ -n "$rlib" ] || die "找不到 riscv64 的 libtensorflow-microlite.a"
  ok "riscv64 庫：$rlib"

  echo "   設給後續步驟用：export TFLM_ROOT=$src"
}

# ======================================================================
stage_toolchain() {
  step "medany toolchain（數小時 —— 通常不需要，見 README）"
  need git; need make
  local src="$WORK/riscv-gnu-toolchain"

  bold "   這一步要幾小時。本專案附了 prebuilt/ 下的 medany libgcc.a，"
  bold "   用它就不需要自建 toolchain。確定要建的話按 Enter，否則 Ctrl-C。"
  read -r _ || true

  mkdir -p "$WORK"
  if [ ! -d "$src/.git" ]; then
    git clone --recursive https://github.com/riscv-collab/riscv-gnu-toolchain.git "$src" \
      || die "clone 失敗"
  fi
  cd "$src"
  # 這組參數與本專案實際使用的那套完全相同
  ./configure --prefix="$RISCV_MEDANY" \
              --with-arch=rv64imc --with-abi=lp64 \
              --with-cmodel=medany --disable-gdb || die "configure 失敗"
  make -j"$JOBS" || die "建置失敗"

  local gpp="$RISCV_MEDANY/bin/riscv64-unknown-elf-g++"
  [ -x "$gpp" ] || die "建好了但找不到 $gpp"
  local m
  m="$("$gpp" -Q --help=target 2>/dev/null | \
       sed -n 's/^[[:space:]]*-mcmodel=[[:space:]]*\(.*\)$/\1/p' | head -1)"
  [ "$m" = "medany" ] || die "預設 code model 是 '$m'，不是 medany —— configure 參數有問題"
  ok "medany toolchain 已裝到 $RISCV_MEDANY"
}

# ======================================================================
stage_gen() {
  step "產生 gen/（模型的 C 陣列、固定輸入、golden）"
  need python3
  [ -f "$MODEL" ] || die "找不到模型 $MODEL"

  # 模型必須是被記錄的那一份 —— 所有效能數字都是用它量的，
  # 而本專案的 BITEXACT 全是相對檢查：兩邊讀同一份模型檔的話，
  # 模型本身有問題會一致地錯，檢查照樣通過。
  if command -v sha256sum >/dev/null 2>&1; then
    local got
    got="$(sha256sum "$MODEL" | cut -d' ' -f1)"
    [ "$got" = "$MODEL_SHA256" ] \
      || die "模型 sha256 不符：$got（應為 $MODEL_SHA256）"
    ok "模型 sha256 相符"
  fi

  cd "$TFLM_DIR"
  python3 tools/model_info.py  "$MODEL" --out gen || die "model_info.py 失敗"
  python3 tools/gen_model_c.py "$MODEL" --out gen || die "gen_model_c.py 失敗"
  python3 tools/gen_io.py      "$MODEL" --out gen --seed 0 || die "gen_io.py 失敗"

  # golden 要換成 TFLM@x86 的輸出，不是 LiteRT 的。
  # 理由：Spike 上要驗的是「同一套 C++ 在兩個 ISA 上是否位元相同」，
  # 那必須零容忍。留著 LiteRT 的 golden 的話，bare-metal 會帶著跨框架
  # 那幾個 LSB 差異，就分不出那是正常繼承還是移植寫錯。
  make -C host CONFIG=B0 run >/dev/null 2>&1 \
    || die "x86 的 B0 跑不起來 —— 先確認 TFLM_ROOT 設對了（make -C host paths）"
  [ -s host/tflm_out.bin ] || die "沒有產生 host/tflm_out.bin"
  python3 tools/gen_golden_from_bin.py host/tflm_out.bin --out gen --config B0 \
    || die "golden 交換失敗"

  # 還原/產生之後必須讓 gen/ 比物件檔新，否則 make 認為不用重編。
  find gen -type f -exec touch {} + 2>/dev/null || true

  grep -q 'random seed=0' gen/input_data.h  || die "input_data.h 的來源不對"
  grep -q 'TFLM@x86'      gen/golden_data.h || die "golden_data.h 的來源不對"
  ok "gen/ 就緒（輸入 = random seed=0，golden = TFLM@x86(B0)）"
}

# ======================================================================
stage_verify() {
  step "驗證"
  cd "$TFLM_DIR"

  bold "   (1) kernel 等價性（x86，秒級）"
  bash tests/kernel/run.sh || die "kernel 等價性測試失敗"

  bold "   (2) 自訂指令（Spike）"
  need riscv64-unknown-elf-gcc
  # -fno-tree-loop-distribute-patterns 不能省：t_xqmac8.c 的 splat() 裡有一個
  # 填值迴圈，GCC 會把它認成 memset 的 pattern 而產生對 memset 的呼叫 ——
  # 但 -nostdlib 沒有 libc，於是連結時 undefined reference to `memset'。
  # 這跟 bare/Makefile 為 libc_min.cc 關掉同一個轉換是同一個理由。
  riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
    -O2 -nostdlib -nostartfiles \
    -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns \
    -T "$REPO/tests/env/link.ld" \
    bare/t_xqmac8.c -o /tmp/t_xqmac8 || die "t_xqmac8 編譯失敗"
  spike --isa=rv64imc_zicsr -m64 /tmp/t_xqmac8 </dev/null | tail -3
  make -C bare t_xqrq_run || die "t_xqrq 失敗"

  bold "   (3) 完整模型（Spike，幾分鐘）"
  make -C bare CONFIG=X2 run || die "X2 不是 BITEXACT"

  ok "全部通過"
  echo
  echo "   接著可以跑完整量測："
  echo "     bash scripts/measure.sh B0 B1 B1C B1D X1 X2"
  echo "     bash scripts/instr_mix.sh X2"
  echo "     bash scripts/verify_image.sh B0 B1 X1 X2"
}

# ======================================================================
usage() {
  sed -n '3,45p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

[ $# -ge 1 ] || usage

# 每個階段的結束狀態。腳本沒開 set -e（階段內部要自己決定什麼算致命），
# 所以回傳值得自己收 —— 不收的話失敗的階段後面照樣印「完成」。
RC=0

for arg in "$@"; do
  case "$arg" in
    python)    stage_python     || RC=1 ;;
    spike)     stage_spike      || RC=1 ;;
    tflm)      stage_tflm       || RC=1 ;;
    toolchain) stage_toolchain   || RC=1 ;;
    gen)       stage_gen        || RC=1 ;;
    verify)    stage_verify     || RC=1 ;;
    check)     stage_check      || RC=1 ;;
    all)
      stage_check || die "環境檢查未通過，先補齊依賴"
      stage_python
      stage_spike
      stage_tflm
      stage_gen
      stage_verify
      ;;
    *) red "未知的階段：$arg"; usage ;;
  esac
done

printf "\n"
if [ "$RC" = "0" ]; then
  green "完成。"
else
  red "未完成 —— 上面有階段失敗了。"
fi
exit "$RC"