#!/usr/bin/env bash
# run.sh — 確認 toolchain / Spike / 自訂指令 / bare-metal 環境正常。
#
# 設計原則（上一版的教訓）：
#   * 測試程式是磁碟上的靜態 .c 檔，不在腳本裡動態生成
#   * 每一步都印出實際執行的指令，你可以複製出來手動重跑
#   * 每次 spike 呼叫都有 timeout、-m64，以及 </dev/null
#     （stdout 是 pipe 時 Spike 會輪詢 stdin，不關掉就會卡住）
#
# 用法:
#   bash run.sh            全部跑
#   bash run.sh t02        只跑一項
#   bash run.sh -v         印出每個指令

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
ISA="rv64imc_zicsr"
CFLAGS="-march=$ISA -mabi=lp64 -mcmodel=medany -O2 -Wall -nostdlib -nostartfiles -T $HERE/link.ld"
SPIKE_FLAGS="--isa=$ISA -m64"
TIMEOUT=15

VERBOSE=""
ONLY=""
for a in "$@"; do
    case "$a" in
        -v) VERBOSE=1 ;;
        *)  ONLY="$a" ;;
    esac
done

PASS=0; FAIL=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); }
inf()  { printf "        %s\n" "$1"; }
cmd()  { [ -n "$VERBOSE" ] && printf "        \033[90m\$ %s\033[0m\n" "$*"; }

mkdir -p "$BUILD"

# ---------------------------------------------------------------- 工具
echo "=== 工具 ==="
for t in riscv64-unknown-elf-gcc riscv64-unknown-elf-objdump \
         riscv64-unknown-elf-nm spike spike-dasm; do
    if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t 找不到"; fi
done
command -v riscv64-unknown-elf-gcc >/dev/null 2>&1 &&
    inf "$(riscv64-unknown-elf-gcc --version | head -1)"
command -v spike >/dev/null 2>&1 &&
    inf "spike: $(command -v spike)"

echo
echo "=== Spike 的 profiling 選項 ==="
if command -v spike >/dev/null 2>&1; then
    H="$(spike --help 2>&1)"
    for opt in "--log-commits" "--log-cache-miss" "--ic=" "--dc=" "--l2="; do
        printf '%s' "$H" | grep -q -- "$opt" && ok "$opt" || bad "$opt 不存在"
    done
fi

# ---------------------------------------------------------------- 編譯與執行
# run_one <name> <expect: pass|fail>
run_one() {
    local name="$1" expect="$2"
    local src="$HERE/${name}"*.c
    src="$(echo $src)"
    [ -f "$src" ] || { bad "$name: 找不到原始碼"; return; }
    local base="$BUILD/$name"

    cmd "riscv64-unknown-elf-gcc $CFLAGS $src -o $base"
    if ! riscv64-unknown-elf-gcc $CFLAGS "$src" -o "$base" 2>"$base.cc.log"; then
        bad "$name: 編譯失敗"
        sed 's/^/        /' "$base.cc.log" | head -6
        return
    fi
    local warn
    warn="$(grep -c 'warning:' "$base.cc.log" 2>/dev/null)"
    warn="${warn:-0}"
    [ "$warn" -gt 0 ] 2>/dev/null && inf "($warn 個編譯警告，見 $base.cc.log)"

    # </dev/null 不能省：stdout 是 pipe（命令替換）時，Spike 的 HTIF
    # 會去輪詢 stdin 的 console 輸入；stdin 連著終端機就會永遠等下去。
    cmd "timeout $TIMEOUT spike $SPIKE_FLAGS $base </dev/null"
    local out rc
    out="$(timeout $TIMEOUT spike $SPIKE_FLAGS "$base" </dev/null 2>&1)"
    rc=$?

    case "$expect:$rc" in
        pass:0)   ok "$name" ;;
        fail:0)   bad "$name: 預期失敗卻通過 —— 測試無效" ;;
        *:124)    bad "$name: 逾時 ${TIMEOUT}s"
                  inf "手動重跑: spike $SPIKE_FLAGS $base </dev/null" ;;
        fail:*)   ok "$name (預期失敗，確實失敗)" ;;
        pass:*)   bad "$name: exit=$rc"
                  [ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/        /' | head -5
                  inf "手動重跑: spike $SPIKE_FLAGS $base </dev/null" ;;
    esac
}

echo
echo "=== bare-metal 執行 ==="
run_all=1
[ -n "$ONLY" ] && run_all=0

for t in t01_plain t02_xqtest t03_xqtest_neg t04_csr t05_stack; do
    [ $run_all -eq 1 ] || [[ "$t" == "$ONLY"* ]] || continue
    case "$t" in
        t03_*) run_one "$t" fail ;;
        *)     run_one "$t" pass ;;
    esac
done

# ---------------------------------------------------------------- 靜態檢查
if [ $run_all -eq 1 ] && [ -f "$BUILD/t01_plain" ]; then
    echo
    echo "=== 靜態檢查 ==="
    if riscv64-unknown-elf-objdump -d "$BUILD/t01_plain" \
         | grep -qE '^\s+[0-9a-f]+:.*\s(f(add|sub|mul|div|sqrt|cvt|sgnj)|fl[wd]|fs[wd])\b'; then
        bad "偵測到浮點指令"
    else
        ok "t01 不含浮點指令"
    fi

    if [ -f "$BUILD/t02_xqtest" ]; then
        if riscv64-unknown-elf-objdump -d "$BUILD/t02_xqtest" \
             | grep -qE '^\s+[0-9a-f]+:\s+[0-9a-f]{6}0b\s'; then
            ok "t02 含 opcode 0x0b 的自訂指令"
        else
            bad "t02 找不到 opcode 0x0b"
        fi
    fi

    SP="$(riscv64-unknown-elf-nm "$BUILD/t01_plain" | grep ' _stack_top$' | cut -d' ' -f1)"
    [ -n "$SP" ] && ok "_stack_top = 0x$SP" || bad "link.ld 未定義 _stack_top"
fi

# ---------------------------------------------------------------- 工具鏈其餘
if [ $run_all -eq 1 ]; then
    echo
    echo "=== spike-dasm ==="
    if command -v spike-dasm >/dev/null 2>&1; then
        D="$(echo 'DASM(00c5850b)' | spike-dasm 2>&1)"
        printf '%s' "$D" | grep -q xqtest && ok "認得 xqtest: $D" \
                                          || bad "未認出 xqtest: $D"
    fi

    echo
    echo "=== commit log（instruction mix 分析用）==="
    if [ -f "$BUILD/t01_plain" ]; then
        cmd "spike $SPIKE_FLAGS -l --log-commits $BUILD/t01_plain"
        timeout $TIMEOUT spike $SPIKE_FLAGS -l --log-commits \
            "$BUILD/t01_plain" </dev/null > "$BUILD/trace.log" 2>&1
        N="$(wc -l < "$BUILD/trace.log")"
        if [ "$N" -gt 20 ]; then
            ok "trace $N 行 -> $BUILD/trace.log"
            inf "$(sed -n '10p' "$BUILD/trace.log" | cut -c1-72)"
        else
            bad "trace 只有 $N 行"
        fi
    fi

    echo
    echo "=== cache 模型 ==="
    if [ -f "$BUILD/t01_plain" ]; then
        cmd "spike $SPIKE_FLAGS --ic=64:4:64 --dc=64:4:64 --l2=512:8:64 ..."
        timeout $TIMEOUT spike $SPIKE_FLAGS --ic=64:4:64 --dc=64:4:64 \
            --l2=512:8:64 "$BUILD/t01_plain" </dev/null >/dev/null 2>&1
        [ $? -eq 0 ] && ok "cache 參數可接受" || bad "cache 參數被拒"
    fi
fi

echo
echo "============================================================"
printf "PASS %d   FAIL %d\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] && echo "環境正常" || echo "有失敗項目，上面每項都附了手動重跑指令"
echo "產物保留於 $BUILD"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)