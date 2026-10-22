/* t_console.c — 單獨驗 HTIF console，不牽涉 TFLM。
 *
 * 目的：把「HTIF 協定對不對」跟「5 MB 的 TFLM 移植對不對」分開。
 * 兩個未知放在一起除錯會非常痛苦（上次 selftest_env 卡住就是這樣）。
 *
 * 建置（用 tests/env/link.ld，那份已經驗證過）：
 *   riscv64-unknown-elf-gcc -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
 *     -O2 -nostdlib -nostartfiles -T ../../tests/env/link.ld \
 *     t_console.c -o /tmp/t_console
 *
 * 執行：
 *   timeout 10 spike --isa=rv64imc_zicsr -m64 /tmp/t_console </dev/null
 *   echo "exit=$?"
 *
 * 預期：螢幕出現 HTIF console OK ，離開碼 0。
 *
 * 失敗的意義：
 *   逾時（124）、沒有輸出  -> 協定錯，或 sp 沒設好
 *   有輸出但離開碼不是 0   -> 結束協定錯
 *   完全沒輸出但離開碼 0   -> console 不通但結束可用，
 *                             那就先用離開碼帶訊息，不要靠 print
 */

#include <stdint.h>

volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;

static void htif_send(uint64_t dev, uint64_t cmd, uint64_t payload) {
    tohost = (dev << 56) | (cmd << 48) | (payload & 0xffffffffffffULL);
    /* 等 tohost 歸零，不是等 fromhost。
       fesvr/device.cc 的 bcd_t::handle_write 不呼叫 respond()，
       所以 console 寫入永遠不會設 fromhost；等 fromhost 會永遠等下去。
       fesvr/htif.cc 的主迴圈讀到非零的 tohost 後會立刻寫回 0。 */
    while (tohost != 0) { }
}

static void puts_(const char *s) {
    for (; *s; ++s) htif_send(1, 1, (uint64_t)(uint8_t)*s);   /* dev 1 = console, cmd 1 = write */
}

void cmain(void) {
    puts_("HTIF console OK\n");
    tohost = (0 << 1) | 1;          /* dev 0, cmd 0, payload 奇數 = 結束，離開碼 = payload>>1 */
    for (;;) { }
}

__attribute__((naked, section(".text.init")))
void _start(void) {
    __asm__ volatile("la sp, _stack_top\n\tj cmain");
}
