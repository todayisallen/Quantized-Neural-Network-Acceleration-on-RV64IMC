/* stack 深度：200 層遞迴，每層 128 bytes 區域陣列 (~25 KB)
 * TFLM 是 C++，呼叫深度不會比這淺。 */
#include <stdint.h>
volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;
void cmain(void);
__attribute__((naked, section(".text.init")))
void _start(void){ __asm__ volatile("la sp, _stack_top\n\tj cmain"); }
static void finish(int pass){ tohost = pass ? 1 : 3; for(;;); }

static uint64_t rec(int n){
    volatile uint64_t buf[16];
    for (int i = 0; i < 16; i++) buf[i] = (uint64_t)(n + i);
    return n ? rec(n - 1) + buf[0] : buf[0];
}
void cmain(void){ finish(rec(200) > 0); }
