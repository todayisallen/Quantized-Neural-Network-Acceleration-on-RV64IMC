/* 標準指令 + stack：乘法(M 擴充)、迴圈、volatile 區域變數 */
#include <stdint.h>
volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;
void cmain(void);
__attribute__((naked, section(".text.init")))
void _start(void){ __asm__ volatile("la sp, _stack_top\n\tj cmain"); }
static void finish(int pass){ tohost = pass ? 1 : 3; for(;;); }

void cmain(void){
    uint64_t a = 1234, b = 5678, d = a * b;
    volatile uint64_t s = 0;
    for (int i = 0; i < 100; i++) s += i;
    finish(d == 7006652u && s == 4950u);
}
