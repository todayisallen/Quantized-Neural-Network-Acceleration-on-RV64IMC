/* 自訂指令 xqtest：rd = rs1 + rs2，custom-0 opcode 0x0b */
#include <stdint.h>
volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;
void cmain(void);
__attribute__((naked, section(".text.init")))
void _start(void){ __asm__ volatile("la sp, _stack_top\n\tj cmain"); }
static void finish(int pass){ tohost = pass ? 1 : 3; for(;;); }

#define XQTEST(d,a,b) __asm__ volatile(".insn r 0x0b,0,0,%0,%1,%2" \
                                       : "=r"(d) : "r"(a), "r"(b))
void cmain(void){
    uint64_t d, a = 1234, b = 5678;
    XQTEST(d, a, b);
    finish(d == a + b);
}
