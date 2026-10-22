/* CSR 計數器：Phase 3 的指令數統計靠 rdinstret */
#include <stdint.h>
volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;
void cmain(void);
__attribute__((naked, section(".text.init")))
void _start(void){ __asm__ volatile("la sp, _stack_top\n\tj cmain"); }
static void finish(int pass){ tohost = pass ? 1 : 3; for(;;); }

static inline uint64_t rdinstret_(void){ uint64_t x;
    __asm__ volatile("rdinstret %0" : "=r"(x)); return x; }
static inline uint64_t rdcycle_(void){ uint64_t x;
    __asm__ volatile("rdcycle %0" : "=r"(x)); return x; }
void cmain(void){
    uint64_t i0 = rdinstret_(), c0 = rdcycle_();
    volatile uint64_t s = 0;
    for (int i = 0; i < 100; i++) s += i;
    uint64_t di = rdinstret_() - i0;
    uint64_t dc = rdcycle_()  - c0;
    finish(s == 4950u && di > 100 && dc > 0);
}
