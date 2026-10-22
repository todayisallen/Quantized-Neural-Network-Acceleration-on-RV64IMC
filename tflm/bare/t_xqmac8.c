/* t_xqmac8.c — 單獨驗 xqmac8 指令，不牽涉 TFLM。
 *
 * 跟 t_console.c 一樣刻意自給自足：把「指令對不對」跟「kernel 用得對不對」
 * 分開。兩個未知放在一起除錯會很痛苦。
 *
 * 建置（用 tests/env/link.ld，那份已經驗證過）：
 *   cd ~/ca-project/repo/tflm/bare
 *   $HOME/ca-project/riscv-medany/bin/riscv64-unknown-elf-gcc \
 *     -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
 *     -O2 -nostdlib -nostartfiles -ffreestanding \
 *     -T ../../tests/env/link.ld t_xqmac8.c -o /tmp/t_xqmac8
 *
 * 執行（先用新建的 spike，不要用已安裝的）：
 *   ~/ca-project/work/riscv-isa-sim/build-xqmac8/spike \
 *     --isa=rv64imc_zicsr -m64 /tmp/t_xqmac8 </dev/null
 *   echo "exit=$?"
 *
 * 離開碼：0 = 全部通過，n = 失敗 n 項（上限 100）。
 *
 * 失敗的意義：
 *   illegal instruction          -> riscv.mk.in 的指令清單漏了 xqmac8，
 *                                   或 MATCH/MASK 打錯
 *   ACCUM 失敗                   -> 實作是覆寫 rd 而不是累加（最常見）
 *   LANE 失敗                    -> lane 抽取的順序或符號擴展錯
 *   SIGN 失敗                    -> 遮罩與轉型的順序錯（要先遮再轉 int8）
 *   RANDOM 失敗但 directed 全過   -> 某個值域邊界沒處理，看印出的第一個反例
 */

#include <stdint.h>

/* ------------------------------------------------------------------ HTIF */

volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;

static void htif_send(uint64_t dev, uint64_t cmd, uint64_t payload) {
    tohost = (dev << 56) | (cmd << 48) | (payload & 0xffffffffffffULL);
    /* 等 tohost 歸零，不是等 fromhost —— bcd_t::handle_write 不呼叫
       respond()，等 fromhost 會永遠等下去。 */
    while (tohost != 0) { }
}

static void putc_(char c) { htif_send(1, 1, (uint64_t)(uint8_t)c); }

static void puts_(const char *s) { for (; *s; ++s) putc_(*s); }

static void put_i64(int64_t v) {
    char buf[24];
    int n = 0;
    if (v < 0) { putc_('-'); }
    uint64_t u = (v < 0) ? (uint64_t)(-(v + 1)) + 1u : (uint64_t)v;
    do { buf[n++] = (char)('0' + (u % 10u)); u /= 10u; } while (u);
    while (n) putc_(buf[--n]);
}

static void put_hex64(uint64_t u) {
    puts_("0x");
    for (int i = 15; i >= 0; --i) {
        unsigned d = (unsigned)((u >> (4 * i)) & 0xf);
        putc_((char)(d < 10 ? '0' + d : 'a' + d - 10));
    }
}

/* ------------------------------------------------------------- 被測指令 */

/* 刻意用與 app/conv_xq.cc 完全相同的 inline asm 寫法，
   這樣也順便驗證那邊的運算元約束（"+r" 代表讀寫，不是只寫）。 */
static inline int64_t xqmac8(int64_t acc, uint64_t a, uint64_t b) {
    __asm__ volatile(".insn r 0x0b, 1, 0, %0, %1, %2"
                     : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

/* 純量參考實作。xqmac8 必須與這個逐位元相同。 */
static int64_t ref(uint64_t a, uint64_t b) {
    int64_t s = 0;
    for (int i = 0; i < 8; ++i) {
        int32_t x = (int8_t)(uint8_t)(a >> (8 * i));
        int32_t y = (int8_t)(uint8_t)(b >> (8 * i));
        s += (int64_t)x * (int64_t)y;
    }
    return s;
}

/* lane i 放在 bits [8i+7 : 8i] */
static uint64_t pack(const int8_t v[8]) {
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i)
        r |= ((uint64_t)(uint8_t)v[i]) << (8 * i);
    return r;
}

static uint64_t splat(int8_t x) {
    int8_t v[8];
    for (int i = 0; i < 8; ++i) v[i] = x;
    return pack(v);
}

/* --------------------------------------------------------------- 計分板 */

static int g_fail = 0;
static int g_pass = 0;

static void check(const char *tag, int64_t got, int64_t want,
                  uint64_t a, uint64_t b) {
    if (got == want) {
        ++g_pass;
        return;
    }
    ++g_fail;
    puts_("  FAIL ");
    puts_(tag);
    puts_("  got=");   put_i64(got);
    puts_("  want=");  put_i64(want);
    puts_("\n       rs1="); put_hex64(a);
    puts_(" rs2=");         put_hex64(b);
    putc_('\n');
}

/* ------------------------------------------------------ directed 測試 */

static void directed(void) {
    puts_("-- directed --\n");

    /* 1. 零 */
    check("ZERO", xqmac8(0, 0, 0), 0, 0, 0);

    /* 2. 單一 lane */
    {
        int8_t a[8] = {1,0,0,0,0,0,0,0};
        int8_t b[8] = {1,0,0,0,0,0,0,0};
        check("UNIT", xqmac8(0, pack(a), pack(b)), 1, pack(a), pack(b));
    }

    /* 3. 全 1 */
    check("ONES", xqmac8(0, splat(1), splat(1)), 8, splat(1), splat(1));

    /* 4. 正負相消 —— 若符號擴展錯，這個會變成很大的正數 */
    {
        int8_t a[8] = {1,-1,1,-1,1,-1,1,-1};
        uint64_t pa = pack(a), pb = splat(1);
        check("CANCEL", xqmac8(0, pa, pb), 0, pa, pb);
    }

    /* 5. 負 x 負 = 正 */
    check("NEGNEG", xqmac8(0, splat(-1), splat(-1)), 8,
          splat(-1), splat(-1));

    /* 6. 最大正積 127*127*8 */
    check("MAXPOS", xqmac8(0, splat(127), splat(127)), 129032,
          splat(127), splat(127));

    /* 7. 最小積 -128*127*8 */
    check("MINNEG", xqmac8(0, splat(-128), splat(127)), -130048,
          splat(-128), splat(127));

    /* 8. -128 * -128 —— 模型不會產生（權重對稱量化不會是 -128），
          但硬體必須對整個 int8 值域正確。 */
    check("M128SQ", xqmac8(0, splat(-128), splat(-128)), 131072,
          splat(-128), splat(-128));

    /* 9. lane 對應關係 —— 這組才抓得到「lane 順序顛倒」。
          對稱的測資（全 1、全 127）即使順序錯也會通過。 */
    {
        int8_t a[8] = {1,2,3,4,5,6,7,8};
        uint64_t pa = pack(a);
        for (int i = 0; i < 8; ++i) {
            int8_t b[8] = {0,0,0,0,0,0,0,0};
            b[i] = 1;
            uint64_t pb = pack(b);
            check("LANE", xqmac8(0, pa, pb), (int64_t)(i + 1), pa, pb);
        }
        /* 全部 lane 一起：1+2+...+8 = 36 */
        check("LANESUM", xqmac8(0, pa, splat(1)), 36, pa, splat(1));
    }

    /* 10. 符號擴展：0x80 這個 byte 必須被當成 -128，不是 +128 */
    {
        int8_t a[8] = {-128,0,0,0,0,0,0,0};
        int8_t b[8] = {1,0,0,0,0,0,0,0};
        uint64_t pa = pack(a), pb = pack(b);
        check("SIGN", xqmac8(0, pa, pb), -128, pa, pb);
    }

    /* 11. 累加而非覆寫 —— 整個設計的前提。
           實作若寫成 rd = ... 而不是 rd += ...，只有這一項會抓到。 */
    {
        int64_t acc = 5;
        acc = xqmac8(acc, splat(1), splat(1));      /* 5 + 8  = 13 */
        acc = xqmac8(acc, splat(1), splat(1));      /* 13 + 8 = 21 */
        acc = xqmac8(acc, splat(2), splat(3));      /* 21 + 48 = 69 */
        check("ACCUM", acc, 69, splat(2), splat(3));
    }

    /* 12. 負的初值 */
    check("ACCNEG", xqmac8(-100, splat(1), splat(1)), -92,
          splat(1), splat(1));

    /* 12b. 累加器必須是 64 位元寬。
            初值取 INT32_MAX，加 8 之後若累加器只有 32 位元會變成負數。
            （這個 bug 原本只有 randomized 抓得到 —— 確定性的邊界
             不該靠隨機去碰。） */
    check("ACC32", xqmac8((int64_t)2147483647, splat(1), splat(1)),
          (int64_t)2147483655, splat(1), splat(1));

    /* 12c. 8 個 lane 的加總不能用 int16 做。
            單一乘積的範圍是 [-16256, +16384]，int16 裝得下（所以每個
            lane 的乘法器只要 16 位元）；但 8 個加總是 [-130048, +131072]，
            int16 會溢位。這一項把兩者分開。 */
    check("SUM16", xqmac8(0, splat(127), splat(126)), 128016,
          splat(127), splat(126));

    /* 13. rd = x0：寫入必須被丟棄，而且不能影響後續 */
    {
        uint64_t a = splat(127), b = splat(127);
        __asm__ volatile(".insn r 0x0b, 1, 0, x0, %0, %1" :: "r"(a), "r"(b));
        int64_t z;
        __asm__ volatile("mv %0, x0" : "=r"(z));
        check("RDZERO", z, 0, a, b);
    }

    /* 14. 大量累加不溢位：K=1152 的最壞情形是 144 次
           （這裡做 144 次 -128*-128，= 144 * 131072） */
    {
        int64_t acc = 0;
        for (int i = 0; i < 144; ++i)
            acc = xqmac8(acc, splat(-128), splat(-128));
        check("NOOVF", acc, (int64_t)144 * 131072,
              splat(-128), splat(-128));
    }
}

/* ---------------------------------------------------- randomized 測試 */

static uint64_t g_s = 0x243f6a8885a308d3ull;

static uint64_t rnd(void) {
    /* xorshift64*，不需要 libc */
    g_s ^= g_s >> 12;
    g_s ^= g_s << 25;
    g_s ^= g_s >> 27;
    return g_s * 2685821657736338717ull;
}

/* 均勻隨機幾乎碰不到值域端點，所以刻意偏向邊界 */
static int8_t rnd_lane(void) {
    uint64_t r = rnd();
    if ((r & 3) == 0) {
        static const int8_t edge[6] = {-128, -127, -1, 0, 1, 127};
        return edge[(r >> 8) % 6u];
    }
    return (int8_t)(uint8_t)(r >> 16);
}

static uint64_t rnd_word(void) {
    int8_t v[8];
    for (int i = 0; i < 8; ++i) v[i] = rnd_lane();
    return pack(v);
}

static void randomized(int n) {
    puts_("-- randomized x");
    put_i64(n);
    puts_(" --\n");
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t a = rnd_word(), b = rnd_word();
        int64_t acc0 = (int64_t)rnd();          /* 任意初值，驗累加 */
        acc0 >>= 20;                            /* 縮小避免無意義的大數 */
        int64_t got = xqmac8(acc0, a, b);
        int64_t want = acc0 + ref(a, b);
        if (got != want) {
            if (bad < 3) {                      /* 只印前 3 個反例 */
                puts_("  FAIL RANDOM #");
                put_i64(i);
                puts_("  got="); put_i64(got);
                puts_("  want="); put_i64(want);
                puts_("\n       rs1="); put_hex64(a);
                puts_(" rs2=");         put_hex64(b);
                puts_(" acc=");         put_i64(acc0);
                putc_('\n');
            }
            ++bad;
            ++g_fail;
        } else {
            ++g_pass;
        }
    }
    if (bad) {
        puts_("  共 ");
        put_i64(bad);
        puts_(" 組不符\n");
    }
}

/* ------------------------------------------------------------------ main */

void cmain(void) {
    puts_("=== t_xqmac8 ===\n");

    directed();
    randomized(10000);

    puts_("\npass=");
    put_i64(g_pass);
    puts_("  fail=");
    put_i64(g_fail);
    putc_('\n');
    puts_(g_fail ? "FAILED\n" : "ALL OK\n");

    int code = g_fail > 100 ? 100 : g_fail;
    tohost = ((uint64_t)(code & 0x7f) << 1) | 1;
    for (;;) { }
}

__attribute__((naked, section(".text.init")))
void _start(void) {
    __asm__ volatile("la sp, _stack_top\n\tj cmain");
}
