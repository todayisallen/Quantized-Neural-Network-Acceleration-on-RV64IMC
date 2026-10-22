// t_xqrq.cc — `xqrq` 指令的 bare-metal 驗證（規格書 §5 要求 directed + randomized）。
//
// ======================================================================
// 為什麼 golden 用 TFLM 本體，而不是自己寫一份
// ======================================================================
// MultiplyByQuantizedMultiplier 有兩套實作 —— 單次捨入與 gemmlowp 的
// 雙次捨入 —— 在「剛好落在 .5」的輸入上差 1 個 LSB。
//
// 如果自己轉錄一份當 golden，那就只驗證了「Spike 跟我的轉錄一致」，
// 而我的轉錄可能抄錯邊。所以這裡直接 include TFLM 的標頭、呼叫真正的
// tflite::MultiplyByQuantizedMultiplier。
//
// 這帶來一個有用的副作用：**這支測試本身就是捨入模式的探針**。
//   全過   -> Spike 的實作與 TFLM 一致，端到端 BITEXACT 是結構上的結果
//   全錯   -> 捨入模式選錯了，把 riscv/insns/xqrq.h 的
//             XQRQ_SINGLE_ROUNDING 翻一下再跑
//   零星錯 -> 不是捨入模式，是欄位解碼或飽和邊界（看印出的案例名）
//
// ======================================================================
// 編譯與執行
// ======================================================================
//   make -C bare t_xqrq            # 編
//   make -C bare t_xqrq_run        # 編 + 在 Spike 上跑
//
// 期望輸出最後一行：pass=N fail=0

#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "plat.h"
#include "xqrq.h"

namespace {

long g_pass = 0;
long g_fail = 0;

// golden：TFLM 的再量化，加上我們 kernel 在它之後做的 offset 與 clamp。
// 這五個步驟就是規格書 §3 要求 xqrq 一條指令做完的那條鏈。
int32_t golden(int32_t acc, int32_t mult, int32_t shift, int32_t offset,
               int32_t lo, int32_t hi) {
  int32_t v = tflite::MultiplyByQuantizedMultiplier(acc, mult, shift);
  v += offset;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return v;
}

void check(const char* name, int32_t acc, int32_t mult, int32_t shift,
           int32_t offset, int32_t lo, int32_t hi) {
  const uint64_t d = xqrq_pack(mult, shift, offset, lo, hi);

  // 先驗打包/解包的來回。這一步不碰指令，所以打包寫錯會在這裡被抓到，
  // 而不是混在指令的錯誤裡 —— 五個欄位擠一個字，最常見的 bug 是
  // 移位量寫錯一個而讀到鄰居的位元。
  if (xqrq_mult(d) != mult || xqrq_shift(d) != shift ||
      xqrq_offset(d) != offset || xqrq_lo(d) != lo || xqrq_hi(d) != hi) {
    ++g_fail;
    MicroPrintf("FAIL %-9s 打包/解包不一致", name);
    MicroPrintf("     desc=%08x%08x", (unsigned)(d >> 32), (unsigned)d);
    MicroPrintf("     mult %d->%d  shift %d->%d  off %d->%d  lo %d->%d  hi %d->%d",
                (int)mult, (int)xqrq_mult(d), (int)shift, (int)xqrq_shift(d),
                (int)offset, (int)xqrq_offset(d), (int)lo, (int)xqrq_lo(d),
                (int)hi, (int)xqrq_hi(d));
    return;
  }

  const int32_t want = golden(acc, mult, shift, offset, lo, hi);
  const int32_t got = xqrq_apply(acc, d);

  if (want != got) {
    ++g_fail;
    MicroPrintf("FAIL %-9s acc=%d mult=%d shift=%d off=%d lo=%d hi=%d",
                name, (int)acc, (int)mult, (int)shift, (int)offset, (int)lo,
                (int)hi);
    MicroPrintf("     want=%d got=%d  (差 %d)", (int)want, (int)got,
                (int)(got - want));
  } else {
    ++g_pass;
  }
}

// ----------------------------------------------------------------------
// Directed 案例
// ----------------------------------------------------------------------
// 每一個都針對一種具體的實作錯誤，不是湊數量。
void directed() {
  const int32_t M30 = 1 << 30;      // 0.5 的定點表示（shift=0 時 v = acc/2）
  const int32_t MMAX = 2147483647;  // INT32_MAX
  const int32_t AMIN = -2147483647 - 1;

  // 基本路徑
  check("ZERO", 0, M30, 0, 0, -128, 127);
  check("OFFONLY", 0, M30, 0, 37, -128, 127);
  check("ONE", 2, M30, 0, 0, -128, 127);

  // 捨入方向。
  //
  // 注意這四個**分辨不出**單次與雙次捨入 —— shift=0 時兩者在數學上
  // 完全相同（單次的 total_shift=31 / round=2^30 就是雙次的 SRDHM）。
  // 初版文件誤把「HALF* 全錯」當成捨入模式的判據，實測否證了那個判據：
  // 27 個 directed 全過，模式卻是錯的，是隨機測試抓到的。
  // 真正能分辨的是下面的 DBLROUND* 與 TRUNCDIV。
  check("HALF", 3, M30, 0, 0, -128, 127);
  check("HALFNEG", -3, M30, 0, 0, -128, 127);
  check("HALF2", 1, M30, 0, 0, -128, 127);
  check("HALF2NEG", -1, M30, 0, 0, -128, 127);

  // ------------------------------------------------------------------
  // 能分辨單次 / 雙次捨入的案例
  // ------------------------------------------------------------------
  // 這四筆是從隨機測試實際抓到的失敗案例收進來的，所以它們**確定**
  // 會在模式選錯時失敗。兩種模式的分岔點有兩個：
  //
  //   (a) shift > 0：雙次捨入的 x = acc << left 是 int32 乘法、溢位環繞；
  //       單次捨入走 64 位元再夾到 int32。大累加值會放大這個差異。
  //   (b) shift < 0：雙次捨入先 SRDHM（在 2^31 處捨入）再
  //       RoundingDivideByPOT（第二次捨入）；單次捨入只捨入一次。
  check("DBLROUND1", 2147483647, 1410351355, 20, -38, 69, 119);
  check("DBLROUND2", AMIN, M30, -31, 60, -23, 117);
  check("DBLROUND3", AMIN, 2128109924, 27, 86, -59, 102);
  check("DBLROUND4", 1592655855, MMAX, 4, -124, -101, 80);

  // ------------------------------------------------------------------
  // offset 相加的位寬
  // ------------------------------------------------------------------
  // 軟體參考是 int32 相加、溢位環繞：
  //     acc = MultiplyByQuantizedMultiplier(...);  // int32
  //     acc += out_off;                             // 溢位環繞
  // Spike 若用 int64 相加，在 |rq| 接近 INT32_MAX 時會分岔。
  //
  // 這三筆也是從隨機測試抓到的（當時 9 筆失敗全部 mult = INT32_MAX），
  // 收進 directed 當回歸測試。真實模型碰不到這些輸入 —— 它們要求
  // 再量化結果接近 INT32_MAX，而 multiplier/shift 的設計目的正是
  // 讓結果落在 int8 附近。但語意有缺口就是有缺口，要守住。
  check("OFFWRAP1", AMIN, MMAX, 0, -71, -65, 6);
  check("OFFWRAP2", MMAX, MMAX, 0, 17, 48, 108);
  check("OFFWRAP3", 2, MMAX, 30, -37, -99, -89);

  // gemmlowp 的 SaturatingRoundingDoublingHighMul 用的是
  // 「/ (1ll << 31)」—— 整數除法，對負數往零截斷。
  // 寫成「>> 31」（往下取整）在這一筆上會差 1。
  //   x=-1, mult=1 -> ab=-1, +nudge = -2^30
  //   除法 -> 0（|值| < 2^31，往零截斷）
  //   右移 -> -1
  check("TRUNCDIV", -1, 1, 0, 0, -128, 127);
  check("TRUNCDIV2", -3, 1, 0, 0, -128, 127);
  check("TRUNCDIV3", -1, 3, 0, 0, -128, 127);

  // shift 的三條路徑與兩個邊界
  check("SHIFTPOS", 7, M30, 5, 0, -128, 127);
  check("SHIFTNEG", 100000, M30, -14, 0, -128, 127);
  check("SHIFTMIN", 1000000, M30, -31, 0, -128, 127);  // total_shift=62
  check("SHIFTMAX", 1, M30, 30, 0, -128, 127);         // total_shift=1, round=1

  // multiplier 與累加值的邊界
  check("MULTMAX", 1000, MMAX, -20, 0, -128, 127);
  check("MULT1", 12345, 1, 0, 0, -128, 127);
  check("ACCMAX", MMAX, M30, -24, 0, -128, 127);
  check("ACCMIN", AMIN, M30, -24, 0, -128, 127);  // 負數右移的符號

  // 中間結果超出 int32 —— 測 §2.1 的 int32 夾取有沒有生效
  check("INT32SAT", MMAX, MMAX, 0, 0, -128, 127);
  check("INT32SATN", AMIN, MMAX, 0, 0, -128, 127);

  // 飽和的兩端
  check("CLAMPHI", 1000, M30, 0, 0, -128, 127);
  check("CLAMPLO", -1000, M30, 0, 0, -128, 127);

  // offset 的 int8 邊界
  check("OFFMAX", 0, M30, 0, 127, -128, 127);
  check("OFFMIN", 0, M30, 0, -128, -128, 127);

  // 真實的 ReLU6 參數：下界不是 -128，而是 output zero point
  check("RELU6", 50000, 1234567890, -18, -128, -128, 127);
  check("RELU6B", -50000, 1234567890, -18, -128, -128, 127);

  // 非對稱的 clamp 區間
  check("NARROW", 1000, M30, 0, 0, -10, 10);
  check("LOEQHI", 1000, M30, 0, 0, 5, 5);

  // 最重要的一個：五個欄位都放互不相同、不對稱的值，
  // 一次把解碼的移位量掃過。任一個寫錯就會讀到鄰居的位元。
  check("DESCPACK", 777, 0x12345678, -7, 0x1A, -0x40, 0x5B);
  check("DESCPACK2", -777, 0x7EDCBA98, 11, -0x33, -0x7F, 0x7E);
}

// ----------------------------------------------------------------------
// Randomized
// ----------------------------------------------------------------------
// xorshift64*：bare-metal 沒有 <random>，而且要可重現。
uint64_t g_s = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
  g_s ^= g_s >> 12;
  g_s ^= g_s << 25;
  g_s ^= g_s >> 27;
  return g_s * 0x2545F4914F6CDD1Dull;
}

// 往邊界偏的取值 —— 均勻取樣幾乎碰不到飽和與捨入邊界，
// 而錯誤正好都集中在那裡。
int32_t rnd_acc() {
  switch (rnd() % 8) {
    case 0: return 0;
    case 1: return (int32_t)(rnd() % 5) - 2;                  // 0 附近
    case 2: return 2147483647;
    case 3: return -2147483647 - 1;
    case 4: return (int32_t)(rnd() % 2001) - 1000;            // 小值
    default: return (int32_t)(uint32_t)rnd();                 // 全域
  }
}

int32_t rnd_mult() {
  switch (rnd() % 4) {
    case 0: return 1 << 30;
    case 1: return 2147483647;
    // 真實模型的 multiplier 集中在 2^30 附近（正規化到 [0.5, 1)）
    default: return (int32_t)((1u << 30) + (uint32_t)(rnd() % (1u << 30)));
  }
}

void randomized(long n) {
  for (long i = 0; i < n; ++i) {
    const int32_t acc = rnd_acc();
    const int32_t mult = rnd_mult();
    const int32_t shift = (int32_t)(rnd() % 62) - 31;  // [-31, 30]
    const int32_t offset = (int32_t)(rnd() % 256) - 128;
    int32_t lo = (int32_t)(rnd() % 256) - 128;
    int32_t hi = (int32_t)(rnd() % 256) - 128;
    if (lo > hi) { const int32_t t = lo; lo = hi; hi = t; }
    check("rand", acc, mult, shift, offset, lo, hi);
    // 失敗太多就停，否則 console 會被洗掉、看不到最早的那幾筆
    if (g_fail > 20) {
      MicroPrintf("!! 失敗超過 20 筆，提早停止");
      return;
    }
  }
}

}  // namespace

extern "C" int nn_main(void) {
  MicroPrintf("=== t_xqrq: xqrq 指令驗證 ===");
  MicroPrintf("golden = tflite::MultiplyByQuantizedMultiplier + offset + clamp");
#if TFLITE_SINGLE_ROUNDING
  MicroPrintf("TFLM 編譯為【單次捨入】(TFLITE_SINGLE_ROUNDING=1)");
  MicroPrintf("  -> riscv/insns/xqrq.h 要用 docs/X2_xqrq_design.md §2.1");
#else
  MicroPrintf("TFLM 編譯為【雙次捨入】(gemmlowp)");
  MicroPrintf("  -> riscv/insns/xqrq.h 要用 docs/X2_xqrq_design.md §2.2");
#endif
  MicroPrintf("");

  directed();
  const long d_pass = g_pass, d_fail = g_fail;
  MicroPrintf("directed:   pass=%ld fail=%ld", d_pass, d_fail);

  randomized(10000);
  MicroPrintf("randomized: pass=%ld fail=%ld", g_pass - d_pass,
              g_fail - d_fail);

  MicroPrintf("");
  MicroPrintf("pass=%ld fail=%ld", g_pass, g_fail);
  if (g_fail == 0) {
    MicroPrintf("ALL OK");
    return 0;
  }
  MicroPrintf("有失敗。判讀：");
  MicroPrintf("  DBLROUND* / 隨機案例錯（大 acc 配正 shift）-> 捨入模式選錯");
  MicroPrintf("  只有 TRUNCDIV* 錯 -> SRDHM 用了 >>31，應為 / (1<<31)");
  MicroPrintf("  只有 OFFWRAP* 錯  -> offset 相加用了 int64，應為 int32 環繞");
  MicroPrintf("  只有 DESCPACK 錯  -> 欄位解碼的移位量寫錯");
  MicroPrintf("  只有 INT32SAT* 錯 -> int32 夾取漏了");
  return 1;
}