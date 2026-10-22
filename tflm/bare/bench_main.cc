// bench_main.cc — kernel 層的 microbenchmark（規格書 §5 Evaluation Coverage）。
//
// ======================================================================
// 為什麼需要這支程式
// ======================================================================
// 整個專案的效能數字到目前都是**完整模型**的量測。那是好的（§5 要求
// 「Complete model inference」），但它有兩個盲點：
//
//   1. 每個 kernel 的加速比被 Amdahl 稀釋。B1D 的 depthwise kernel 本身
//      快了 4.57x，但端到端只有 1.089x —— 看端到端數字無法回答
//      「這個 kernel 寫得好不好」。
//
//   2. 形狀只有模型裡實際出現的那些。
//
// microbenchmark 把單一 kernel 用固定尺寸隔離出來量，所以數字直接對應
// kernel 本身的品質，與模型其他部分無關。
//
// ======================================================================
// 設計：一個 binary 裡同時量參考實作與我們的實作
// ======================================================================
// TFLM 的參考 kernel 是 header-only 的 template，所以**任何配置**都拿得到。
// 於是同一個 binary 可以在相同輸入上跑兩份實作、相同的計數器量兩次，
// 兩者的差異只可能來自 kernel 本身。
//
// 配置之間的差異由編譯期旗標決定（跟 nn_bare 完全一致）：
//   B1  Conv1x1Int8 / DwConvInt8 走純量
//   X1  同一份 kernel，內積換成 xqmac8
//   X2  再加上 xqrq
//
// 參考實作那一欄在所有配置中應該**幾乎相同**（只受編譯器 inline 決策
// 影響），這本身就是個健全性檢查。
//
// ======================================================================
// 量測方法
// ======================================================================
// plat_counter() 在 bare-metal 上是 rdinstret —— 已退休的指令數，精確值，
// 不是取樣。計數器自己的開銷是個位數條指令，相對 10^5~10^7 可忽略。
//
// 每個 case 都會比對兩份輸出是否位元相同。**跑得快但算錯的 kernel
// 沒有意義**，所以正確性與效能在同一次執行裡一起檢查。
//
//   make -C bare CONFIG=X2 bench_run

#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "plat.h"

// 只取純數值的 kernel，不碰 TFLM 的 runtime。
// 兩個檔案在同一個 TU 裡 —— 它們的全域變數刻意用不同名字
// （g_sum_w / g_dw_sum_w、g_rq_conv / g_rq_dw），不會撞。
#define CONV_XQ_KERNEL_ONLY 1
#define DWCONV_XQ_KERNEL_ONLY 1
#include "conv_xq.cc"
#include "dwconv_xq.cc"

#ifdef USE_XQRQ
#include "xqrq.h"
#endif

using tflite::RuntimeShape;

namespace {

// 緩衝區上限。最大的 case 是 28x28x96 的 depthwise（75,264 bytes）。
#define BENCH_MAX_IN 80000
#define BENCH_MAX_FLT 16384
#define BENCH_MAX_OUT 80000
#define BENCH_MAX_CH 1536

alignas(16) int8_t g_bin[BENCH_MAX_IN];
alignas(16) int8_t g_bflt[BENCH_MAX_FLT];
alignas(16) int8_t g_bref[BENCH_MAX_OUT];
alignas(16) int8_t g_bgot[BENCH_MAX_OUT];
int32_t g_bbias[BENCH_MAX_CH];
int32_t g_bmult[BENCH_MAX_CH];
int32_t g_bshft[BENCH_MAX_CH];

// xorshift64*：bare-metal 沒有 <random>，而且要可重現。
uint64_t g_s = 0x243F6A8885A308D3ull;
uint64_t rnd() {
  g_s ^= g_s >> 12;
  g_s ^= g_s << 25;
  g_s ^= g_s >> 27;
  return g_s * 0x2545F4914F6CDD1Dull;
}
int32_t rnd_range(int32_t lo, int32_t hi) {
  return lo + (int32_t)(rnd() % (uint64_t)(hi - lo + 1));
}

long g_fails = 0;

// 把兩份輸出逐位元比對。
void compare(const char* what, long n) {
  long bad = 0;
  long first = -1;
  for (long i = 0; i < n; ++i) {
    if (g_bref[i] != g_bgot[i]) {
      if (first < 0) first = i;
      ++bad;
    }
  }
  if (bad) {
    ++g_fails;
    MicroPrintf("  !! %s 輸出不一致 %ld/%ld（第一筆 i=%ld ref=%d got=%d）",
                what, bad, n, first, (int)g_bref[first], (int)g_bgot[first]);
  }
}

// 一行結果。單位是「條指令 / 工作單位」，用整數千分比避免浮點
// （bare-metal 的浮點要走軟浮點，會污染我們正在量的指令數）。
void row(const char* name, uint64_t insn, long work, const char* unit) {
  const uint64_t per_milli = work ? (insn * 1000ULL) / (uint64_t)work : 0;
  MicroPrintf("  %-26s %13llu  %6llu.%03llu %s", name,
              (unsigned long long)insn, (unsigned long long)(per_milli / 1000),
              (unsigned long long)(per_milli % 1000), unit);
}

// ----------------------------------------------------------------------
// 1x1 convolution
// ----------------------------------------------------------------------
void bench_conv1x1(const char* tag, int IH, int IW, int K, int OC) {
  const int32_t d_in[4] = {1, IH, IW, K};
  const int32_t d_flt[4] = {OC, 1, 1, K};
  const int32_t d_out[4] = {1, IH, IW, OC};
  const int32_t d_bias[1] = {OC};
  RuntimeShape in_shape(4, d_in), flt_shape(4, d_flt);
  RuntimeShape out_shape(4, d_out), bias_shape(1, d_bias);

  for (long i = 0; i < (long)IH * IW * K; ++i)
    g_bin[i] = (int8_t)rnd_range(-128, 127);
  for (long i = 0; i < (long)OC * K; ++i)
    g_bflt[i] = (int8_t)rnd_range(-127, 127);
  for (int c = 0; c < OC; ++c) {
    g_bbias[c] = rnd_range(-200000, 200000);
    // 真實模型的 multiplier 正規化到 [0.5, 1)，所以集中在 2^30 附近
    g_bmult[c] = (int32_t)((1u << 30) + (uint32_t)(rnd() % (1u << 30)));
    g_bshft[c] = rnd_range(-14, 0);
  }

  tflite::ConvParams p;
  for (unsigned i = 0; i < sizeof(p); ++i) ((char*)&p)[i] = 0;
  p.stride_width = 1;
  p.stride_height = 1;
  p.dilation_width_factor = 1;
  p.dilation_height_factor = 1;
  p.input_offset = rnd_range(-128, 127);
  p.output_offset = rnd_range(-128, 127);
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;

  const long macs = (long)IH * IW * OC * K;
  const long outs = (long)IH * IW * OC;

  uint64_t t0 = plat_counter();
  tflite::micro::reference_integer_ops::ConvPerChannel(
      p, g_bmult, g_bshft, in_shape, g_bin, flt_shape, g_bflt, bias_shape,
      g_bbias, out_shape, g_bref);
  uint64_t t1 = plat_counter();
  Conv1x1Int8(p, g_bmult, g_bshft, in_shape, g_bin, flt_shape, g_bflt, g_bbias,
              out_shape, g_bgot);
  uint64_t t2 = plat_counter();

  MicroPrintf("");
  MicroPrintf("--- 1x1 conv %s：%dx%dx%d -> %dx%dx%d，%ld MAC，%ld 輸出元素",
              tag, IH, IW, K, IH, IW, OC, macs, outs);
  row("TFLM 參考實作", t1 - t0, macs, "條/MAC");
  row("本專案 kernel", t2 - t1, macs, "條/MAC");
  if (t2 - t1) {
    MicroPrintf("  加速 %llu.%02llux", (unsigned long long)((t1 - t0) / (t2 - t1)),
                (unsigned long long)(((t1 - t0) * 100 / (t2 - t1)) % 100));
  }
  compare("1x1 conv", outs);
}

// ----------------------------------------------------------------------
// depthwise convolution
// ----------------------------------------------------------------------
void bench_dwconv(const char* tag, int IH, int IW, int C, int FH, int FW) {
  // same padding、stride 1
  const int ph = (FH - 1) / 2;
  const int pw = (FW - 1) / 2;
  const int32_t d_in[4] = {1, IH, IW, C};
  const int32_t d_flt[4] = {1, FH, FW, C};
  const int32_t d_out[4] = {1, IH, IW, C};
  const int32_t d_bias[1] = {C};
  RuntimeShape in_shape(4, d_in), flt_shape(4, d_flt);
  RuntimeShape out_shape(4, d_out), bias_shape(1, d_bias);

  for (long i = 0; i < (long)IH * IW * C; ++i)
    g_bin[i] = (int8_t)rnd_range(-128, 127);
  for (long i = 0; i < (long)FH * FW * C; ++i)
    g_bflt[i] = (int8_t)rnd_range(-127, 127);
  for (int c = 0; c < C; ++c) {
    g_bbias[c] = rnd_range(-200000, 200000);
    g_bmult[c] = (int32_t)((1u << 30) + (uint32_t)(rnd() % (1u << 30)));
    g_bshft[c] = rnd_range(-14, 0);
  }

  tflite::DepthwiseParams p;
  for (unsigned i = 0; i < sizeof(p); ++i) ((char*)&p)[i] = 0;
  p.stride_width = 1;
  p.stride_height = 1;
  p.dilation_width_factor = 1;
  p.dilation_height_factor = 1;
  p.depth_multiplier = 1;
  p.padding_values.width = (int16_t)pw;
  p.padding_values.height = (int16_t)ph;
  p.input_offset = rnd_range(-128, 127);
  p.output_offset = rnd_range(-128, 127);
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;

  if (!Eligible(p, in_shape, flt_shape, out_shape)) {
    MicroPrintf("  !! depthwise %s 不符合特化條件，略過", tag);
    ++g_fails;
    return;
  }

  const long macs = (long)IH * IW * C * FH * FW;
  const long outs = (long)IH * IW * C;

  uint64_t t0 = plat_counter();
  tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(
      p, g_bmult, g_bshft, in_shape, g_bin, flt_shape, g_bflt, bias_shape,
      g_bbias, out_shape, g_bref);
  uint64_t t1 = plat_counter();
  DwConvInt8(p, g_bmult, g_bshft, in_shape, g_bin, flt_shape, g_bflt, g_bbias,
             out_shape, g_bgot);
  uint64_t t2 = plat_counter();

  MicroPrintf("");
  MicroPrintf("--- depthwise %s：%dx%dx%d，%dx%d，%ld MAC，%ld 輸出元素", tag,
              IH, IW, C, FH, FW, macs, outs);
  row("TFLM 參考實作", t1 - t0, macs, "條/MAC");
  row("本專案 kernel", t2 - t1, macs, "條/MAC");
  if (t2 - t1) {
    MicroPrintf("  加速 %llu.%02llux", (unsigned long long)((t1 - t0) / (t2 - t1)),
                (unsigned long long)(((t1 - t0) * 100 / (t2 - t1)) % 100));
  }
  compare("depthwise", outs);
}

// ----------------------------------------------------------------------
// requantize —— 規格書 §3 要求的 Instructions / output element
// ----------------------------------------------------------------------
// 這一段是 X2 的直接量測。前面兩個 kernel 的數字是「條/MAC」，
// 但 requantize 的成本與 MAC 無關、只與**輸出元素數**成正比
// —— 這正是它的占比會隨 MAC 被優化而上升的原因（1.26% -> 15.5%）。
//
// 隔離量測比從完整模型相減可靠：後者要假設「省下的全來自 requantize」，
// 前者直接量同一組輸入下兩條路徑各花幾條指令。
volatile int32_t g_sink = 0;

void bench_requant(long n) {
  const int32_t off = 7;
  const int32_t lo = -128;
  const int32_t hi = 127;

  // 用一小組 (acc, mult, shift) 循環，模擬 per-channel 參數的存取模式。
  // 參數全放在陣列裡而不是常數，否則編譯器會把整條鏈折成常數。
  const int NP = 64;
  static int32_t acc[NP], mult[NP], shft[NP];
  for (int i = 0; i < NP; ++i) {
    acc[i] = rnd_range(-2000000, 2000000);
    mult[i] = (int32_t)((1u << 30) + (uint32_t)(rnd() % (1u << 30)));
    shft[i] = rnd_range(-14, 0);
  }

  // ---------------------------------------------------------- 軟體路徑
  int32_t sum_sw = 0;
  uint64_t t0 = plat_counter();
  for (long i = 0; i < n; ++i) {
    const int k = (int)(i & (NP - 1));
    int32_t v = tflite::MultiplyByQuantizedMultiplier(acc[k], mult[k], shft[k]);
    v += off;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    sum_sw += v;
  }
  uint64_t t1 = plat_counter();
  g_sink = sum_sw;

  MicroPrintf("");
  MicroPrintf("--- requantize：%ld 個輸出元素", n);
  row("軟體（TFLM + offset + clamp）", t1 - t0, n, "條/輸出元素");

#ifdef USE_XQRQ
  // ---------------------------------------------------------- xqrq
  // 描述子在迴圈外打包好，跟 kernel 裡的做法一致
  //（per-layer 一次，攤提到所有輸出位置）。
  static uint64_t desc[NP];
  for (int i = 0; i < NP; ++i) {
    desc[i] = xqrq_pack(mult[i], shft[i], off, lo, hi);
  }

  int32_t sum_hw = 0;
  uint64_t t2 = plat_counter();
  for (long i = 0; i < n; ++i) {
    const int k = (int)(i & (NP - 1));
    sum_hw += xqrq_apply(acc[k], desc[k]);
  }
  uint64_t t3 = plat_counter();
  g_sink = sum_hw;

  row("xqrq", t3 - t2, n, "條/輸出元素");
  if (t3 - t2) {
    MicroPrintf("  加速 %llu.%02llux",
                (unsigned long long)((t1 - t0) / (t3 - t2)),
                (unsigned long long)(((t1 - t0) * 100 / (t3 - t2)) % 100));
  }
  if (sum_sw != sum_hw) {
    ++g_fails;
    MicroPrintf("  !! 兩條路徑的累加和不同 sw=%d hw=%d", (int)sum_sw,
                (int)sum_hw);
  }
#else
  MicroPrintf("  （xqrq 未編入此配置；用 CONFIG=X2 量）");
#endif
}

}  // namespace

extern "C" int nn_main(void) {
  MicroPrintf("=== kernel microbenchmark（單位 %s）===", plat_counter_unit());
  MicroPrintf("配置旗標：");
#ifdef USE_XQMAC8
  MicroPrintf("  USE_XQMAC8");
#endif
#ifdef USE_XQRQ
  MicroPrintf("  USE_XQRQ");
#endif
#if !defined(USE_XQMAC8) && !defined(USE_XQRQ)
  MicroPrintf("  （無，純量路徑 = B1）");
#endif
  MicroPrintf("");
  MicroPrintf("「TFLM 參考實作」那一列在各配置中應該幾乎相同 ——");
  MicroPrintf("它不受我們的旗標影響，差異只來自編譯器的 inline 決策。");

  // 形狀取自 EfficientNet-Lite0 實際出現的層，縮小到 Spike 上跑得快的規模。
  bench_conv1x1("後段(K大)", 7, 7, 192, 64);
  bench_conv1x1("中段", 14, 14, 48, 48);
  bench_dwconv("14x14x240 3x3", 14, 14, 240, 3, 3);
  bench_dwconv("28x28x96 3x3", 28, 28, 96, 3, 3);
  bench_requant(1000000);

  MicroPrintf("");
  if (g_fails == 0) {
    MicroPrintf("所有 case 的輸出與參考實作位元一致");
    return 0;
  }
  MicroPrintf("!! %ld 個 case 有問題 —— 效能數字在修好之前沒有意義",
              g_fails);
  return 1;
}
