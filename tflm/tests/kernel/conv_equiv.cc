// conv_equiv.cc — 1x1 conv kernel 的等價性測試。
//
// 把 app/conv_xq.cc 的 Conv1x1Int8 跟 TFLM 的
// reference_integer_ops::ConvPerChannel 直接對打，要求輸出**位元完全相同**。
//
// 為什麼這個比 depthwise 那份更重要：
//   1x1 conv 占全模型 88.18% 的 MAC、占 B1 全程 63.6% 的指令。
//   depthwise 很早就有 kernel 層測試，這一份卻一直缺 —— 那是個缺口。
//
// 這份特別針對 X1 新增的三段邏輯：
//   (a) dot_xqmac8 的對齊判斷與退路
//   (b) K 不是 8 的倍數時的尾巴迴圈
//   (c) 權重未對齊時複製到 g_wbuf，且複製與 Σw 融在同一趟
//
// (c) 是實測逼出來的：輸入張量永遠 8 對齊，但權重住在 flatbuffer 裡，
// 起始位址必然落在 0 或 4 (mod 8)，實測 55% 的 MAC 因此走不了 packed。
// 所以**未對齊的 filter 是必測情境，不是邊角案例**。
//
// 編譯時定義：
//   CONV_XQ_KERNEL_ONLY  只取純數值的 kernel，不連 TFLM runtime
//   USE_XQMAC8           走 X1 的路徑
//   XQ_SIM_INSN          在 x86 上用純量模擬 xqmac8（語意與 Spike 版一致）
//
// 用法: bash tests/kernel/run_conv.sh

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CONV_XQ_KERNEL_ONLY 1
#include "conv_xq.cc"

using tflite::RuntimeShape;

namespace {

long g_cases = 0, g_fails = 0;
long g_misaligned_cases = 0, g_tail_cases = 0;

struct Case {
  int K, OC, IH, IW, OH, OW, sh, sw;
  int off_in, off_flt;      // 故意製造的位址偏移
  const char* why;
};

int Run(const Case& k, unsigned seed, bool verbose) {
  std::mt19937 rng(seed);
  auto U = [&](int lo, int hi) {
    return (int)std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  const int32_t d_in[4] = {1, k.IH, k.IW, k.K};
  const int32_t d_flt[4] = {k.OC, 1, 1, k.K};
  const int32_t d_out[4] = {1, k.OH, k.OW, k.OC};
  const int32_t d_bias[1] = {k.OC};
  RuntimeShape in_shape(4, d_in), flt_shape(4, d_flt);
  RuntimeShape out_shape(4, d_out), bias_shape(1, d_bias);

  // 多配一點，讓指標可以往後位移去製造未對齊
  std::vector<int8_t> in_buf((size_t)k.IH * k.IW * k.K + 16);
  std::vector<int8_t> flt_buf((size_t)k.OC * k.K + 16);
  // vector 的資料起始未必對齊，先對到 8 再加上我們要的偏移
  auto align8 = [](int8_t* p) {
    uintptr_t u = reinterpret_cast<uintptr_t>(p);
    return reinterpret_cast<int8_t*>((u + 7u) & ~(uintptr_t)7u);
  };
  int8_t* in = align8(in_buf.data()) + k.off_in;
  int8_t* flt = align8(flt_buf.data()) + k.off_flt;

  for (long i = 0; i < (long)k.IH * k.IW * k.K; ++i) in[i] = (int8_t)U(-128, 127);
  for (long i = 0; i < (long)k.OC * k.K; ++i) flt[i] = (int8_t)U(-127, 127);

  std::vector<int32_t> bias(k.OC), mult(k.OC), shft(k.OC);
  for (int c = 0; c < k.OC; ++c) {
    bias[c] = U(-200000, 200000);
    mult[c] = (int32_t)U(1 << 30, (int)((1u << 31) - 1));
    shft[c] = U(-14, 1);
  }

  tflite::ConvParams p;
  memset(&p, 0, sizeof(p));
  p.stride_width = (int16_t)k.sw;
  p.stride_height = (int16_t)k.sh;
  p.dilation_width_factor = 1;
  p.dilation_height_factor = 1;
  p.padding_values.width = 0;
  p.padding_values.height = 0;
  p.input_offset = U(-128, 127);
  p.output_offset = U(-128, 127);
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;

  const long n_out = (long)k.OH * k.OW * k.OC;
  std::vector<int8_t> ref(n_out, 0), got(n_out, 0);

  tflite::micro::reference_integer_ops::ConvPerChannel(
      p, mult.data(), shft.data(), in_shape, in, flt_shape, flt,
      bias_shape, bias.data(), out_shape, ref.data());

#ifdef USE_XQMAC8
  g_xq_sim_calls = 0;
#endif
  Conv1x1Int8(p, mult.data(), shft.data(), in_shape, in, flt_shape, flt,
              bias.data(), out_shape, got.data());
#ifdef USE_XQMAC8
  const long calls = g_xq_sim_calls;
#else
  const long calls = 0;
#endif

  int bad = 0;
  long first = -1;
  for (long i = 0; i < n_out; ++i) {
    if (ref[i] != got[i]) { if (first < 0) first = i; ++bad; }
  }

#ifdef USE_XQMAC8
  // xqmac8 應該被發出幾次 —— 「結果正確」不代表「指令生效」。
  // 例如權重複製做了、但 wsrc 忘了指向對齊緩衝，結果會完全正確、
  // 只是沒加速；只有數次數抓得到。
  //
  // 規格（獨立於實作重算一次）：每個 (位置, oc) 的內積，兩個指標都
  // 8 對齊時發出 K/8 次，否則 0 次。
  //
  // 注意 K 不是 8 的倍數時，a = in + 位移*K 的對齊**隨空間位置變動**，
  // 不能用「基底對齊與否」一概而論 —— 這一點第一版的預期值就寫錯了。
  const bool flt_ok = (reinterpret_cast<uintptr_t>(flt) & 7u) == 0;
  const bool copied =
      !flt_ok && ((long)k.OC * k.K <= (long)XQ_WBUF_BYTES);
  const int8_t* wbase = copied ? g_wbuf : flt;
  long want_calls = 0;
  for (int oy = 0; oy < k.OH; ++oy) {
    for (int ox = 0; ox < k.OW; ++ox) {
      const int8_t* a = in + ((long)(oy * k.sh) * k.IW + (ox * k.sw)) * k.K;
      for (int oc = 0; oc < k.OC; ++oc) {
        const int8_t* w = wbase + (long)oc * k.K;
        if (((reinterpret_cast<uintptr_t>(a) |
              reinterpret_cast<uintptr_t>(w)) & 7u) == 0)
          want_calls += k.K >> 3;
      }
    }
  }
  const bool call_bad = (calls != want_calls);
#else
  const long want_calls = 0;
  const bool call_bad = false;
  (void)calls; (void)want_calls;
#endif

  ++g_cases;
  const bool mis = ((reinterpret_cast<uintptr_t>(flt) |
                     reinterpret_cast<uintptr_t>(in)) & 7u) != 0;
  if (mis) ++g_misaligned_cases;
  if (k.K % 8) ++g_tail_cases;

  if (bad || call_bad || verbose || k.why) {
    printf("  K=%-5d OC=%-5d %dx%d->%dx%d s=%d  in%%8=%zu flt%%8=%zu  %s",
           k.K, k.OC, k.IH, k.IW, k.OH, k.OW, k.sh,
           (size_t)(reinterpret_cast<uintptr_t>(in) & 7u),
           (size_t)(reinterpret_cast<uintptr_t>(flt) & 7u),
           bad ? "** 不同 **" : (call_bad ? "** 指令沒發出 **" : "ok"));
    if (k.why) printf("  [%s]", k.why);
    if (bad)
      printf("\n      %d/%ld 筆，第一筆 idx=%ld ref=%d got=%d",
             bad, n_out, first, (int)ref[first], (int)got[first]);
    if (call_bad)
      printf("\n      xqmac8 發出 %ld 次，應為 %ld 次", calls, want_calls);
    printf("\n");
  }
  if (bad || call_bad) ++g_fails;
  return (bad || call_bad) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 600;

  // ---------------------------------------------------------- 針對性案例
  printf("針對性案例\n");
  {
    const Case d[] = {
      // 模型裡真實出現的形狀，兩種對齊都測
      {32,   16,  112,112,112,112,1,1, 0,0, "模型 #1：對齊"},
      {16,   96,  112,112,112,112,1,1, 0,4, "模型 #2：flt 偏 4（實測就是這樣）"},
      {1152, 192, 7,7,    7,7,    1,1, 0,0, "模型 #25：最大 K，對齊"},
      {320,  1280,7,7,    7,7,    1,1, 0,4, "模型 #32：最後一層，flt 偏 4"},
      {192,  1152,7,7,    7,7,    1,1, 0,4, "模型 #24：flt 偏 4"},
      // 對齊的各種組合
      {64, 8, 4,4, 4,4, 1,1, 0,0, "兩邊都對齊"},
      {64, 8, 4,4, 4,4, 1,1, 4,0, "in 偏 4"},
      {64, 8, 4,4, 4,4, 1,1, 0,4, "flt 偏 4"},
      {64, 8, 4,4, 4,4, 1,1, 4,4, "兩邊都偏 4（錯位相同）"},
      {64, 8, 4,4, 4,4, 1,1, 1,0, "in 偏 1（奇數）"},
      {64, 8, 4,4, 4,4, 1,1, 0,3, "flt 偏 3（奇數）"},
      {64, 8, 4,4, 4,4, 1,1, 5,2, "兩邊都奇數且不同"},
      // K 的尾巴
      {1,  8, 4,4, 4,4, 1,1, 0,0, "K=1，全尾巴"},
      {7,  8, 4,4, 4,4, 1,1, 0,0, "K=7 < 8"},
      {9,  8, 4,4, 4,4, 1,1, 0,0, "K=9 = 8+1"},
      {20, 8, 4,4, 4,4, 1,1, 0,0, "K=20 = 2*8+4"},
      {20, 8, 4,4, 4,4, 1,1, 0,4, "K=20 且 flt 未對齊"},
      {63, 8, 4,4, 4,4, 1,1, 0,0, "K=63"},
      // stride
      {64, 8, 8,8, 4,4, 2,2, 0,0, "stride 2"},
      {64, 8, 9,9, 4,4, 2,2, 0,4, "stride 2 + 未對齊"},
      // 單一輸出通道 / 單一位置
      {64, 1, 4,4, 4,4, 1,1, 0,4, "OC=1"},
      {64, 8, 1,1, 1,1, 1,1, 0,4, "1x1 輸出"},
      // 超過 g_wbuf 的情形（應該退回未對齊路徑，仍然正確）
      {1024,512, 2,2, 2,2, 1,1, 0,4, "權重 512KB > 緩衝，不複製"},
    };
    for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); ++i)
      Run(d[i], 3000u + (unsigned)i, false);
  }

  // ---------------------------------------------------------- 隨機案例
  printf("\n隨機案例 x%d（前 4 個印出來）\n", N);
  std::mt19937 rng(20261020);
  auto U = [&](int lo, int hi) {
    return (int)std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  int done = 0;
  for (int t = 0; done < N && t < N * 4; ++t) {
    static const int kK[] = {1,2,3,7,8,9,15,16,17,20,24,31,32,40,48,64,96,112,144,192};
    Case k{};
    k.K = kK[U(0, 19)];
    k.OC = (int[]){1,2,3,8,16,24,32,64}[U(0, 7)];
    k.sh = k.sw = U(1, 2);
    k.IH = U(1, 9); k.IW = U(1, 9);
    k.OH = (k.IH - 1) / k.sh + 1;
    k.OW = (k.IW - 1) / k.sw + 1;
    if (k.OH <= 0 || k.OW <= 0) continue;
    k.off_in = U(0, 7);
    k.off_flt = U(0, 7);
    Run(k, 90000u + (unsigned)t, done < 4);
    ++done;
  }

  // ---------------------------------------------------------- 覆蓋率與結果
  printf("\n覆蓋：%ld 個案例，其中未對齊 %ld、K 非 8 倍數 %ld\n",
         g_cases, g_misaligned_cases, g_tail_cases);
  if (!g_misaligned_cases || !g_tail_cases) {
    printf("!! 有一類情境完全沒測到，這個測試證明不了什麼\n");
    ++g_fails;
  }
  printf("%s：%ld 個案例不符\n", g_fails ? "失敗" : "全部位元相同", g_fails);
  return g_fails ? 1 : 0;
}
