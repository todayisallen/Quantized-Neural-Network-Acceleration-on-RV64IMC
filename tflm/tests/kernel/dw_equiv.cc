// dw_equiv.cc — depthwise conv kernel 的等價性測試。
//
// 把 app/dwconv_xq.cc 的 DwConvInt8 跟 TFLM 的
// reference_integer_ops::DepthwiseConvPerChannel 直接對打，
// 要求輸出**位元完全相同**（不是「接近」）。
//
// 為什麼要有這個測試，而不是只靠 end-to-end 的 BITEXACT:
//   end-to-end 只走模型裡實際出現的那幾種形狀。而 DwConvInt8 的
//   風險全部集中在 padding 的處理上 —— 快路徑（全界內）跟慢路徑
//   （邊界）的切分如果算錯一格，在某些形狀下會剛好不顯現。
//   這裡用隨機形狀 + 針對性的退化形狀把切分邏輯掃過。
//
// 編譯時定義 DWCONV_XQ_KERNEL_ONLY，只取純數值的 kernel，
// 不連 TFLM runtime（那會拉進 MicroContext / flatbuffers 等
// 跟數值無關的一大串依賴）。
//
// 用法: bash tests/kernel/run.sh

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define DWCONV_XQ_KERNEL_ONLY 1
#include "dwconv_xq.cc"

using tflite::RuntimeShape;

namespace {

struct Case {
  int C, FH, FW, sh, sw, IH, IW, OH, OW, ph, pw;
  const char* why;  // 非空代表是針對性案例，會一律印出來
};

long g_fast = 0, g_slow = 0;  // 累計走過的輸出像素數，用來證明兩條路都有被走到

// 回傳不符的筆數
int Compare(const Case& k, unsigned seed, bool verbose) {
  std::mt19937 rng(seed);
  auto U = [&](int lo, int hi) {
    return (int)(std::uniform_int_distribution<int>(lo, hi)(rng));
  };

  const int32_t d_in[4] = {1, k.IH, k.IW, k.C};
  const int32_t d_flt[4] = {1, k.FH, k.FW, k.C};
  const int32_t d_out[4] = {1, k.OH, k.OW, k.C};
  const int32_t d_bias[1] = {k.C};
  RuntimeShape in_shape(4, d_in);
  RuntimeShape flt_shape(4, d_flt);
  RuntimeShape out_shape(4, d_out);
  RuntimeShape bias_shape(1, d_bias);

  std::vector<int8_t> in((size_t)k.IH * k.IW * k.C);
  std::vector<int8_t> flt((size_t)k.FH * k.FW * k.C);
  std::vector<int32_t> bias(k.C), mult(k.C), shft(k.C);
  // 取滿值域：int8 的兩端、以及 requantize 會分歧的 multiplier/shift 組合
  for (auto& v : in) v = (int8_t)U(-128, 127);
  for (auto& v : flt) v = (int8_t)U(-127, 127);  // 對稱量化不會產生 -128
  for (int c = 0; c < k.C; ++c) {
    bias[c] = U(-200000, 200000);
    mult[c] = (int32_t)U(1 << 30, (int)((1u << 31) - 1));
    shft[c] = U(-14, 1);
  }

  tflite::DepthwiseParams p;
  memset(&p, 0, sizeof(p));
  p.stride_width = (int16_t)k.sw;
  p.stride_height = (int16_t)k.sh;
  p.dilation_width_factor = 1;
  p.dilation_height_factor = 1;
  p.depth_multiplier = 1;
  p.padding_values.width = (int16_t)k.pw;
  p.padding_values.height = (int16_t)k.ph;
  p.input_offset = U(-128, 127);
  p.output_offset = U(-128, 127);
  p.quantized_activation_min = -128;
  p.quantized_activation_max = 127;

  if (!Eligible(p, in_shape, flt_shape, out_shape)) {
    printf("  !! Eligible() 回 false：C=%d %dx%d —— 這個形狀不該被排除\n",
           k.C, k.FH, k.FW);
    return 1;
  }

  // 分析式的路徑覆蓋：算出有幾個輸出像素會走快路徑。
  // 這段必須跟 DwConvInt8 裡的判斷式一致。
  int fast = 0;
  for (int oy = 0; oy < k.OH; ++oy) {
    const int iy0 = oy * k.sh - k.ph;
    if (!(iy0 >= 0 && iy0 + k.FH <= k.IH)) continue;
    for (int ox = 0; ox < k.OW; ++ox) {
      const int ix0 = ox * k.sw - k.pw;
      if (ix0 >= 0 && ix0 + k.FW <= k.IW) ++fast;
    }
  }
  const int total = k.OH * k.OW;
  g_fast += fast;
  g_slow += total - fast;

  std::vector<int8_t> ref((size_t)total * k.C, 0), got((size_t)total * k.C, 0);

  tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(
      p, mult.data(), shft.data(), in_shape, in.data(), flt_shape, flt.data(),
      bias_shape, bias.data(), out_shape, ref.data());

  DwConvInt8(p, mult.data(), shft.data(), in_shape, in.data(), flt_shape,
             flt.data(), bias.data(), out_shape, got.data());

  int bad = 0;
  long first = -1;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) {
      if (first < 0) first = (long)i;
      ++bad;
    }
  }

  if (bad || verbose || k.why) {
    printf("  C=%-5d %dx%d s=%d,%d in=%dx%d pad=%d,%d -> %dx%d  "
           "快/總=%d/%d  %s",
           k.C, k.FH, k.FW, k.sh, k.sw, k.IH, k.IW, k.ph, k.pw, k.OH, k.OW,
           fast, total, bad ? "** 不同 **" : "ok");
    if (k.why) printf("  [%s]", k.why);
    if (bad) {
      printf("\n      %d/%zu 筆不同，第一筆 idx=%ld ref=%d got=%d", bad,
             ref.size(), first, (int)ref[first], (int)got[first]);
    }
    printf("\n");
  }
  return bad ? 1 : 0;
}

// TFLite 的 SAME padding 計算，跟 TFLM 的 ComputePadding 一致
void SamePad(int in, int fs, int stride, int* out, int* pad) {
  *out = (in + stride - 1) / stride;
  int needed = (*out - 1) * stride + fs - in;
  *pad = needed > 0 ? needed / 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 2000;
  int fails = 0;

  // ------------------------------------------------------------ 針對性案例
  // 全部瞄準 padding 的切分邏輯 —— 新程式碼唯一可能算錯的地方。
  printf("針對性案例（邊界切分）\n");
  {
    const struct {
      int C, F, s, IH, IW;
      bool same;
      const char* why;
    } d[] = {
      {32, 3, 1, 112, 112, true, "最大的 depthwise 層"},
      {1152, 3, 1, 7, 7, true, "最大 C，暫存區上限"},
      {96, 3, 2, 112, 112, true, "stride 2 + SAME，pad 不對稱"},
      {144, 5, 2, 56, 56, true, "5x5 stride 2"},
      {8, 3, 1, 3, 3, true, "輸入剛好等於 filter"},
      {8, 3, 1, 2, 2, true, "輸入比 filter 小，快路徑應為空"},
      {8, 5, 1, 3, 3, true, "輸入比 filter 小很多"},
      {8, 3, 1, 1, 1, true, "1x1 輸入，全是 padding"},
      {8, 3, 1, 4, 20, true, "細長：高度窄、寬度長"},
      {8, 3, 1, 20, 4, true, "細長：反向"},
      {8, 1, 1, 8, 8, true, "1x1 depthwise，TAPS=1"},
      {8, 3, 2, 7, 7, false, "VALID + stride 2，無邊界"},
      {8, 3, 1, 8, 8, false, "VALID，應該全走快路徑"},
      // 以下專打 channel 分塊的尾巴迴圈
      {1, 3, 1, 9, 9, true, "C=1，只有尾巴迴圈"},
      {7, 3, 1, 9, 9, true, "C=7 < BLK，只有尾巴迴圈"},
      {9, 3, 1, 9, 9, true, "C=9 = BLK+1，一塊加一個尾巴"},
      {33, 5, 1, 9, 9, true, "C=33 = 4*BLK+1"},
      {100, 3, 2, 14, 14, true, "C=100 = 12*BLK+4"},
    };
    for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); ++i) {
      Case k{};
      k.C = d[i].C; k.FH = k.FW = d[i].F; k.sh = k.sw = d[i].s;
      k.IH = d[i].IH; k.IW = d[i].IW; k.why = d[i].why;
      if (d[i].same) {
        SamePad(k.IH, k.FH, k.sh, &k.OH, &k.ph);
        SamePad(k.IW, k.FW, k.sw, &k.OW, &k.pw);
      } else {
        k.ph = k.pw = 0;
        k.OH = (k.IH - k.FH) / k.sh + 1;
        k.OW = (k.IW - k.FW) / k.sw + 1;
        if (k.OH <= 0 || k.OW <= 0) continue;
      }
      fails += Compare(k, 1000u + (unsigned)i, false);
    }
  }

  // ------------------------------------------------------------ 隨機案例
  printf("\n隨機案例 x%d（前 5 個印出來）\n", N);
  std::mt19937 rng(12345);
  auto U = [&](int lo, int hi) {
    return (int)(std::uniform_int_distribution<int>(lo, hi)(rng));
  };
  int done = 0;
  for (int t = 0; done < N && t < N * 4; ++t) {
    // 刻意混入不是 DW_XQ_BLK 倍數的 channel 數（1/3/5/7/12/20/33/100），
    // 否則分塊的尾巴迴圈永遠走不到 —— EfficientNet 的 channel 數剛好
    // 全是 8 的倍數，只靠真實模型測不出那段。
    static const int kCh[] = {1,  2,  3,  5,  7,   8,   12,  16,
                              20, 24, 32, 33, 40,  48,  96,  100,
                              144, 192, 240, 336};
    Case k{};
    k.C = kCh[U(0, (int)(sizeof(kCh) / sizeof(kCh[0])) - 1)];
    k.FH = k.FW = (int[]){1, 3, 5}[U(0, 2)];
    k.sh = k.sw = U(1, 2);
    k.IH = U(1, 20);
    k.IW = U(1, 20);
    if (U(0, 1)) {
      SamePad(k.IH, k.FH, k.sh, &k.OH, &k.ph);
      SamePad(k.IW, k.FW, k.sw, &k.OW, &k.pw);
    } else {
      k.ph = k.pw = 0;
      k.OH = (k.IH - k.FH) / k.sh + 1;
      k.OW = (k.IW - k.FW) / k.sw + 1;
    }
    if (k.OH <= 0 || k.OW <= 0) continue;  // 退化形狀，TFLite 不會產生
    fails += Compare(k, 7000u + (unsigned)t, done < 5);
    ++done;
  }

  // ------------------------------------------------------------ 結果
  printf("\n路徑覆蓋：快路徑 %ld 個輸出像素，慢路徑（邊界）%ld 個\n", g_fast,
         g_slow);
  if (g_fast == 0 || g_slow == 0) {
    printf("!! 有一條路徑完全沒被走到，這個測試證明不了什麼\n");
    ++fails;
  }
  printf("%s：%d 個案例不符\n", fails ? "失敗" : "全部位元相同", fails);
  return fails ? 1 : 0;
}