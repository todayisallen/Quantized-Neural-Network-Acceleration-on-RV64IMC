#include "dwconv_xq.h"

#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/depthwise_conv.h"

#ifdef USE_XQRQ
#include "xqrq.h"
#endif

// DWCONV_XQ_KERNEL_ONLY：只編純數值的 kernel，不碰 TFLM 的 runtime
// （Eval / Register / 張量存取）。
//
// 用途是 tests/kernel/dw_equiv.cc —— 把這個檔案 #include 進去，就能拿
// 匿名 namespace 裡的 DwConvInt8 跟參考實作直接對打，不需要跑整個
// interpreter，也不需要連整個 libtflm.a（那會連帶拉進 MicroContext、
// flatbuffers 等一大串跟數值無關的東西）。
#ifndef DWCONV_XQ_KERNEL_ONLY
#include "tensorflow/lite/micro/kernels/conv.h"            // ConvInit / OpDataConv
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"  // Prepare / ParamsQuantized
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#endif

#ifndef XQ_MAX_OUT_CHANNELS
#define XQ_MAX_OUT_CHANNELS 1536
#endif

// ======================================================================
// 為什麼 depthwise 的手法跟 1x1 不一樣
// ======================================================================
//
// 1x1 conv 的瓶頸是「每個 MAC 都重算索引」，拿掉索引算術就好。
// depthwise 的瓶頸不同 —— 它的迴圈次序本身就是錯的。
//
// TFLM 參考實作的次序是：
//     for oy: for ox: for channel: for fy: for fx:
//
// 這個次序下，最內層的 9 個 tap 是**同一個 channel**的 3x3 視窗。
// 在 NHWC 佈局裡，相鄰 tap 的位址差是 C（往 x）或 W*C（往 y），
// 也就是跳著走；而且邊界檢查寫在 channel 迴圈裡面，
// 同樣的 9 次檢查對每個 channel 各做一遍 —— 總共 9*C 次。
//
// 把 channel 換到最內層：
//     for oy: for ox: for fy: for fx: for channel:
//
// 兩件事同時變好：
//   (1) 邊界檢查變成每個輸出位置 9 次，不是 9*C 次（C 最大 1152，
//       差 3 個數量級）。檢查結果跟 channel 無關，本來就不該在裡面。
//   (2) 最內層對 input 和 filter 都是**連續**存取
//       （input[(iy*W+ix)*C + c] 與 filter[(fy*FW+fx)*C + c]，
//       兩邊 c 的 stride 都是 1）。這對指令數沒直接幫助，
//       但是之後要上 packed load / SIMD 的前提。
//
// ----------------------------------------------------------------------
// 為什麼要分塊（channel blocking）
// ----------------------------------------------------------------------
// 單純把 channel 換到最內層會踩到一個坑：累加器變成 acc[c]，
// 被內層迴圈變數索引，編譯器不可能把它放進暫存器。
//
// 第一版就是這樣寫的，實測 17 條指令/MAC。看生成的碼就知道為什麼：
//     lb   a[c]
//     lb   w[c]
//     mulw
//     auipc+addi        <- 取全域陣列的位址，medany 要兩條，每個 MAC 都付
//     lw   acc[c]
//     addw
//     sw   acc[c]       <- 累加器每個 tap 都往返記憶體一次
//     addi / bne
// 真正做事的只有 lb/lb/mulw/addw 四條，其餘都是搬運。
//
// 解法是一次只處理 BLK 個 channel，讓 BLK 個累加器以**區域變數**的形式
// 跨越全部 TAPS 個 tap 都留在暫存器裡：
//
//     for c0 in steps of BLK:
//         int32_t acc[BLK];              // 迴圈界限是編譯期常數 -> 進暫存器
//         for t in 0..TAPS:
//             for j in 0..BLK: acc[j] += a[c0+j] * w[c0+j];
//         requantize BLK 個並寫出
//
// 這樣累加器完全不碰記憶體，全域位址也只在每個 block 取一次。
// 每個 MAC 剩下 lb + lb + mulw + addw，加上攤提後的 requantize。
//
// BLK = 8 的理由：RV64 有 32 個暫存器，8 個累加器加上兩個指標、
// 迴圈變數還很寬裕；同時 8 也對齊未來 packed 指令的寬度。
// 注意 depthwise 要的是「8 個乘積進 8 個累加器」（element-wise），
// 跟 xqmac8 的「8 個乘積橫向加總進 1 個累加器」是**不同的指令** ——
// 這一點對 X2 的設計很重要。
//
// 每個 channel 內部的累加次序仍然是 t = 0..TAPS-1，跟參考實作相同，
// 所以位元結果不變。
//
// ----------------------------------------------------------------------
// 快路徑 / 慢路徑的切法
// ----------------------------------------------------------------------
// depthwise 有 padding，這是跟 1x1 最大的差別 —— 邊界檢查不能直接刪掉。
//
// 而且 zero-point 外提在邊界會失效：參考實作只對「界內」的 tap 累加
//     acc += w * (a + z)
// 所以 zero-point 的貢獻是 z * Σw(界內 tap)，會隨 padding 樣式變化，
// 不是一個固定的 per-channel 常數。
//
// 所以這裡拆成兩條路：
//   * 內部位置（9 個 tap 全在界內）走快路徑：無檢查、用預先算好的 Σw。
//   * 邊界位置走慢路徑：**照抄參考實作**，一行一行對應。
//
// 刻意讓慢路徑是參考實作的逐行轉寫，而不是「有檢查的快路徑」：
// 唯一有可能算錯的地方就是 padding 處理，那段用已知正確的程式碼，
// 新寫的程式碼只負責最單純的情形。
//
// 112x112 的層，3x3 pad=1 時邊界只占 444/12544 = 3.5%，
// 慢路徑的成本可以忽略。
// ======================================================================

// 一次處理幾個 channel。可以在編譯期改，用來量「分塊寬度 vs 指令數」。
//   BLK=1  等價於每個 channel 獨立做（累加器是純量，但位址算術攤提不掉）
//   BLK=8  預設
#ifndef DW_XQ_BLK
#define DW_XQ_BLK 8
#endif

namespace {

// ----------------------------------------------------------------------
// 編譯期展開 BLK 寬的迴圈
// ----------------------------------------------------------------------
// 這裡不是效能微調，是決定 acc[BLK] 會不會留在暫存器 ——
// 沒展開就整個分塊白做，所以不能交給編譯器自己判斷。
//
// 不用 #pragma GCC unroll：那是 GCC 8 才有的，舊版會印
// -Wunknown-pragmas 然後**靜默忽略**（這台機器的 host g++ 就是這樣），
// 而且 pragma 的參數還不做巨集展開。失敗模式是「編得過、跑得對、
// 就是沒變快」，最難察覺的那一種。
//
// 樣板遞迴展開之後這根本不是迴圈，沒有東西讓編譯器決定。
// 索引是型別層級的常數（Idx<J>::value），所以 GCC 的 SRA
// （scalar replacement of aggregates）能把 acc[] 拆成 BLK 個獨立暫存器。
//
// ** 要求 **：所有碰 acc[] 的地方都必須用常數索引，包含 requantize。
// 只要有一處用執行期的 j，整個陣列就必須可定址，SRA 失效，
// MAC 迴圈又會變成對堆疊 lw/sw —— 即使它本身展開了也一樣。
template <int J>
struct Idx {
  static constexpr int value = J;
};

template <int I, int N>
struct UnrollFrom {
  template <typename F>
  static inline void Run(F& f) {
    f(Idx<I>{});
    UnrollFrom<I + 1, N>::Run(f);
  }
};
template <int N>
struct UnrollFrom<N, N> {
  template <typename F>
  static inline void Run(F&) {}
};

#define DW_XQ_UNROLL(f) UnrollFrom<0, DW_XQ_BLK>::Run(f)

int32_t g_dw_sum_w[XQ_MAX_OUT_CHANNELS];

#ifdef USE_XQRQ
// X2：每層打包一次的再量化描述子。名字跟 conv_xq.cc 的 g_rq_conv 分開 ——
// 兩個 kernel 在單元測試裡會被 include 進同一個 TU。
alignas(8) uint64_t g_rq_dw[XQ_MAX_OUT_CHANNELS];
#endif

void DwConvInt8(const tflite::DepthwiseParams& params,
                const int32_t* out_multiplier, const int32_t* out_shift,
                const tflite::RuntimeShape& in_shape, const int8_t* in,
                const tflite::RuntimeShape& flt_shape, const int8_t* flt,
                const int32_t* bias, const tflite::RuntimeShape& out_shape,
                int8_t* out) {
  const int IH = in_shape.Dims(1);
  const int IW = in_shape.Dims(2);
  const int C = in_shape.Dims(3);  // depth_multiplier == 1，所以 IC == OC
  const int FH = flt_shape.Dims(1);
  const int FW = flt_shape.Dims(2);
  const int OH = out_shape.Dims(1);
  const int OW = out_shape.Dims(2);
  const int sh = params.stride_height;
  const int sw = params.stride_width;
  const int ph = params.padding_values.height;
  const int pw = params.padding_values.width;

  const int32_t in_off = params.input_offset;  // = -input_zero_point
  const int32_t out_off = params.output_offset;
  const int32_t amin = params.quantized_activation_min;
  const int32_t amax = params.quantized_activation_max;

  const int TAPS = FH * FW;

  // Σw：每個 channel 對整個視窗加總，只算一次。
  //
  // 溢位界限（Eligible() 保證 TAPS <= 256）：
  //   |Σ w*a|   <= 256 * 127 * 128 = 4,161,536
  //   |z * Σw|  <= 128 * 256 * 127 = 4,161,536
  // 兩者相加遠小於 2^31，所以外提不會改變結果。
  for (int c = 0; c < C; ++c) g_dw_sum_w[c] = 0;
  for (int t = 0; t < TAPS; ++t) {
    const int8_t* w = flt + static_cast<long>(t) * C;
    for (int c = 0; c < C; ++c) g_dw_sum_w[c] += w[c];
  }

#ifdef USE_XQRQ
  // X2：再量化描述子跟 Σw 同一個 per-layer 時機打包，見 app/xqrq.h。
  for (int c = 0; c < C; ++c) {
    g_rq_dw[c] = xqrq_pack(out_multiplier[c], out_shift[c], out_off, amin, amax);
  }
#endif

  for (int oy = 0; oy < OH; ++oy) {
    const int iy0 = oy * sh - ph;
    const bool row_inside = (iy0 >= 0) && (iy0 + FH <= IH);

    for (int ox = 0; ox < OW; ++ox) {
      const int ix0 = ox * sw - pw;
      int8_t* o = out + (static_cast<long>(oy) * OW + ox) * C;

      if (row_inside && ix0 >= 0 && ix0 + FW <= IW) {
        // -------------------------------------------------- 快路徑
        // tap (fy,fx) 的輸入基底 = in_base + (fy*IW + fx)*C，
        // 因為 in_base 已經把 (iy0, ix0) 吃進去了。
        const int8_t* const in_base =
            in + (static_cast<long>(iy0) * IW + ix0) * C;
        const int32_t* const sw = g_dw_sum_w;  // 取一次全域位址，不要每輪重算

        int c0 = 0;
        for (; c0 + DW_XQ_BLK <= C; c0 += DW_XQ_BLK) {
          // 三個 DW_XQ_UNROLL 把這個陣列的每次存取都變成常數索引，
          // 所以它不會真的存在 —— 會被拆成 BLK 個暫存器。
          int32_t acc[DW_XQ_BLK];

          // tap 0 直接寫入，省掉歸零。
          {
            const int8_t* a = in_base + c0;
            const int8_t* w = flt + c0;
            auto f = [&](auto J) {
              constexpr int j = decltype(J)::value;
              acc[j] = static_cast<int32_t>(a[j]) * static_cast<int32_t>(w[j]);
            };
            DW_XQ_UNROLL(f);
          }
          // 其餘 tap。fy/fx 用遞增維護，不做除法。
          // 次序是 (0,0) (0,1) ... (FH-1,FW-1)，跟參考實作完全相同。
          int fy = 0, fx = 0;
          for (int t = 1; t < TAPS; ++t) {
            if (++fx == FW) { fx = 0; ++fy; }
            const int8_t* a =
                in_base + (static_cast<long>(fy) * IW + fx) * C + c0;
            const int8_t* w = flt + static_cast<long>(t) * C + c0;
            auto f = [&](auto J) {
              constexpr int j = decltype(J)::value;
              acc[j] += static_cast<int32_t>(a[j]) * static_cast<int32_t>(w[j]);
            };
            DW_XQ_UNROLL(f);
          }
          // requantize 也必須展開，不是為了它自己快，而是因為用執行期的 j
          // 索引 acc[] 會讓整個陣列變成可定址，上面兩個迴圈的 SRA 就失效了。
          // 代價是 BLK 份的 code size，換上面每個 MAC 省掉一次 lw+sw。
          {
            auto f = [&](auto J) {
              constexpr int j = decltype(J)::value;
              const int c = c0 + j;
              int32_t v = acc[j] + in_off * sw[c];
              if (bias) v += bias[c];
#ifdef USE_XQRQ
              o[c] = static_cast<int8_t>(xqrq_apply(v, g_rq_dw[c]));
#else
              v = tflite::MultiplyByQuantizedMultiplier(v, out_multiplier[c],
                                                        out_shift[c]);
              v += out_off;
              if (v < amin) v = amin;
              if (v > amax) v = amax;
              o[c] = static_cast<int8_t>(v);
#endif
            };
            DW_XQ_UNROLL(f);
          }
        }
        // 尾巴：C 不是 BLK 的倍數時。EfficientNet-Lite0 的 depthwise
        // channel 數（32/96/144/240/480/672/1152）都是 8 的倍數，
        // 所以這段實際上不會執行 —— 但不能因此省略。
        for (; c0 < C; ++c0) {
          int32_t v = static_cast<int32_t>(in_base[c0]) *
                      static_cast<int32_t>(flt[c0]);
          int fy = 0, fx = 0;
          for (int t = 1; t < TAPS; ++t) {
            if (++fx == FW) { fx = 0; ++fy; }
            v += static_cast<int32_t>(
                     in_base[(static_cast<long>(fy) * IW + fx) * C + c0]) *
                 static_cast<int32_t>(flt[static_cast<long>(t) * C + c0]);
          }
          v += in_off * sw[c0];
          if (bias) v += bias[c0];
#ifdef USE_XQRQ
          o[c0] = static_cast<int8_t>(xqrq_apply(v, g_rq_dw[c0]));
#else
          v = tflite::MultiplyByQuantizedMultiplier(v, out_multiplier[c0],
                                                    out_shift[c0]);
          v += out_off;
          if (v < amin) v = amin;
          if (v > amax) v = amax;
          o[c0] = static_cast<int8_t>(v);
#endif
        }
      } else {
        // -------------------------------------------------- 慢路徑（邊界）
        // 以下是 reference_integer_ops::DepthwiseConvPerChannel 的逐行轉寫，
        // 只把 Offset() 展開成明確的索引。刻意不做任何優化。
        for (int c = 0; c < C; ++c) {
          int32_t acc = 0;
          for (int fy = 0; fy < FH; ++fy) {
            for (int fx = 0; fx < FW; ++fx) {
              const int iy = iy0 + fy;
              const int ix = ix0 + fx;
              if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
                const int32_t a = in[(static_cast<long>(iy) * IW + ix) * C + c];
                const int32_t w =
                    flt[static_cast<long>(fy * FW + fx) * C + c];
                acc += w * (a + in_off);
              }
            }
          }
          if (bias) acc += bias[c];
#ifdef USE_XQRQ
          // 邊界慢路徑也要改。漏掉不會算錯，只是這部分沒加速 —— 而
          // depthwise 的邊界在後段層占到 49%~82%（報告 §9.3），
          // 漏掉的話數字會莫名其妙地差，而且不會有任何錯誤訊息。
          o[c] = static_cast<int8_t>(xqrq_apply(acc, g_rq_dw[c]));
#else
          acc = tflite::MultiplyByQuantizedMultiplier(acc, out_multiplier[c],
                                                      out_shift[c]);
          acc += out_off;
          if (acc < amin) acc = amin;
          if (acc > amax) acc = amax;
          o[c] = static_cast<int8_t>(acc);
#endif
        }
      }
    }
  }
}

// 任何一項不符就退回 TFLM 的參考實作 —— 寧可慢，不要算錯。
bool Eligible(const tflite::DepthwiseParams& p,
              const tflite::RuntimeShape& in_shape,
              const tflite::RuntimeShape& flt_shape,
              const tflite::RuntimeShape& out_shape) {
  if (in_shape.Dims(0) != 1) return false;          // batch 只支援 1
  if (flt_shape.Dims(0) != 1) return false;         // filter 的第 0 維按定義是 1
  if (p.depth_multiplier != 1) return false;        // 只做 1:1（EfficientNet 如此）
  if (p.dilation_width_factor != 1) return false;
  if (p.dilation_height_factor != 1) return false;
  const int C = in_shape.Dims(3);
  if (flt_shape.Dims(3) != C) return false;
  if (out_shape.Dims(3) != C) return false;
  if (C > XQ_MAX_OUT_CHANNELS) return false;        // 暫存區容量
  if (flt_shape.Dims(1) * flt_shape.Dims(2) > 256) return false;  // 見溢位界限
  return true;
}

#ifndef DWCONV_XQ_KERNEL_ONLY

TfLiteStatus XqDwConvEval(TfLiteContext* context, TfLiteNode* node) {
  // 取張量的部分照抄 TFLM 的 DepthwiseConvEval（去掉 float / int16 / int4
  // 分支與 compression）。唯一該有的差異是內層迴圈，不是 boilerplate。
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(
      context, node, tflite::kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(
      context, node, tflite::kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node,
                                        tflite::kDepthwiseConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, tflite::kDepthwiseConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const tflite::OpDataConv*>(node->user_data));

  const tflite::RuntimeShape in_shape = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape flt_shape = tflite::micro::GetTensorShape(filter);
  const tflite::RuntimeShape out_shape = tflite::micro::GetTensorShape(output);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8) {
    MicroPrintf("dwconv_xq: 只支援 int8，收到 input=%d filter=%d",
                (int)input->type, (int)filter->type);
    return kTfLiteError;
  }

  const tflite::DepthwiseParams dw_params =
      tflite::DepthwiseConvParamsQuantized(params, data);

  if (Eligible(dw_params, in_shape, flt_shape, out_shape)) {
    DwConvInt8(dw_params, data.per_channel_output_multiplier,
               data.per_channel_output_shift, in_shape,
               tflite::micro::GetTensorData<int8_t>(input), flt_shape,
               tflite::micro::GetTensorData<int8_t>(filter),
               bias ? tflite::micro::GetOptionalTensorData<int32_t>(bias)
                    : nullptr,
               out_shape, tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }

  tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(
      dw_params, data.per_channel_output_multiplier,
      data.per_channel_output_shift, in_shape,
      tflite::micro::GetTensorData<int8_t>(input), flt_shape,
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<int32_t>(bias), out_shape,
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

#endif  // DWCONV_XQ_KERNEL_ONLY

}  // namespace

#ifndef DWCONV_XQ_KERNEL_ONLY
// Init 用 tflite::ConvInit：TFLM 的 DepthwiseConvInit 在匿名 namespace 裡
// 連不到，但兩者的 body 完全相同
// （都是 AllocatePersistentBuffer(sizeof(OpDataConv))），
// 而 DepthwiseConvPrepare 也正是把 user_data 當 OpDataConv 用。
TFLMRegistration Register_DEPTHWISE_CONV_2D_XQ() {
  return tflite::micro::RegisterOp(tflite::ConvInit,
                                   tflite::DepthwiseConvPrepare, XqDwConvEval);
}
#endif  // DWCONV_XQ_KERNEL_ONLY