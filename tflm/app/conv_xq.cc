#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"

#ifdef USE_XQRQ
#include "xqrq.h"
#endif

// CONV_XQ_KERNEL_ONLY：只編純數值的 kernel，不碰 TFLM 的 runtime
// （Eval / Register / 張量存取）。給 tests/kernel/conv_equiv.cc 用 ——
// 把這個檔案 #include 進去就能拿匿名 namespace 裡的 Conv1x1Int8
// 直接跟參考實作對打，不需要跑整個 interpreter。
#ifndef CONV_XQ_KERNEL_ONLY
#include "conv_xq.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#endif

// sum_w 的暫存區上限。EfficientNet-Lite0 最大的 output channel 數是 1280。
// 超過就退回參考實作，不會算錯，只是沒加速。
#ifndef XQ_MAX_OUT_CHANNELS
#define XQ_MAX_OUT_CHANNELS 1536
#endif

#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
// 地面真相的計數器：直接數 xqmac8 實際發出幾次、多少 MAC 走純量。
//
// 第一版的 XQ_REPORT 是**預測**：在 Eval 裡看 flt_p 的對齊、推算
// 會不會走 packed。那是錯的 —— 權重複製發生在 Conv1x1Int8 裡面，
// 診斷在它之前執行，所以永遠報複製前的狀態，修好了也看不出來。
//
// 教訓跟 hotloop.py 那次一樣：有辦法量就不要推。
long g_xq_blocks = 0;    // xqmac8 發出的次數（每次 = 8 個 MAC）
long g_xq_scalar = 0;    // 走純量路徑的 MAC 數
#endif

#ifdef USE_XQMAC8
// 對齊用的權重緩衝。只有 X1 需要（B1 的純量路徑不在乎對齊）。
//
// 大小取這個模型未對齊 filter 的最大值：
//   最後一層 1x1 是 K=320、OC=1280 -> 409,600 bytes。
// 超過這個大小的層不複製，直接走未對齊（仍然正確，只是那一層沒加速）。
//
// 400 KB 的 .bss 對真正的 MCU 來說不小。替代方案是把 oc 迴圈換到外層、
// 每個 oc 複製 K bytes（緩衝只要 1152 bytes），複製總量完全相同；
// 代價是輸出的寫入變成 stride-OC 的散佈寫，指令數一樣但 cache 行為較差。
// 這裡選大緩衝，因為 Spike 不模擬 cache，先不要為了量測指標犧牲真實行為。
#ifndef XQ_WBUF_BYTES
#define XQ_WBUF_BYTES 409600
#endif
alignas(16) int8_t g_wbuf[XQ_WBUF_BYTES];
#endif

namespace {

int32_t g_sum_w[XQ_MAX_OUT_CHANNELS];

#ifdef USE_XQRQ
// X2：每層打包一次的再量化描述子，見 app/xqrq.h。
// 名字跟 dwconv_xq.cc 的 g_rq_dw 刻意分開 —— 兩個 kernel 在單元測試裡
// 會被 include 進同一個 TU，同名的話匿名 namespace 會撞。
alignas(8) uint64_t g_rq_conv[XQ_MAX_OUT_CHANNELS];
#endif

// ======================================================================
// 內積：B1 與 X1 唯一的差別就在這裡
// ======================================================================

// B1：純 RV64IMC。
// 相對 TFLM 參考實作省掉的東西：
//   * 兩次 Offset() 的四維索引展開（各 3 乘 3 加）
//   * 每個 tap 的邊界檢查（1x1 不可能越界）
//   * 每個 MAC 的 (input_val + input_offset)（外提成 z * sum_w）
// 剩下的就是兩個載入、一個乘、一個加。
inline int32_t dot_scalar(const int8_t* a, const int8_t* w, int K) {
  int32_t s = 0;
  for (int k = 0; k < K; ++k) {
    s += static_cast<int32_t>(a[k]) * static_cast<int32_t>(w[k]);
  }
  return s;
}

#ifdef USE_XQMAC8
// X1：一條指令做 8 個 int8 MAC 並累加。
//
//   xqmac8 rd, rs1, rs2
//     rd  += Σ(i=0..7) sext8(rs1[i]) * sext8(rs2[i])
//
// 編碼 custom-0 (opcode 0x0b), funct3=1, funct7=0。
// funct3=0 已經被環境自檢用的 xqtest 占用，所以這裡用 1。
// Spike 端必須有對應的實作，否則會是 illegal instruction。
//
// 硬體不需要處理 zero-point：權重是對稱量化（z_w = 0），
// 輸入的 zero-point 已經外提成 z * sum_w。這讓指令維持單純的
// 「8 路乘加」，不必多帶一個運算元。
#ifdef XQ_SIM_INSN
// 測試用：數 xqmac8 實際被發出幾次。
//
// 為什麼需要：「結果正確」不代表「指令生效」。例如權重複製做了、
// 但 wsrc 忘了指向對齊緩衝，結果會完全正確、只是沒加速 ——
// 這跟 x86 那次 -Igen 順序的失敗是同一類。數次數才抓得到。
long g_xq_sim_calls = 0;
#endif

inline void xqmac8_step(int64_t& acc, uint64_t a8, uint64_t w8) {
#ifdef XQ_SIM_INSN
  ++g_xq_sim_calls;
  // 測試用：在沒有 xqmac8 的平台（x86）上用純量模擬這條指令。
  // 這讓「對齊判斷、K 的尾巴、權重複製」這些純軟體邏輯可以在 host 上
  // 用秒級的迴圈驗證，而不必每次都上 Spike。
  //
  // 語意必須與 bare/t_xqmac8.c 驗過的 Spike 實作完全一致。
  for (int i = 0; i < 8; ++i) {
    const int32_t x = static_cast<int8_t>(static_cast<uint8_t>(a8 >> (8 * i)));
    const int32_t y = static_cast<int8_t>(static_cast<uint8_t>(w8 >> (8 * i)));
    acc += static_cast<int64_t>(x) * y;
  }
#else
  __asm__ volatile(".insn r 0x0b, 1, 0, %0, %1, %2"
                   : "+r"(acc)
                   : "r"(a8), "r"(w8));
#endif
}

// 不假設 K 是 8 的倍數，也不假設指標對齊 —— 兩者都在這裡處理：
//   * 兩個指標都 8 對齊 -> 前 (K/8)*8 個元素用 xqmac8，剩下 0~7 個用純量
//   * 任一個沒對齊       -> 全部走純量（仍然正確，只是沒加速）
//
// 為什麼不像之前那樣由呼叫端整層判斷：
//   K 不是 8 的倍數時，整層退回純量會連「能用的那 (K/8)*8 個」一起放棄。
//   K=20 的話浪費 80% 的可加速空間。切塊加尾巴沒有這個問題。
//
//   這個模型實測 32 個 1x1 層的 K 全部是 8 的倍數（tools/check_packed.py），
//   所以尾巴在這裡一次都不會執行。寫它是為了不依賴那個巧合 ——
//   也因此驗證測試必須自己造非 8 倍數的 K，真實模型測不到那條路徑。
//
// 對齊檢查的成本：兩個指標用一次 OR 併起來檢查，是**每個內積一次**，
// 攤提在 K 個 MAC 上（K 平均 84），不是每個 MAC 一次。
inline int32_t dot_xqmac8(const int8_t* a, const int8_t* w, int K) {
  int64_t acc = 0;
  int k = 0;
  if (((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(w)) & 7u)
      == 0) {
    const uint64_t* pa =
        static_cast<const uint64_t*>(static_cast<const void*>(a));
    const uint64_t* pw =
        static_cast<const uint64_t*>(static_cast<const void*>(w));
    const int n = K >> 3;
    for (int i = 0; i < n; ++i) xqmac8_step(acc, pa[i], pw[i]);
    k = n << 3;
#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
    g_xq_blocks += n;
#endif
  }
#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
  g_xq_scalar += (K - k);
#endif
  for (; k < K; ++k) {
    acc += static_cast<int32_t>(a[k]) * static_cast<int32_t>(w[k]);
  }
  return static_cast<int32_t>(acc);
}
#endif  // USE_XQMAC8

// ======================================================================
// 1x1 conv 的特化路徑
// ======================================================================
//
// 1x1 conv 在數學上就是：對每個空間位置，把長度 K 的輸入向量
// 跟每個 output channel 的長度 K 權重向量做內積。
//
// 記憶體佈局正好配合：
//   input  是 NHWC   -> input[(y*W + x)*K .. +K] 連續
//   filter 是 [oc][fy][fx][ic]，1x1 時 -> filter[oc*K .. +K] 連續
// 兩個運算元都是連續的 int8 陣列，所以 64 位元 packed load 直接可用。
void Conv1x1Int8(const tflite::ConvParams& params,
                 const int32_t* out_multiplier, const int32_t* out_shift,
                 const tflite::RuntimeShape& in_shape, const int8_t* in,
                 const tflite::RuntimeShape& flt_shape, const int8_t* flt,
                 const int32_t* bias, const tflite::RuntimeShape& out_shape,
                 int8_t* out) {
  const int K = in_shape.Dims(3);
  const int IW = in_shape.Dims(2);
  const int OH = out_shape.Dims(1);
  const int OW = out_shape.Dims(2);
  const int OC = out_shape.Dims(3);
  const int sh = params.stride_height;
  const int sw = params.stride_width;

  const int32_t in_off = params.input_offset;    // = -input_zero_point
  const int32_t out_off = params.output_offset;
  const int32_t amin = params.quantized_activation_min;
  const int32_t amax = params.quantized_activation_max;

  // ------------------------------------------------------------------
  // 權重的對齊修正（只有 X1 需要）
  // ------------------------------------------------------------------
  // 實測：輸入張量永遠 8 對齊（TFLM 自己配置 arena），但**權重不是**。
  // 權重住在模型的 flatbuffer 裡，vector 的格式是 [uint32 長度][資料]，
  // 長度欄位 4 對齊，所以資料的起始位址必然落在 0 或 4 (mod 8)。
  // EfficientNet-Lite0 的 32 個 1x1 層實測剛好一半一半，
  // 導致 55% 的 MAC 走不了 packed 路徑。
  //
  // 把整個模型陣列位移 4 bytes 沒用 —— 那只會把兩組對調。
  //
  // 修法是把未對齊的權重複製到對齊的緩衝。這之所以划算，是因為
  // **權重對空間位置是迴圈不變量**：同一組權重會在 OH*OW 個位置被
  // 重複讀取，但只需要複製一次。實測 1,662,976 bytes 的複製，
  // 換掉 186,253,312 個 MAC 的未對齊處理，差 42 倍。
  //
  // 而且複製可以融進下面算 Σw 的那一趟 —— 本來就要走過整個 filter
  // 一次，所以沒有額外的遍歷成本。
  const int8_t* wsrc = flt;
#ifdef USE_XQMAC8
  const long wbytes = static_cast<long>(OC) * K;
  const bool need_copy =
      ((reinterpret_cast<uintptr_t>(flt) & 7u) != 0) &&
      (wbytes <= static_cast<long>(XQ_WBUF_BYTES));
  if (need_copy) wsrc = g_wbuf;
#else
  const bool need_copy = false;
#endif

  // Σw 每個 output channel 只算一次。
  //
  // TFLM 原版每個 MAC 都做 (a + z)，等價於把 z*Σw 重算了 H*W*K 次。
  // 外提之後成本是 OC*K 次加法，相對整層的 H*W*OC*K 次乘加是 1/(H*W)。
  // 對 112x112 的層是 0.008%，對 7x7 的層約 2%。
  //
  // 這個 Σw 只跟權重有關，理想上應該在 Prepare 算一次就好。
  // 留在這裡是為了不動 TFLM 的 OpDataConv 結構；單次推論沒有差別。
  if (need_copy) {
#ifdef USE_XQMAC8
    // 複製與加總同一趟
    for (int oc = 0; oc < OC; ++oc) {
      const int8_t* src = flt + static_cast<long>(oc) * K;
      int8_t* dst = g_wbuf + static_cast<long>(oc) * K;
      int32_t s = 0;
      for (int k = 0; k < K; ++k) {
        const int8_t v = src[k];
        dst[k] = v;
        s += v;
      }
      g_sum_w[oc] = s;
    }
#endif
  } else {
    for (int oc = 0; oc < OC; ++oc) {
      const int8_t* w = flt + static_cast<long>(oc) * K;
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += w[k];
      g_sum_w[oc] = s;
    }
  }

#ifdef USE_XQRQ
  // X2：再量化描述子也在這裡打包，跟 Σw 同一個 per-layer 時機。
  // 成本 O(OC)，攤提到 OH*OW 個輸出位置上；後三個欄位是每層常數，
  // 重複存 OC 份，換掉指令裡的三個運算元。
  for (int oc = 0; oc < OC; ++oc) {
    g_rq_conv[oc] =
        xqrq_pack(out_multiplier[oc], out_shift[oc], out_off, amin, amax);
  }
#endif

  for (int oy = 0; oy < OH; ++oy) {
    for (int ox = 0; ox < OW; ++ox) {
      const int8_t* a =
          in + (static_cast<long>(oy * sh) * IW + (ox * sw)) * K;
      int8_t* o = out + (static_cast<long>(oy) * OW + ox) * OC;

      for (int oc = 0; oc < OC; ++oc) {
        const int8_t* w = wsrc + static_cast<long>(oc) * K;

        // B1 與 X1 的唯一差別就是這一行。
        // dot_xqmac8 自己處理對齊與 K 的尾巴，呼叫端不需要先判斷。
#ifdef USE_XQMAC8
        int32_t acc = dot_xqmac8(a, w, K);
#else
        int32_t acc = dot_scalar(a, w, K);
#endif

        acc += in_off * g_sum_w[oc];          // zero-point 補償，外提的結果
        if (bias) acc += bias[oc];

#ifdef USE_XQRQ
        // X2：乘、捨入、移位、加 offset、飽和，一條指令做完。
        o[oc] = static_cast<int8_t>(xqrq_apply(acc, g_rq_conv[oc]));
#else
        acc = tflite::MultiplyByQuantizedMultiplier(acc, out_multiplier[oc],
                                                    out_shift[oc]);
        acc += out_off;
        if (acc < amin) acc = amin;
        if (acc > amax) acc = amax;
        o[oc] = static_cast<int8_t>(acc);
#endif
      }
    }
  }
}

// ======================================================================
// XQ_REPORT：證明 xqmac8 真的被用上了
// ======================================================================
// 用 -DXQ_REPORT 開啟。每個走特化路徑的 1x1 層印一行，
// 並累計「有多少 MAC 真的走 packed 路徑」。
//
// 為什麼需要這個：之前在 x86 上踩過一次 —— -Igen 的順序讓原本的
// resolver_gen.h 勝出，自訂 kernel 根本沒被註冊，但程式跑得完全正常，
// 於是量到一個 1.13x 的假數字，追了很久才用 nm 發現。
//
// 「數字變好了」不能證明「指令生效了」。這個診斷在相信任何 X1 數字
// 之前先把因果釘住：每一層都要是 packed，而且 packed MAC 數要等於
// tools/check_packed.py 由模型結構算出的 338,537,472。
#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
long g_xq_calls = 0;

// 印出這一層實際發生的事 —— 全部來自計數器，沒有任何推算。
//
// 必須在 Conv1x1Int8 **之後**呼叫：權重的對齊複製發生在它裡面，
// 在它之前看 flt_p 只會看到複製前的狀態。
void XqReport1x1(const tflite::RuntimeShape& in_shape,
                 const tflite::RuntimeShape& out_shape,
                 const int8_t* in_p, const int8_t* flt_p,
                 long blocks0, long scalar0) {
  const int K = in_shape.Dims(3);
  const int OH = out_shape.Dims(1), OW = out_shape.Dims(2);
  const int OC = out_shape.Dims(3);
  const long macs = static_cast<long>(OH) * OW * OC * K;

  const long blocks = g_xq_blocks - blocks0;      // 這一層發出的 xqmac8
  const long scalar = g_xq_scalar - scalar0;      // 這一層走純量的 MAC
  const long packed = blocks * 8;
  // 這一層最多能有幾個 MAC 走 packed（K 的尾巴本來就只能走純量）
  const long best = static_cast<long>(OH) * OW * OC * (K & ~7);

  ++g_xq_calls;
  MicroPrintf("[xq] #%-2ld K=%-4d OC=%-4d %dx%d  in%%8=%u flt%%8=%u  "
              "packed=%ld/%ld %s  scalar=%ld | 全程 packed=%ld scalar=%ld",
              g_xq_calls, K, OC, OH, OW,
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(in_p) & 7u),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(flt_p) & 7u),
              packed, best,
              (packed == best) ? "OK" : "** 未達標 **",
              scalar, g_xq_blocks * 8, g_xq_scalar);
  (void)macs;
}
#endif  // XQ_REPORT

#ifndef CONV_XQ_KERNEL_ONLY

// 這一層能不能走特化路徑。任何一項不符就退回 TFLM 的參考實作 ——
// 寧可慢，不要算錯。
bool Eligible(const TfLiteConvParams& params, const tflite::OpDataConv& data,
              const tflite::RuntimeShape& in_shape,
              const tflite::RuntimeShape& flt_shape,
              const tflite::RuntimeShape& out_shape) {
  if (flt_shape.Dims(1) != 1 || flt_shape.Dims(2) != 1) return false;  // 只做 1x1
  if (params.dilation_width_factor != 1) return false;
  if (params.dilation_height_factor != 1) return false;
  if (data.padding.width != 0 || data.padding.height != 0) return false;
  if (in_shape.Dims(0) != 1) return false;                 // batch 只支援 1
  if (in_shape.Dims(3) != flt_shape.Dims(3)) return false; // groups 必須是 1
  if (out_shape.Dims(3) > XQ_MAX_OUT_CHANNELS) return false;
  return true;
}

TfLiteStatus XqConvEval(TfLiteContext* context, TfLiteNode* node) {
  // 以下取張量的部分完全照抄 TFLM 的 ConvEval，刻意不做任何改動：
  // 唯一該有的差異是內層迴圈，不是 boilerplate。
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, tflite::kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const tflite::OpDataConv*>(node->user_data));

  const tflite::RuntimeShape in_shape = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape flt_shape = tflite::micro::GetTensorShape(filter);
  const tflite::RuntimeShape out_shape = tflite::micro::GetTensorShape(output);

  if (input->type == kTfLiteInt8 && filter->type == kTfLiteInt8 &&
      Eligible(params, data, in_shape, flt_shape, out_shape)) {
    const int8_t* in_p = tflite::micro::GetTensorData<int8_t>(input);
    const int8_t* flt_p = tflite::micro::GetTensorData<int8_t>(filter);

#ifdef XQ_REPORT
    const long blocks0 = g_xq_blocks, scalar0 = g_xq_scalar;
#endif

    Conv1x1Int8(tflite::ConvParamsQuantized(params, data),
                data.per_channel_output_multiplier,
                data.per_channel_output_shift, in_shape, in_p, flt_shape,
                flt_p,
                bias ? tflite::micro::GetOptionalTensorData<int32_t>(bias)
                     : nullptr,
                out_shape, tflite::micro::GetTensorData<int8_t>(output));

#ifdef XQ_REPORT
    // 必須在 Conv1x1Int8 之後 —— 權重的對齊複製在它裡面發生。
    XqReport1x1(in_shape, out_shape, in_p, flt_p, blocks0, scalar0);
#endif
    return kTfLiteOk;
  }

  // 其他情況（3x3 的 stem conv、非 int8）走 TFLM 的參考實作。
  // B1 與 X1 都只動 1x1，這樣兩者的比較範圍完全一致。
  if (input->type != kTfLiteInt8) {
    MicroPrintf("conv_xq: 只支援 int8，收到 type %d", (int)input->type);
    return kTfLiteError;
  }
  tflite::micro::reference_integer_ops::ConvPerChannel(
      tflite::ConvParamsQuantized(params, data),
      data.per_channel_output_multiplier, data.per_channel_output_shift,
      in_shape, tflite::micro::GetTensorData<int8_t>(input), flt_shape,
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetOptionalTensorData<int32_t>(bias), out_shape,
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

#endif  // CONV_XQ_KERNEL_ONLY

}  // namespace

#ifndef CONV_XQ_KERNEL_ONLY
TFLMRegistration Register_CONV_2D_XQ() {
  return tflite::micro::RegisterOp(tflite::ConvInit, tflite::ConvPrepare,
                                   XqConvEval);
}
#endif  // CONV_XQ_KERNEL_ONLY