#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"

#ifdef USE_XQRQ
#include "xqrq.h"
#endif

#ifndef CONV_XQ_KERNEL_ONLY
#include "conv_xq.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#endif

#ifndef XQ_MAX_OUT_CHANNELS
#define XQ_MAX_OUT_CHANNELS 1536
#endif

#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
long g_xq_blocks = 0;
long g_xq_scalar = 0;
#endif

#ifdef USE_XQMAC8
#ifndef XQ_WBUF_BYTES
#define XQ_WBUF_BYTES 409600
#endif
alignas(16) int8_t g_wbuf[XQ_WBUF_BYTES];
#endif

namespace {

int32_t g_sum_w[XQ_MAX_OUT_CHANNELS];

#ifdef USE_XQRQ
alignas(8) uint64_t g_rq_conv[XQ_MAX_OUT_CHANNELS];
#endif

// ======================================================================
// 內積輔助邏輯 (保持原有陣列索引存取)
// ======================================================================

inline int32_t DoScalarDotProduct(const int8_t* a, const int8_t* w, int K) {
  int32_t sum = 0;
  for (int k = 0; k < K; ++k) {
    sum += static_cast<int32_t>(a[k]) * static_cast<int32_t>(w[k]);
  }
  return sum;
}

#ifdef USE_XQMAC8
#ifdef XQ_SIM_INSN
long g_xq_sim_calls = 0;
#endif

inline void DoXqMac8Insn(int64_t& acc, uint64_t a8, uint64_t w8) {
#ifdef XQ_SIM_INSN
  ++g_xq_sim_calls;
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

inline int32_t DoXqMac8DotProduct(const int8_t* a, const int8_t* w, int K) {
  int64_t acc = 0;
  int k = 0;
  
  const bool pointers_aligned = (((reinterpret_cast<uintptr_t>(a) | 
                                    reinterpret_cast<uintptr_t>(w)) & 7u) == 0);
  if (pointers_aligned) {
    const auto* pa = static_cast<const uint64_t*>(static_cast<const void*>(a));
    const auto* pw = static_cast<const uint64_t*>(static_cast<const void*>(w));
    const int n = K >> 3;

    for (int i = 0; i < n; ++i) {
      DoXqMac8Insn(acc, pa[i], pw[i]);
    }
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
// 1x1 conv 核心實作
// ======================================================================

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

  const int32_t in_off = params.input_offset;
  const int32_t out_off = params.output_offset;
  const int32_t amin = params.quantized_activation_min;
  const int32_t amax = params.quantized_activation_max;

  // 權重對齊邏輯
  const int8_t* wsrc = flt;
#ifdef USE_XQMAC8
  const long wbytes = static_cast<long>(OC) * K;
  const bool need_copy = ((reinterpret_cast<uintptr_t>(flt) & 7u) != 0) &&
                          (wbytes <= static_cast<long>(XQ_WBUF_BYTES));
  wsrc = need_copy ? g_wbuf : flt;
#else
  const bool need_copy = false;
#endif

  // 預先計算 Σw 與複製權重
  if (need_copy) {
#ifdef USE_XQMAC8
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
      for (int k = 0; k < K; ++k) {
        s += w[k];
      }
      g_sum_w[oc] = s;
    }
  }

#ifdef USE_XQRQ
  for (int oc = 0; oc < OC; ++oc) {
    g_rq_conv[oc] = xqrq_pack(out_multiplier[oc], out_shift[oc], out_off, amin, amax);
  }
#endif

  // 空間與通道迴圈
  for (int oy = 0; oy < OH; ++oy) {
    for (int ox = 0; ox < OW; ++ox) {
      const int8_t* a = in + (static_cast<long>(oy * sh) * IW + (ox * sw)) * K;
      int8_t* o = out + (static_cast<long>(oy) * OW + ox) * OC;

      for (int oc = 0; oc < OC; ++oc) {
        const int8_t* w = wsrc + static_cast<long>(oc) * K;

#ifdef USE_XQMAC8
        int32_t acc = DoXqMac8DotProduct(a, w, K);
#else
        int32_t acc = DoScalarDotProduct(a, w, K);
#endif

        acc += in_off * g_sum_w[oc];
        if (bias) {
          acc += bias[oc];
        }

#ifdef USE_XQRQ
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
// XQ_REPORT 與 TFLM Integration
// ======================================================================

#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY)
long g_xq_calls = 0;

void XqReport1x1(const tflite::RuntimeShape& in_shape,
                 const tflite::RuntimeShape& out_shape,
                 const int8_t* in_p, const int8_t* flt_p,
                 long blocks0, long scalar0) {
  const int K = in_shape.Dims(3);
  const int OH = out_shape.Dims(1), OW = out_shape.Dims(2);
  const int OC = out_shape.Dims(3);
  const long macs = static_cast<long>(OH) * OW * OC * K;

  const long blocks = g_xq_blocks - blocks0;
  const long scalar = g_xq_scalar - scalar0;
  const long packed = blocks * 8;
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

bool Eligible(const TfLiteConvParams& params, const tflite::OpDataConv& data,
              const tflite::RuntimeShape& in_shape,
              const tflite::RuntimeShape& flt_shape,
              const tflite::RuntimeShape& out_shape) {
  if (flt_shape.Dims(1) != 1 || flt_shape.Dims(2) != 1) return false;
  if (params.dilation_width_factor != 1) return false;
  if (params.dilation_height_factor != 1) return false;
  if (data.padding.width != 0 || data.padding.height != 0) return false;
  if (in_shape.Dims(0) != 1) return false;
  if (in_shape.Dims(3) != flt_shape.Dims(3)) return false;
  if (out_shape.Dims(3) > XQ_MAX_OUT_CHANNELS) return false;
  return true;
}

TfLiteStatus XqConvEval(TfLiteContext* context, TfLiteNode* node) {
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
    XqReport1x1(in_shape, out_shape, in_p, flt_p, blocks0, scalar0);
#endif
    return kTfLiteOk;
  }

  if (input->type != kTfLiteInt8) {
    MicroPrintf("conv_xq: 只支援 int8，收到 type %d", static_cast<int>(input->type));
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