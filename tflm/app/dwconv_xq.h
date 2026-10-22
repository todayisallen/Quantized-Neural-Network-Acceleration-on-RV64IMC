#ifndef APP_DWCONV_XQ_H_
#define APP_DWCONV_XQ_H_

#include "tensorflow/lite/micro/micro_common.h"

// 重寫過的 depthwise conv kernel（int8 per-channel）。
//
// 用 -DTFLM_REG_DEPTHWISE_CONV_2D='Register_DEPTHWISE_CONV_2D_XQ()' 掛進
// resolver_gen.h 產生的註冊表，不需要改 TFLM 的原始碼。
//
// 不符合特化條件的層會自動退回 TFLM 的參考實作，所以掛上去永遠是安全的。
TFLMRegistration Register_DEPTHWISE_CONV_2D_XQ();

#endif  // APP_DWCONV_XQ_H_
