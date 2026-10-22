// conv_xq.h — 自訂的 CONV_2D kernel，取代 TFLM 的參考實作。
//
// 用法（在 resolver 裡）：
//     r.AddConv2D(Register_CONV_2D_XQ());
//
// 不需要改 TFLM 的原始碼：MicroMutableOpResolver::AddConv2D() 本來就接受
// 自訂的 TFLMRegistration，而 ConvInit / ConvPrepare 在 conv.h 有匯出。
// 所以我們只換 Eval，multiplier 的計算、padding、activation range
// 全部沿用 TFLM 的，確保 B0 與 B1/X1 之間唯一的變數是內層迴圈。
//
// 編譯期開關：
//   （不定義）     B1  —— 重寫過的迴圈，內積用純 RV64IMC scalar
//   -DUSE_XQMAC8   X1  —— 同一份迴圈，內積換成 xqmac8
//
// 兩個配置共用同一個檔案，所以「軟體最佳化的貢獻」與「指令的貢獻」
// 可以乾淨地分開歸因。

#ifndef CONV_XQ_H_
#define CONV_XQ_H_

#include "tensorflow/lite/micro/micro_common.h"

TFLMRegistration Register_CONV_2D_XQ();

#endif  // CONV_XQ_H_
