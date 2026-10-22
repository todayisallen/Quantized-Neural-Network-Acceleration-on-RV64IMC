// 由 tools/model_info.py 自動產生，請勿手改。
// 來源模型: model.tflite
// op 總數 64，種類 8
#ifndef RESOLVER_GEN_H_
#define RESOLVER_GEN_H_

#define TFLM_OP_COUNT 8

// 每個 operator 的 registration 都可以在編譯期換成自訂 kernel，例如
//     -DTFLM_REG_CONV_2D='Register_CONV_2D_XQ()'
// 這是 B1 / X1 替換 1x1 conv 實作的方式 —— 不需要改 TFLM 的原始碼。

#ifndef TFLM_REG_QUANTIZE
#define TFLM_REG_QUANTIZE tflite::micro::Register_QUANTIZE()
#endif
#ifndef TFLM_REG_CONV_2D
#define TFLM_REG_CONV_2D tflite::micro::Register_CONV_2D()
#endif
#ifndef TFLM_REG_DEPTHWISE_CONV_2D
#define TFLM_REG_DEPTHWISE_CONV_2D tflite::micro::Register_DEPTHWISE_CONV_2D()
#endif
#ifndef TFLM_REG_ADD
#define TFLM_REG_ADD tflite::micro::Register_ADD()
#endif
#ifndef TFLM_REG_AVERAGE_POOL_2D
#define TFLM_REG_AVERAGE_POOL_2D tflite::micro::Register_AVERAGE_POOL_2D()
#endif
#ifndef TFLM_REG_RESHAPE
#define TFLM_REG_RESHAPE tflite::micro::Register_RESHAPE()
#endif
#ifndef TFLM_REG_FULLY_CONNECTED
#define TFLM_REG_FULLY_CONNECTED tflite::micro::Register_FULLY_CONNECTED()
#endif
#ifndef TFLM_REG_SOFTMAX
#define TFLM_REG_SOFTMAX tflite::micro::Register_SOFTMAX()
#endif

#define TFLM_ADD_OPS(r)  do {  \
    r.AddQuantize(TFLM_REG_QUANTIZE);    /* QUANTIZE x2 */ \
    r.AddConv2D(TFLM_REG_CONV_2D);    /* CONV_2D x33 */ \
    r.AddDepthwiseConv2D(TFLM_REG_DEPTHWISE_CONV_2D);    /* DEPTHWISE_CONV_2D x16 */ \
    r.AddAdd(TFLM_REG_ADD);    /* ADD x9 */ \
    r.AddAveragePool2D(TFLM_REG_AVERAGE_POOL_2D);    /* AVERAGE_POOL_2D x1 */ \
    r.AddReshape(TFLM_REG_RESHAPE);    /* RESHAPE x1 */ \
    r.AddFullyConnected(TFLM_REG_FULLY_CONNECTED);    /* FULLY_CONNECTED x1 */ \
    r.AddSoftmax(TFLM_REG_SOFTMAX);    /* SOFTMAX x1 */ \
  } while (0)

#endif  // RESOLVER_GEN_H_
