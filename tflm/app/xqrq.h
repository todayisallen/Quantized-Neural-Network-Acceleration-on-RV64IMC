// xqrq.h — X2 的再量化指令：打包、呼叫、以及 host 上的模擬。
//
// 語意規格見 docs/X2_xqrq_design.md。這裡只放兩個 kernel 共用的部分。
//
//   xqrq rd, rs1, rs2
//     rs1 = int32 累加值
//     rs2 = multiplier(32) | shift(8) | offset(8) | min(8) | max(8)
//     rd  = 再量化後的 int8（已符號擴展）
//
// 一條指令做完規格書 §3 列的整條鏈：
//   fixed-point multiply → rounding → shift → zero-point addition → saturation
//
// ======================================================================
// 為什麼打包，而不是多開運算元
// ======================================================================
// R-type 只有兩個來源暫存器，這條指令需要五個參數。前兩個
// （multiplier / shift）是 per-channel，後三個是 per-layer 常數。
//
// 關鍵是打包**每層只做一次**：兩個 kernel 本來就有一趟 per-layer 的
// 前置迴圈（算 Σw），打包表掛在同一趟，成本 O(C)，攤提到 OH*OW 個
// 輸出位置上。在呼叫端即時組描述子會需要 ld+ld+slli+or 四條指令去餵
// 一條指令，省下的大半被吃掉 —— 那是初版設計的錯誤。
//
// ======================================================================
// host 模擬（XQ_SIM_INSN）刻意不自己實作捨入
// ======================================================================
// MultiplyByQuantizedMultiplier 有兩套實作（單次捨入 / gemmlowp 雙次
// 捨入），差 1 個 LSB。host 的模擬直接呼叫 TFLM 本體，所以：
//
//   host 的等價測試驗證「打包與四個呼叫點」—— 不驗證捨入
//   bare/t_xqrq 在 Spike 上驗證「指令本身的捨入」—— 用 TFLM 當 golden
//
// 兩者分工明確。自己轉錄一份捨入邏輯當 golden 等於把選錯捨入模式的
// 風險再引入一次，所以不做。

#ifndef APP_XQRQ_H_
#define APP_XQRQ_H_

#include <stdint.h>

#include "tensorflow/lite/micro/kernels/internal/common.h"

// 描述子的位元配置。改這裡就要同步改 riscv/insns/xqrq.h。
#define XQRQ_SH_MULT 32
#define XQRQ_SH_SHIFT 24
#define XQRQ_SH_OFFSET 16
#define XQRQ_SH_MIN 8
#define XQRQ_SH_MAX 0

// 把五個參數打包成一個 64 位元描述子。
//
// multiplier 是 int32（TFLM 保證非負），其餘四個都在 int8 範圍內：
//   shift  −31 … 30            （TFLITE_DCHECK 保證）
//   offset output zero point   （int8 張量的 zero point 必在 int8 內）
//   min/max quantized_activation_min/max（同上）
static inline uint64_t xqrq_pack(int32_t mult, int32_t shift, int32_t offset,
                                 int32_t lo, int32_t hi) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(mult)) << XQRQ_SH_MULT) |
         (static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(shift)))
          << XQRQ_SH_SHIFT) |
         (static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(offset)))
          << XQRQ_SH_OFFSET) |
         (static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(lo)))
          << XQRQ_SH_MIN) |
         (static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(hi)))
          << XQRQ_SH_MAX);
}

// 解包。給 host 模擬與 bare/t_xqrq 共用，確保兩邊對位元配置的理解一致
// —— 這類「五個欄位擠在一個字裡」的程式碼，最常見的 bug 是移位量寫錯
// 一個而讀到鄰居的位元，而解碼只寫一份就不會兩邊不一致。
static inline int32_t xqrq_mult(uint64_t d) {
  return static_cast<int32_t>(static_cast<uint32_t>(d >> XQRQ_SH_MULT));
}
static inline int32_t xqrq_shift(uint64_t d) {
  return static_cast<int8_t>((d >> XQRQ_SH_SHIFT) & 0xff);
}
static inline int32_t xqrq_offset(uint64_t d) {
  return static_cast<int8_t>((d >> XQRQ_SH_OFFSET) & 0xff);
}
static inline int32_t xqrq_lo(uint64_t d) {
  return static_cast<int8_t>((d >> XQRQ_SH_MIN) & 0xff);
}
static inline int32_t xqrq_hi(uint64_t d) {
  return static_cast<int8_t>((d >> XQRQ_SH_MAX) & 0xff);
}

#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY) && \
    !defined(DWCONV_XQ_KERNEL_ONLY)
// 地面真相：xqrq 實際發出幾次。應該等於走特化路徑的輸出元素數。
//
// 「數字變好了」不能證明「指令生效了」—— X1 階段踩過一次（include 順序
// 讓自訂 kernel 根本沒被註冊，但程式跑得完全正常，量到 1.13x 的假數字）。
// 這個計數器會改變指令數，所以 REPORT 版進獨立的 build 目錄，不用來量效能。
extern long g_xqrq_calls;
#endif

// ======================================================================
// 指令本體
// ======================================================================
// 刻意**不加** volatile：這是輸入的純函式，讓 GCC 能做 CSE 與提升。
// xqmac8 用 volatile 是因為它讀寫同一個累加器、有順序相依。
static inline int32_t xqrq_apply(int32_t acc, uint64_t desc) {
#if defined(XQ_REPORT) && !defined(CONV_XQ_KERNEL_ONLY) && \
    !defined(DWCONV_XQ_KERNEL_ONLY)
  ++g_xqrq_calls;
#endif

#ifdef XQ_SIM_INSN
  // host：捨入交給 TFLM 本體，見檔頭說明
  int32_t v = tflite::MultiplyByQuantizedMultiplier(acc, xqrq_mult(desc),
                                                    xqrq_shift(desc));
  v += xqrq_offset(desc);
  const int32_t lo = xqrq_lo(desc);
  const int32_t hi = xqrq_hi(desc);
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return v;
#else
  int64_t out;
  __asm__(".insn r 0x0b, 2, 0, %0, %1, %2"
          : "=r"(out)
          : "r"(static_cast<int64_t>(acc)), "r"(desc));
  return static_cast<int32_t>(out);
#endif
}

#endif  // APP_XQRQ_H_
