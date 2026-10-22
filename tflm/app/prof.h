// prof.h — 以 TFLM 的 MicroProfilerInterface 收集每一層的指令數。
//
// TFLM 在 micro_interpreter_graph.cc 的 invoke 迴圈裡，用 ScopedMicroProfiler
// 把每個 operator 的 invoke 包起來，tag 是 operator 名稱（"CONV_2D" 之類）。
// 我們只要實作 BeginEvent / EndEvent，在兩端讀 plat_counter() 就有每層的數字。
//
// 兩件要注意的事：
//
// 1. 那個 ScopedMicroProfiler 被 #if !defined(TF_LITE_STRIP_ERROR_STRINGS)
//    包住。所以 B1 若開了 STRIP_ERROR_STRINGS，per-layer 的數字就會全變 0。
//    B1 要分兩個 build：量測用的保留字串，量 code size 的砍掉。
//
// 2. profiler 只包 Invoke，不包 Prepare。TFLM 的 Prepare() 是在
//    AllocateTensors() 裡面跑的，所以分段是
//      AllocateTensors（記憶體規劃 + Prepare，含浮點的 multiplier 計算）
//      Invoke（純推論，Xqmac8 要加速的就是這段）

#ifndef PROF_H_
#define PROF_H_

#include <stdint.h>

#include "tensorflow/lite/micro/micro_profiler_interface.h"

// 我們的模型是 64 個 op，開 128 有餘裕
#define PROF_MAX_EVENTS 128
#define PROF_MAX_KINDS 32

class CountProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override;
  void EndEvent(uint32_t event_handle) override;

  void Reset() { n_ = 0; overflow_ = 0; }
  int count() const { return n_; }

  // 所有層的計數加總。跟 Invoke 的總數相減就是框架的 overhead，
  // 也就是 B1 要削的那一塊。
  uint64_t total() const;

  // invoke_total 用來算百分比；傳 0 就不印百分比。
  void ReportPerLayer(uint64_t invoke_total) const;
  void ReportByKind(uint64_t invoke_total) const;

 private:
  const char* tag_[PROF_MAX_EVENTS];
  uint64_t start_[PROF_MAX_EVENTS];
  uint64_t delta_[PROF_MAX_EVENTS];
  int n_ = 0;
  int overflow_ = 0;
};

#endif  // PROF_H_
