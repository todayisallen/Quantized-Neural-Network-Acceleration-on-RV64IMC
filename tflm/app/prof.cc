#include "prof.h"

#include "plat.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace {

// 整數百分比，回傳千分比以免用到浮點。
// bare-metal 上浮點要走軟浮點，印個百分比不值得付那個代價，
// 而且會污染我們正在量的指令數。
uint64_t permille(uint64_t x, uint64_t total) {
  if (total == 0) return 0;
  return (x * 1000ULL) / total;
}

void print_row(const char* name, uint64_t v, uint64_t total, int reps) {
  uint64_t pm = permille(v, total);
  if (reps >= 0) {
    MicroPrintf("  %-22s %4d  %12llu   %3llu.%llu%%", name, reps,
                (unsigned long long)v, (unsigned long long)(pm / 10),
                (unsigned long long)(pm % 10));
  } else {
    MicroPrintf("  %-22s       %12llu   %3llu.%llu%%", name,
                (unsigned long long)v, (unsigned long long)(pm / 10),
                (unsigned long long)(pm % 10));
  }
}

bool same_str(const char* a, const char* b) {
  if (a == b) return true;
  if (!a || !b) return false;
  while (*a && *a == *b) { a++; b++; }
  return *a == *b;
}

}  // namespace

uint32_t CountProfiler::BeginEvent(const char* tag) {
  if (n_ >= PROF_MAX_EVENTS) {
    overflow_++;
    return PROF_MAX_EVENTS;  // EndEvent 會忽略這個 handle
  }
  int h = n_++;
  tag_[h] = tag;
  delta_[h] = 0;
  // 最後才讀計數器，讓上面那些記帳動作不被算進這一層
  start_[h] = plat_counter();
  return static_cast<uint32_t>(h);
}

void CountProfiler::EndEvent(uint32_t event_handle) {
  uint64_t now = plat_counter();
  if (event_handle >= static_cast<uint32_t>(PROF_MAX_EVENTS)) return;
  delta_[event_handle] = now - start_[event_handle];
}

uint64_t CountProfiler::total() const {
  uint64_t s = 0;
  for (int i = 0; i < n_; i++) s += delta_[i];
  return s;
}

void CountProfiler::ReportPerLayer(uint64_t invoke_total) const {
  if (overflow_) {
    MicroPrintf("!! 事件數超過 PROF_MAX_EVENTS，漏掉 %d 個", overflow_);
  }
  MicroPrintf("");
  MicroPrintf("=== 每層（執行順序），單位 %s ===", plat_counter_unit());
  MicroPrintf("   idx  operator                       count   占Invoke");
  for (int i = 0; i < n_; i++) {
    uint64_t pm = permille(delta_[i], invoke_total);
    MicroPrintf("  %4d  %-22s %12llu   %3llu.%llu%%", i,
                tag_[i] ? tag_[i] : "?", (unsigned long long)delta_[i],
                (unsigned long long)(pm / 10), (unsigned long long)(pm % 10));
  }
}

void CountProfiler::ReportByKind(uint64_t invoke_total) const {
  const char* kind[PROF_MAX_KINDS];
  uint64_t sum[PROF_MAX_KINDS];
  int reps[PROF_MAX_KINDS];
  int nk = 0;

  for (int i = 0; i < n_; i++) {
    int k = -1;
    for (int j = 0; j < nk; j++)
      if (same_str(kind[j], tag_[i])) { k = j; break; }
    if (k < 0) {
      if (nk >= PROF_MAX_KINDS) continue;
      k = nk++;
      kind[k] = tag_[i];
      sum[k] = 0;
      reps[k] = 0;
    }
    sum[k] += delta_[i];
    reps[k]++;
  }

  // 由大到小排。選擇排序，最多 32 個，不需要更快的做法。
  for (int a = 0; a < nk; a++) {
    int best = a;
    for (int b = a + 1; b < nk; b++)
      if (sum[b] > sum[best]) best = b;
    if (best != a) {
      const char* ts = kind[a]; kind[a] = kind[best]; kind[best] = ts;
      uint64_t tv = sum[a];     sum[a] = sum[best];   sum[best] = tv;
      int tr = reps[a];         reps[a] = reps[best]; reps[best] = tr;
    }
  }

  MicroPrintf("");
  MicroPrintf("=== 依 operator 彙總，單位 %s ===", plat_counter_unit());
  MicroPrintf("  operator               次數         count   占Invoke");
  for (int a = 0; a < nk; a++)
    print_row(kind[a] ? kind[a] : "?", sum[a], invoke_total, reps[a]);
}

// ======================================================================
// X2：xqrq 的發出次數
// ======================================================================
// 定義放在 prof.cc 而不是 conv_xq.cc / dwconv_xq.cc，因為 B1D 之類的配置
// 只編其中一個 kernel —— 定義在任一個 kernel 裡都會讓另一個配置連結失敗。
// prof.cc 在每個配置、每個平台都會編，而這本來就是量測插樁的 TU。
//
// 宣告在 app/xqrq.h。用途跟 X1 的 XQ_REPORT 相同：
// 「數字變好了」不能證明「指令生效了」，要直接數次數。
#ifdef XQ_REPORT
long g_xqrq_calls = 0;
#endif