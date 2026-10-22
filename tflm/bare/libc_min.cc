// libc_min.cc — 自己提供 string 與 math，不連 newlib。
//
// 為什麼不連 newlib：
//   newlib 的 libm 是用預設的 medlow code model 建的，lui+addi 絕對定址
//   只到 0x7fffffff。程式放在 0x80000000 會得到一堆
//       libm_a-s_frexp.o: relocation truncated to fit: R_RISCV_HI20
//   要修就得用 --with-cmodel=medany 重建整個 toolchain（幾小時），
//   或者把程式搬到低位址（那又得動 Spike 的記憶體配置）。
//
//   實際需要的東西其實很少，所以自己寫。
//   注意：連結錯誤裡只有 libm、沒有 libgcc，所以軟浮點那組是位置無關的，
//   -lgcc 可以留著。
//
// 哪些會被真的執行（int8 推論路徑）：
//   frexp   kernels/internal/quantization_util.cc 的 QuantizeMultiplier
//   round   同上（TFLM 的 TfLiteRound）
//   floor   softmax 的 CalculateInputRadius 之類的上界計算
//   這三個都是精確運算 —— 純位元操作，不產生捨入誤差 ——
//   所以跟 glibc 位元相同，x86 與 RV64 的 BITEXACT 不受影響。
//
//   exp / pow / tanhf / expm1f 等只是連結期參照。例如
//   kernels/softmax_common.cc 裡的 std::exp 在 `if (input->type ==
//   kTfLiteInt16)` 之內，我們的模型是 int8，執行時不會進去。
//   這些一律做成會 abort 的樁：真的被呼叫到會明確告訴你是哪一個，
//   而不是靜默算出錯的結果。

#include <stdint.h>
#include <stddef.h>

extern "C" void plat_puts(const char* s);
extern "C" void plat_exit(int code);

namespace {

union dbits {
  double d;
  uint64_t u;
};

[[noreturn]] void not_implemented(const char* name) {
  plat_puts("\n*** 呼叫了未實作的數學函式: ");
  plat_puts(name);
  plat_puts("\n    它本來應該只是連結期參照。bare/libc_min.cc 要補上實作。\n");
  plat_exit(72);
  __builtin_unreachable();
}

}  // namespace

extern "C" {

// ===================================================================
// string / memory
// ===================================================================

void* memcpy(void* dst, const void* src, size_t n) {
  unsigned char* d = static_cast<unsigned char*>(dst);
  const unsigned char* s = static_cast<const unsigned char*>(src);
  // 兩邊都 8 對齊時走 64-bit，TFLM 搬張量的量不小
  if (((reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s)) & 7) ==
      0) {
    uint64_t* d8 = reinterpret_cast<uint64_t*>(d);
    const uint64_t* s8 = reinterpret_cast<const uint64_t*>(s);
    for (size_t i = 0; i < n / 8; i++) d8[i] = s8[i];
    size_t done = (n / 8) * 8;
    d += done;
    s += done;
    n -= done;
  }
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
}

void* memset(void* dst, int c, size_t n) {
  unsigned char* d = static_cast<unsigned char*>(dst);
  unsigned char v = static_cast<unsigned char>(c);
  if ((reinterpret_cast<uintptr_t>(d) & 7) == 0) {
    uint64_t w = 0x0101010101010101ULL * v;
    uint64_t* d8 = reinterpret_cast<uint64_t*>(d);
    for (size_t i = 0; i < n / 8; i++) d8[i] = w;
    size_t done = (n / 8) * 8;
    d += done;
    n -= done;
  }
  for (size_t i = 0; i < n; i++) d[i] = v;
  return dst;
}

void* memmove(void* dst, const void* src, size_t n) {
  unsigned char* d = static_cast<unsigned char*>(dst);
  const unsigned char* s = static_cast<const unsigned char*>(src);
  if (d == s || n == 0) return dst;
  if (d < s) return memcpy(dst, src, n);
  for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];   // 重疊時要反向
  return dst;
}

int memcmp(const void* a, const void* b, size_t n) {
  const unsigned char* x = static_cast<const unsigned char*>(a);
  const unsigned char* y = static_cast<const unsigned char*>(b);
  for (size_t i = 0; i < n; i++)
    if (x[i] != y[i]) return static_cast<int>(x[i]) - static_cast<int>(y[i]);
  return 0;
}

size_t strlen(const char* s) {
  size_t n = 0;
  while (s[n]) n++;
  return n;
}

int strcmp(const char* a, const char* b) {
  while (*a && *a == *b) { a++; b++; }
  return static_cast<int>(static_cast<unsigned char>(*a)) -
         static_cast<int>(static_cast<unsigned char>(*b));
}

int strncmp(const char* a, const char* b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    unsigned char ca = static_cast<unsigned char>(a[i]);
    unsigned char cb = static_cast<unsigned char>(b[i]);
    if (ca != cb) return static_cast<int>(ca) - static_cast<int>(cb);
    if (ca == 0) break;
  }
  return 0;
}

char* strcpy(char* d, const char* s) {
  char* r = d;
  while ((*d++ = *s++) != 0) { }
  return r;
}

int isprint(int c) { return c >= 0x20 && c < 0x7f; }

// ===================================================================
// math —— 會被真的執行的三個，全部走整數位元操作
// ===================================================================

double fabs(double x) {
  dbits v;
  v.d = x;
  v.u &= ~(1ULL << 63);
  return v.d;
}
float fabsf(float x) { return x < 0.0f ? -x : x; }

// 回傳 [0.5, 1) 的尾數，並把指數寫到 *e，使 x = mantissa * 2^(*e)。
// 這是精確運算：只搬動指數欄位，尾數一個 bit 都沒動。
double frexp(double x, int* e) {
  dbits v;
  v.d = x;
  int ex = static_cast<int>((v.u >> 52) & 0x7ff);
  if (ex == 0x7ff) {           // inf / nan
    *e = 0;
    return x;
  }
  if (ex == 0) {               // 零或次正規數
    if ((v.u & ~(1ULL << 63)) == 0) {
      *e = 0;
      return x;                // ±0
    }
    v.d = x * 18014398509481984.0;          // 乘 2^54 先正規化
    ex = static_cast<int>((v.u >> 52) & 0x7ff);
    *e = ex - 1022 - 54;
  } else {
    *e = ex - 1022;
  }
  v.u = (v.u & ~(0x7ffULL << 52)) | (1022ULL << 52);
  return v.d;
}

// x * 2^n。乘以 2 的精確幂次不產生捨入誤差。
double ldexp(double x, int n) {
  if (x == 0.0 || n == 0) return x;
  // 分段相乘，避免中間值溢位或下溢
  while (n > 1023) { x *= 0x1p1023; n -= 1023; }
  while (n < -1022) { x *= 0x1p-1022; n += 1022; }
  dbits p;
  p.u = static_cast<uint64_t>(n + 1023) << 52;
  return x * p.d;
}
double scalbn(double x, int n) { return ldexp(x, n); }

double floor(double x) {
  dbits v;
  v.d = x;
  int ex = static_cast<int>((v.u >> 52) & 0x7ff) - 1023;
  if (ex >= 52) return x;                    // 已是整數，含 inf/nan
  if (ex < 0) {                              // |x| < 1
    if ((v.u & ~(1ULL << 63)) == 0) return x;  // ±0
    return (v.u >> 63) ? -1.0 : 0.0;
  }
  uint64_t mask = (1ULL << (52 - ex)) - 1;
  if ((v.u & mask) == 0) return x;           // 本來就是整數
  // 負數要往更小的方向。符號-數值表示法下，增加數值位元就是變得更負。
  if (v.u >> 63) v.u += (1ULL << (52 - ex));
  v.u &= ~mask;
  return v.d;
}
float floorf(float x) { return static_cast<float>(floor(static_cast<double>(x))); }

double ceil(double x) { return -floor(-x); }
float ceilf(float x) { return static_cast<float>(ceil(static_cast<double>(x))); }

// 四捨五入，.5 一律遠離零（C 標準的 round 語意，不是 banker's rounding）
double round(double x) {
  dbits v;
  v.d = x;
  int ex = static_cast<int>((v.u >> 52) & 0x7ff) - 1023;
  if (ex >= 52) return x;
  if (ex == -1) return (v.u >> 63) ? -1.0 : 1.0;      // 0.5 <= |x| < 1
  if (ex < -1) {                                       // |x| < 0.5
    v.u &= (1ULL << 63);                               // 保留符號，給 ±0
    return v.d;
  }
  uint64_t half = 1ULL << (51 - ex);
  uint64_t mask = (1ULL << (52 - ex)) - 1;
  if ((v.u & mask) == 0) return x;
  v.u += half;
  v.u &= ~mask;
  return v.d;
}
float roundf(float x) { return static_cast<float>(round(static_cast<double>(x))); }

double trunc(double x) { return (x < 0.0) ? -floor(-x) : floor(x); }
float truncf(float x) { return static_cast<float>(trunc(static_cast<double>(x))); }

float fmaxf(float a, float b) { return (a > b) ? a : b; }
float fminf(float a, float b) { return (a < b) ? a : b; }
double fmax(double a, double b) { return (a > b) ? a : b; }
double fmin(double a, double b) { return (a < b) ? a : b; }

// ===================================================================
// math —— 只是連結期參照，int8 路徑不會執行到。
// 真的被呼叫會印出名字再 abort，不要讓它靜默算錯。
// ===================================================================

double exp(double)            { not_implemented("exp"); }
float  expf(float)            { not_implemented("expf"); }
double pow(double, double)    { not_implemented("pow"); }
float  powf(float, float)     { not_implemented("powf"); }
double log(double)            { not_implemented("log"); }
float  logf(float)            { not_implemented("logf"); }
double log2(double)           { not_implemented("log2"); }
double log10(double)          { not_implemented("log10"); }
double sqrt(double)           { not_implemented("sqrt"); }
float  sqrtf(float)           { not_implemented("sqrtf"); }
float  sinf(float)            { not_implemented("sinf"); }
float  cosf(float)            { not_implemented("cosf"); }
double sin(double)            { not_implemented("sin"); }
double cos(double)            { not_implemented("cos"); }
float  tanhf(float)           { not_implemented("tanhf"); }
double tanh(double)           { not_implemented("tanh"); }
float  expm1f(float)          { not_implemented("expm1f"); }
double expm1(double)          { not_implemented("expm1"); }
float  fmodf(float, float)    { not_implemented("fmodf"); }
double fmod(double, double)   { not_implemented("fmod"); }
double atan2(double, double)  { not_implemented("atan2"); }
float  atan2f(float, float)   { not_implemented("atan2f"); }
double cbrt(double)           { not_implemented("cbrt"); }
float  rsqrtf(float)          { not_implemented("rsqrtf"); }

// flatbuffers 的 locale 相關路徑會參照這些，但只在解析 .fbs 文字時用到，
// 我們是直接讀二進位 flatbuffer，不會執行。
double strtod(const char*, char**)  { not_implemented("strtod"); }

// 某些 kernel 會參照 qsort。插入排序，相等元素保持原序（穩定）。
void qsort(void* base, size_t n, size_t sz,
           int (*cmp)(const void*, const void*)) {
  unsigned char* a = static_cast<unsigned char*>(base);
  for (size_t i = 1; i < n; i++) {
    for (size_t j = i; j > 0 && cmp(a + (j - 1) * sz, a + j * sz) > 0; j--) {
      for (size_t k = 0; k < sz; k++) {
        unsigned char t = a[(j - 1) * sz + k];
        a[(j - 1) * sz + k] = a[j * sz + k];
        a[j * sz + k] = t;
      }
    }
  }
}

// ===================================================================
// C++ runtime
// ===================================================================

void __cxa_pure_virtual(void) {
  plat_puts("\n*** 呼叫了純虛擬函式 ***\n");
  plat_exit(73);
}

}  // extern "C"

// TFLM 用 tensor arena，不碰 heap。這些只是為了讓連結過。
void* operator new(size_t) { not_implemented("operator new"); }
void* operator new[](size_t) { not_implemented("operator new[]"); }
void operator delete(void*) noexcept {}
void operator delete[](void*) noexcept {}
void operator delete(void*, size_t) noexcept {}
void operator delete[](void*, size_t) noexcept {}
