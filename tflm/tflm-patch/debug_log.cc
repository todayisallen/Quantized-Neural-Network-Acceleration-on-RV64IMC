/* 放到 tensorflow/lite/micro/riscv64_generic/debug_log.cc
 *
 * TFLM 的 Makefile 會用 specialize_files.py 把
 * tensorflow/lite/micro/debug_log.cc 換成這一份（因為 TARGET=riscv64_generic）。
 *
 * 為什麼要換掉：原本那份用 vfprintf(stderr, ...)，那需要 newlib 的 stdio，
 * 而 stdio 需要一整套 syscall 與檔案緩衝區。bare-metal 上我們只要能吐字元，
 * 所以改成「eyalroz 的 vsnprintf_ 格式化 -> plat_puts 送 HTIF」。
 *
 * plat_puts() 由應用層提供（tflm/bare/main_bare.cc）。
 */
#include "tensorflow/lite/micro/debug_log.h"

#include <cstddef>

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include "eyalroz_printf/src/printf/printf.h"

// tflm/bare/main_bare.cc 提供，走 HTIF 的 bcd console
extern "C" void plat_puts(const char* s);
#endif

extern "C" void DebugLog(const char* format, va_list args) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  // 256 bytes 在 stack 上。TFLM 的錯誤訊息都比這個短。
  constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];
  vsnprintf_(log_buffer, kMaxLogLen, format, args);
  plat_puts(log_buffer);
#endif
}

#ifndef TF_LITE_STRIP_ERROR_STRINGS
extern "C" int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                              va_list vlist) {
  return vsnprintf_(buffer, buf_size, format, vlist);
}
#endif
