// syscalls.cc — newlib 需要的最小 syscall 樁。
//
// 為什麼要連 newlib：TFLM 需要 libm。用 nm -u 看 TFLM 的 object 需要哪些外部
// 符號，數學類有 frexp / round / pow / exp / floor / sqrt —— 其中 frexp 在
// QuantizeMultiplier 裡（kernels/internal/quantization_util.cc），
// 是 int8 推論路徑上真的會呼叫到的。自己手寫這些風險太高，
// 直接用 toolchain 內附的 newlib libm。
//
// 好消息是 frexp / floor / round 都是「精確」運算，不產生捨入誤差，
// 而 rv64imc 沒有 FPU、double 走 libgcc 的軟浮點（IEEE-754，確定性），
// 所以 x86 與 RV64 之間的 BITEXACT 是做得到的。
//
// 連了 libc 之後它會要求這幾個 _xxx 樁。簽名刻意不 include newlib 的標頭
// （避免把 <sys/stat.h> 之類的東西拉進來）—— C 連結只看符號名，不看型別。

#include <stddef.h>
#include <stdint.h>

extern "C" {

void plat_puts(const char* s);
void plat_putc(char c);
void plat_exit(int code);

// eyalroz 的 printf（TFLM 的 riscv target 會連進來）用這個當輸出鉤子。
// 名字結尾的底線是那個函式庫的慣例，不是打錯。
void putchar_(char c) { plat_putc(c); }

extern char _end[], _heap_end[];      // link.ld 提供

void _exit(int code) {
  plat_exit(code);
  __builtin_unreachable();
}

void abort(void) {
  plat_puts("\n*** abort() ***\n");
  plat_exit(70);
  __builtin_unreachable();
}

// 任何走 stdio 的東西都會到這裡。我們的 debug_log.cc 不走 stdio
// （直接呼叫 plat_puts），所以這條路平常不會被用到。
long _write(int fd, const char* buf, long len) {
  (void)fd;
  for (long i = 0; i < len; ++i) plat_putc(buf[i]);
  return len;
}

long _read(int, char*, long) { return 0; }
int _close(int) { return -1; }
long _lseek(int, long, int) { return 0; }
int _isatty(int) { return 1; }
int _fstat(int, void*) { return -1; }
int _kill(int, int) { return -1; }
int _getpid(void) { return 1; }

// 靜態 heap。TFLM 用 tensor arena 不碰 malloc，但 libc 連進來之後
// 可能有路徑會呼叫，給它一塊固定區域，用完就回 -1（不要靜默壞掉）。
void* _sbrk(ptrdiff_t incr) {
  static char* cur = _end;
  char* prev = cur;
  if (cur + incr > _heap_end) {
    plat_puts("\n*** _sbrk: heap 用盡（link.ld 的 heap 要加大）***\n");
    return (void*)-1;
  }
  cur += incr;
  return prev;
}

void __assert_func(const char* file, int line, const char* fn,
                   const char* expr) {
  (void)line;
  (void)fn;
  plat_puts("\n*** assert 失敗: ");
  plat_puts(expr ? expr : "?");
  plat_puts("  @ ");
  plat_puts(file ? file : "?");
  plat_puts("\n");
  plat_exit(71);
}

int atexit(void (*)(void)) { return 0; }   // 永遠不會結束，不需要

void* __dso_handle = nullptr;

}  // extern "C"