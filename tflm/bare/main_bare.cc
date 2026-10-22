// main_bare.cc — bare-metal RV64 的平台層。nn_main.cc 完全不改。
//
// 負責四件事：啟動（sp / gp / .bss / 全域建構子）、HTIF console、HTIF 結束、
// 以及 plat_dump_output 的空實作。

#include <stdint.h>
#include <stddef.h>

extern "C" {
int nn_main(void);

// 以下符號由 link.ld 提供
extern char _bss_start[], _bss_end[];
typedef void (*ctor_t)(void);
extern ctor_t __preinit_array_start[], __preinit_array_end[];
extern ctor_t __init_array_start[], __init_array_end[];
}

// ======================================================================
// HTIF
//
// 以下協定是對照 riscv-isa-sim/fesvr 的原始碼寫的，不是推測：
//
//   tohost 位元配置（fesvr/device.h 的 command_t）
//     device  = tohost >> 56
//     cmd     = tohost >> 48      （取低 8 bit）
//     payload = tohost[47:0]
//
//   device 編號（fesvr/htif.cc 的 register_devices()）
//     0 = syscall_proxy（先註冊）
//     1 = bcd，也就是 console
//
//   結束：device 0 / cmd 0 / payload 為奇數
//     fesvr/syscall.cc  payload & 1 -> exitcode = payload，然後 return，
//                       **不呼叫 respond()**
//     fesvr/htif.cc     exit_code() = exitcode >> 1
//     所以寫 tohost = (code << 1) | 1，離開碼就是 code。
//
//   印一個字元：device 1 / cmd 1 / payload = 字元
//     fesvr/device.cc  bcd_t::handle_write 只呼叫 terminal write，
//                      **同樣不呼叫 respond()**。
//
//   ** 關鍵 **：上面兩種操作都不會設 fromhost。
//   照直覺去 while (fromhost == 0) 等回應，會永遠等下去。
//   要等的是 tohost 歸零 —— fesvr/htif.cc 的主迴圈讀到非零的 tohost
//   之後會立刻把它寫回 0，那就是「host 已經取走」的訊號。
// ======================================================================

volatile uint64_t tohost   __attribute__((section(".tohost"), aligned(8))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(8))) = 0;

static inline void htif_send(uint64_t dev, uint64_t cmd, uint64_t payload) {
  tohost = (dev << 56) | (cmd << 48) | (payload & 0xffffffffffffULL);
  while (tohost != 0) {
    // host 取走後會寫 0。不要改成等 fromhost，見上面的說明。
  }
}

extern "C" void plat_putc(char c) {
  htif_send(1, 1, (uint64_t)(uint8_t)c);
}

extern "C" void plat_puts(const char* s) {
  if (!s) return;
  for (; *s; ++s) plat_putc(*s);
}

extern "C" void plat_exit(int code) {
  // 離開碼只有 0..127 可靠（shell 的離開碼是 8 bit），
  // 所以這裡傳的是 nn_main 的 0/1/2，不是不一致的個數。
  tohost = ((uint64_t)((uint32_t)code & 0x7f) << 1) | 1;
  for (;;) {
  }
}

// bare-metal 沒有檔案系統。golden 是編譯進去的，不需要存檔。
extern "C" void plat_dump_output(const void*, unsigned) {}

// ======================================================================
// 效能計數器
// ======================================================================
//
// rdinstret 讀的是「已退休的指令數」(CSR 0xC02)。-march 有 zicsr，
// 所以組譯器直接認得這個助憶符，不需要用 .insn 繞過。
//
// Spike 會老實地逐條累加，所以這個數字是確定性的 —— 同樣的程式、同樣的輸入，
// 每次跑都一樣。這正是四個配置 B0/B1/X1/X2 能拿來比較的前提。
// （cycle 數就不行：Spike 不是 cycle-accurate，那要等期末的 RTL。）
extern "C" uint64_t plat_counter(void) {
  uint64_t x;
  __asm__ volatile("rdinstret %0" : "=r"(x));
  return x;
}
extern "C" const char* plat_counter_unit(void) { return "instret"; }

// ======================================================================
// 啟動
// ======================================================================

// naked：還沒有 sp 之前不能有 function prologue。
// .text.init：link.ld 把這一節放最前面，確保 _start 在 0x80000000。
extern "C" __attribute__((naked, section(".text.init"))) void _start(void) {
  __asm__ volatile(
      // gp 要在任何 gp 相對定址之前設好。norelax 包起來是標準 crt0 寫法：
      // 設定 gp 的那兩條指令本身不能被鬆弛成 gp 相對。
      ".option push\n"
      ".option norelax\n"
      "la gp, __global_pointer$\n"
      ".option pop\n"
      // -nostartfiles 不連 crt0，sp 必須自己設，否則第一個函式呼叫就當掉。
      "la sp, _stack_top\n"
      "j  cstart\n");
}

extern "C" void cstart(void) {
  // 1. 清 .bss。tensor arena（約 1.65 MB）在這裡。
  //    TFLM 自己會初始化 arena，但它有靜態旗標假設初值為 0。
  for (uint64_t* p = reinterpret_cast<uint64_t*>(_bss_start);
       p < reinterpret_cast<uint64_t*>(_bss_end); ++p) {
    *p = 0;
  }

  // 2. 全域建構子。TFLM 是 C++，少了這步靜態物件不會初始化。
  for (ctor_t* f = __preinit_array_start; f != __preinit_array_end; ++f) (*f)();
  for (ctor_t* f = __init_array_start; f != __init_array_end; ++f) (*f)();

  // 3. 推論。離開碼 0 = BITEXACT、2 = CLOSE、1 = 分歧或失敗。
  plat_exit(nn_main());
}