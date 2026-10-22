// plat.h — 平台層的契約。host 與 bare-metal 各自實作一份。
//
// 這裡只放「同一份 nn_main.cc 要在兩個平台上跑」所必須抽象掉的東西。
// 實作在 host/main_host.cc 與 bare/main_bare.cc。

#ifndef PLAT_H_
#define PLAT_H_

#include <stdint.h>

extern "C" {

// 輸出一行字。bare-metal 走 HTIF 的 bcd console。
// TFLM 的 DebugLog 也會呼叫這個（見 tflm-patch/debug_log.cc）。
void plat_puts(const char* s);

// 效能計數器。
//   bare-metal : rdinstret —— 已退休的指令數
//   host       : clock_gettime 的奈秒數
// 兩邊單位不同，所以 plat_counter_unit() 會說明自己是什麼，
// 報表才不會把 ns 當成指令數。
uint64_t plat_counter(void);
const char* plat_counter_unit(void);

// 把輸出存檔。host 寫 tflm_out.bin，bare-metal 是空實作。
void plat_dump_output(const void* data, unsigned len);

// 結束並回傳離開碼。
void plat_exit(int code);

}  // extern "C"

#endif  // PLAT_H_
