// main_host.cc — x86 版的平台層：進入點與輸出存檔。
//
// DebugLog()（MicroPrintf 最終呼叫的東西）在 host 上由 libtensorflow-microlite.a
// 自己提供（TARGET=linux 會編進 tensorflow/lite/micro/debug_log.cc），這裡不用管。
// bare-metal 版沒有，必須自己寫一份寫到 HTIF。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

extern "C" int nn_main(void);

extern "C" void plat_exit(int code) { std::exit(code); }

// host 上沒有「已退休指令數」這種東西，用奈秒代替。
// plat_counter_unit() 會說明單位，報表才不會把 ns 當成指令數。
// host 的數字只用來確認量測機制本身可用；真正要比的數字來自 Spike。
extern "C" uint64_t plat_counter(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
extern "C" const char* plat_counter_unit(void) { return "ns"; }

extern "C" void plat_puts(const char* s) { std::fputs(s, stdout); }

// 把 TFLM 在 x86 上的輸出存下來。這份才是 Spike 要比對的 golden，
// 用 tools/gen_golden_from_bin.py 轉成 C 陣列。
// 路徑可用 TFLM_OUT 環境變數覆蓋。
extern "C" void plat_dump_output(const void* data, unsigned len) {
  const char* path = std::getenv("TFLM_OUT");
  if (!path) path = "tflm_out.bin";
  std::FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::printf("(警告: 無法寫入 %s)\n", path);
    return;
  }
  size_t w = std::fwrite(data, 1, len, f);
  std::fclose(f);
  if (w != len)
    std::printf("(警告: %s 只寫入 %zu/%u bytes)\n", path, w, len);
  else
    std::printf("輸出已存到 %s (%u bytes)\n", path, len);
}

int main(void) {
  int rc = nn_main();
  switch (rc) {
    case 0:  std::printf("\n== HOST BITEXACT ==\n"); break;
    case 2:  std::printf("\n== HOST CLOSE ==\n");    break;
    default: std::printf("\n== HOST FAIL (rc=%d) ==\n", rc); break;
  }
  return rc;
}