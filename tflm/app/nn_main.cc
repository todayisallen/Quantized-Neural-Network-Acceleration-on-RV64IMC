// nn_main.cc — 可攜的推論主體。x86 與 bare-metal RV64IMC 共用同一份。
//
// 平台相依的東西一律不放這裡：
//   * main() / cmain()      -> host/main_host.cc 或 bare/main_bare.cc
//   * DebugLog()            -> 同上（MicroPrintf 透過它輸出）
//
// 所有資料都是編譯進去的 C 陣列，不碰檔案系統 ——
// 因為 bare-metal 沒有檔案系統，而兩邊餵一樣的 bytes 才能談位元級比較。

#include <stdint.h>
#include <string.h>

// micro_interpreter.h 已經用 IWYU pragma: export 帶出 schema_generated.h，
// 所以 Model 與 TFLITE_SCHEMA_VERSION 都拿得到。
// 不自己 include 那個 header：它的路徑在 TFLM 版本之間搬過
// （tensorflow/lite/schema/ -> tensorflow/lite/micro/schema/），寫死會壞。
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "golden_data.h"
#include "input_data.h"
#include "model_data.h"
// B1 / X1 才會定義 USE_CONV_XQ。B0 不連 conv_xq.cc，
// 所以 B0 的 code size 不會被沒用到的 kernel 污染。
#ifdef USE_CONV_XQ
#include "conv_xq.h"
#endif
#ifdef USE_DWCONV_XQ
#include "dwconv_xq.h"
#endif
#include "plat.h"
#include "prof.h"
#include "resolver_gen.h"

// 第一次先開大，用 arena_used_bytes() 量出實際需求後再縮。
// bare-metal 這塊是靜態配置的，縮小它直接決定 .bss 大小。
#ifndef ARENA_KB
#define ARENA_KB 8192
#endif

#define TOPK 5

// 輸出張量的元素型別由 golden_data.h 的 GOLDEN_DATA_SIGNED 決定，不能寫死。
// EfficientNet-Lite0 的 I/O 是 uint8（內部才是 int8）；
// 用 int8 去讀 uint8 會把 128~255 當成負數，top-k 排序會整個錯掉。
#if GOLDEN_DATA_SIGNED
typedef int8_t out_t;
#else
typedef uint8_t out_t;
#endif

alignas(16) static uint8_t g_arena[ARENA_KB * 1024];

// profiler 放在靜態區而不是 stack：它有 128 筆 ×（tag + 兩個 64-bit），
// 放 stack 會吃掉 3 KB，而 bare-metal 的 stack 我們是自己配的。
static CountProfiler g_prof;

// 把 v 裡最大的 k 個元素的索引由大到小寫進 idx。
static void top_k(const out_t* v, unsigned n, int k, int* idx) {
  for (int r = 0; r < k; r++) {
    int best = -1, bv = -100000;
    for (unsigned i = 0; i < n; i++) {
      bool used = false;
      for (int j = 0; j < r; j++)
        if (idx[j] == (int)i) used = true;
      if (!used && (int)v[i] > bv) {
        bv = (int)v[i];
        best = (int)i;
      }
    }
    idx[r] = best;
  }
}

// 離開碼: 0 = 位元一致, 2 = 接近（top-1 相同且差異很小）, 1 = 分歧或失敗
extern "C" int nn_main(void) {
  MicroPrintf("--- nn_main: arena %d KiB ---", (int)ARENA_KB);

  // 印出輸入與 golden 的來源。這兩個字串由 tools/gen_io.py 編進程式，
  // 所以每次執行都會自己報告「這份數字是哪個輸入跑出來的」。
  //
  // 不印的話，不同輸入的量測結果混在一起只能靠事後比對發現 ——
  // 實際踩過一次：真實圖片與 seed 0 的指令數差 94,157，
  // 一度被誤判成建置系統的問題。
#ifdef INPUT_DATA_SRC
  MicroPrintf("    輸入: %s", INPUT_DATA_SRC);
#endif
#ifdef GOLDEN_DATA_SRC
  MicroPrintf("    golden: %s", GOLDEN_DATA_SRC);
#endif

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("FAIL schema: model=%d runtime=%d", (int)model->version(),
                (int)TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // TFLM_OP_COUNT / TFLM_ADD_OPS 由 tools/model_info.py 從模型產生。
  tflite::MicroMutableOpResolver<TFLM_OP_COUNT> resolver;
  TFLM_ADD_OPS(resolver);

  // 第 5 個參數是 MicroResourceVariables（我們的模型沒有），第 6 個是 profiler。
  tflite::MicroInterpreter interpreter(model, resolver, g_arena,
                                       sizeof(g_arena),
                                       /*resource_variables=*/nullptr,
                                       /*profiler=*/&g_prof);

  // AllocateTensors() 做兩件事：記憶體規劃，以及呼叫每個 op 的 Prepare()。
  // Prepare() 才是浮點工作所在（QuantizeMultiplier 用 frexp 與 double 除法）。
  // 它只跑一次，所以要跟 Invoke 分開計 —— 混在一起會稀釋 Xqmac8 的加速比。
  uint64_t t0 = plat_counter();
  TfLiteStatus alloc_st = interpreter.AllocateTensors();
  uint64_t c_alloc = plat_counter() - t0;
  if (alloc_st != kTfLiteOk) {
    MicroPrintf("FAIL AllocateTensors (arena %d KiB 可能不足)", (int)ARENA_KB);
    return 1;
  }
  // 這個數字要抄下來，它決定 bare-metal 的 arena 大小。
  MicroPrintf("arena_used_bytes = %u  (= %u KiB)",
              (unsigned)interpreter.arena_used_bytes(),
              (unsigned)((interpreter.arena_used_bytes() + 1023) / 1024));

  TfLiteTensor* in = interpreter.input(0);
  if (in->bytes != g_input_data_len) {
    MicroPrintf("FAIL input size: tensor=%u data=%u", (unsigned)in->bytes,
                (unsigned)g_input_data_len);
    return 1;
  }
  memcpy(in->data.raw, g_input_data, g_input_data_len);

  // profiler 在 Invoke 期間由 TFLM 自己呼叫（micro_interpreter_graph.cc 的
  // ScopedMicroProfiler 把每個 op 的 invoke 包起來）。先清空，避免把
  // AllocateTensors 期間可能產生的事件混進來。
  g_prof.Reset();
  uint64_t t1 = plat_counter();
  TfLiteStatus inv_st = interpreter.Invoke();
  uint64_t c_invoke = plat_counter() - t1;
  if (inv_st != kTfLiteOk) {
    MicroPrintf("FAIL Invoke");
    return 1;
  }

  TfLiteTensor* out = interpreter.output(0);
  if (out->bytes != g_golden_data_len) {
    MicroPrintf("FAIL output size: tensor=%u golden=%u", (unsigned)out->bytes,
                (unsigned)g_golden_data_len);
    return 1;
  }
  // 從 raw 取指標再轉型，避免依賴 data.int8 / data.uint8 哪個成員
  const out_t* got = reinterpret_cast<const out_t*>(out->data.raw);
  const out_t* want = reinterpret_cast<const out_t*>(g_golden_data);
  const unsigned n = g_golden_data_len;

  // 由平台層決定要不要把輸出存檔。
  // host 會寫成 tflm_out.bin，那份才是 Spike 要比對的 golden ——
  // 我們要驗的是「同一套 C++ 跑在 RV64 上跟跑在 x86 上位元相同」，
  // 拿 LiteRT 當 golden 的話，bare-metal 會帶著這裡的跨框架差異，
  // 就分不出那是正常繼承還是移植出錯。bare-metal 版是空實作。
  plat_dump_output(got, n);

  // ---------------------------------------------------------------- 差異統計
  //
  // 刻意不用單一門檻做二元判定。TFLM 與 LiteRT 在 requantize 的平手點
  // （乘積剛好落在 .5）可能處理方式不同，差異會隨網路深度累積：
  // 淺的網路通常完全一致，深的網路可能多數元素差 1~2 LSB 而 top-1 仍正確。
  // 這兩種情況的意義完全不同，所以分開報。
  unsigned mismatch = 0;
  int maxdiff = 0;
  unsigned hist[7] = {0};      // <=-3, -2, -1, 0, +1, +2, >=+3
  for (unsigned i = 0; i < n; i++) {
    int d = (int)got[i] - (int)want[i];
    if (d) mismatch++;
    int ad = d < 0 ? -d : d;
    if (ad > maxdiff) maxdiff = ad;
    int b = d < -3 ? 0 : (d > 3 ? 6 : d + 3);
    hist[b]++;
  }

  MicroPrintf("比對 %u 個輸出：不一致 %u (%u%%)，最大差 %d", n, mismatch,
              n ? (unsigned)((uint64_t)mismatch * 100 / n) : 0u, maxdiff);
  MicroPrintf("差異分布  <=-3:%u  -2:%u  -1:%u  0:%u  +1:%u  +2:%u  >=+3:%u",
              hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6]);

  // ---------------------------------------------------------------- top-k
  int tg[TOPK], tw[TOPK];
  top_k(got, n, TOPK, tg);
  top_k(want, n, TOPK, tw);

  int same_order = 0, same_set = 0;
  for (int r = 0; r < TOPK; r++) {
    if (tg[r] == tw[r]) same_order++;
    for (int j = 0; j < TOPK; j++)
      if (tg[r] == tw[j]) { same_set++; break; }
  }
  for (int r = 0; r < TOPK; r++) {
    MicroPrintf("  top%d  TFLM: class %4d q=%4d  |  golden: class %4d q=%4d",
                r + 1, tg[r], (int)got[tg[r]], tw[r], (int)want[tw[r]]);
  }
  MicroPrintf("top-%d 同序 %d/%d，同集合 %d/%d", TOPK, same_order, TOPK,
              same_set, TOPK);
  // 單獨印一行讓腳本好抓。同序的數字看不出 top-1 本身有沒有對上。
  MicroPrintf("top-1 %s", tg[0] == tw[0] ? "相同" : "不同");

  // ---------------------------------------------------------------- 效能
  //
  // AllocateTensors 與 Invoke 一定要分開看：
  //   AllocateTensors  一次性，含 Prepare 的浮點工作（frexp + double 除法）
  //   Invoke           每次推論都要做，Xqmac8 / 要加速的是這一段
  // 把兩者混在一起算加速比會嚴重低估 X1/X2 的效果。
  //
  // kernel 合計與 Invoke 的差額是框架 overhead（graph 迴圈、temp 配置重置、
  // 張量結構搬運）。那正是 B1 要削的東西，所以這個差額本身就是一個指標。
  {
    uint64_t kern = g_prof.total();
    uint64_t over = (c_invoke > kern) ? (c_invoke - kern) : 0;
    MicroPrintf("");
    MicroPrintf("=== 分段計數，單位 %s ===", plat_counter_unit());
    MicroPrintf("  AllocateTensors(含Prepare) %14llu", (unsigned long long)c_alloc);
    MicroPrintf("  Invoke                     %14llu", (unsigned long long)c_invoke);
    MicroPrintf("    其中 kernel 合計         %14llu  (%llu.%llu%%)",
                (unsigned long long)kern,
                (unsigned long long)(c_invoke ? kern * 1000 / c_invoke / 10 : 0),
                (unsigned long long)(c_invoke ? kern * 1000 / c_invoke % 10 : 0));
    MicroPrintf("    框架 overhead            %14llu  (%llu.%llu%%)",
                (unsigned long long)over,
                (unsigned long long)(c_invoke ? over * 1000 / c_invoke / 10 : 0),
                (unsigned long long)(c_invoke ? over * 1000 / c_invoke % 10 : 0));
    MicroPrintf("  合計                       %14llu",
                (unsigned long long)(c_alloc + c_invoke));

    if (g_prof.count() == 0) {
      MicroPrintf("!! 沒有收到任何 per-layer 事件。");
      MicroPrintf("   TFLM 的 ScopedMicroProfiler 被 #if !defined("
                  "TF_LITE_STRIP_ERROR_STRINGS) 包住，");
      MicroPrintf("   所以開了那個旗標就量不到每層。量測用的 build 不要開它。");
    } else {
      g_prof.ReportByKind(c_invoke);
      g_prof.ReportPerLayer(c_invoke);
    }
    MicroPrintf("");
  }

  // ---------------------------------------------------------------- 判定
  if (mismatch == 0) {
    MicroPrintf("BITEXACT  %u 個輸出全部位元一致", n);
    return 0;
  }
  // 判定依據是「top-1 是否保持」加上「不一致的比例」，不是最大差。
  //
  // 最初我用最大差 <= 2 當門檻，那是拿單一樣本的結果反推出來的，
  // 換個種子就把正常情況誤判成 DIVERGENT（seed 1 差 4、seed 2 差 3，
  // 但不一致率都只有約 1%、top-5 集合完全相同）。
  //
  // 有意義的訊號是分布的形狀：
  //   約 1% 不一致、集中在 ±1~4   -> 平手值進位策略不同，隨深度累積
  //   80% 以上不一致、±3 以上鋪開 -> 某個 operator 真的不一樣
  //
  // 再強調一次：跨框架比較沒有先驗的門檻，下面的 5% 只是方便閱讀的慣例。
  // 真正要看的是上面印出來的原始數字。唯一有原則的驗收條件是
  // RV64 vs x86 必須 BITEXACT —— 同一套程式碼，沒有容忍空間。
  const unsigned rate_pm = n ? (unsigned)((uint64_t)mismatch * 1000 / n) : 0;
  if (tg[0] == tw[0] && rate_pm <= 50) {
    MicroPrintf("CLOSE  top-1 相同、不一致率 %u.%u%%（<=5%%），最大差 %d LSB",
                rate_pm / 10, rate_pm % 10, maxdiff);
    MicroPrintf("       跨框架差異屬正常範圍。之後 Spike 要對的 golden"
                " 是這份 TFLM 輸出，不是 LiteRT");
    return 2;
  }
  MicroPrintf("DIVERGENT  不一致率 %u.%u%%，top-1 %s", rate_pm / 10,
              rate_pm % 10, tg[0] == tw[0] ? "相同" : "不同");
  MicroPrintf("           看上面的差異分布：鋪開到 ±3 以上代表某個 operator"
              " 不一樣，要逐一查");
  return 1;
}