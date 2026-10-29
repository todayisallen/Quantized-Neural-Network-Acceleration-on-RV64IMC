// xqrq rd, rs1, rs2 — 再量化指令（X2）
//
// custom-0 (opcode 0x0b), funct3=2, funct7=0
//   MATCH_XQRQ 0x0000200b   MASK_XQRQ 0xfe00707f
//
//   rs1 = int32 累加值
//   rs2 = multiplier(32) | shift(8) | offset(8) | min(8) | max(8)
//   rd  = 再量化後的 int8（符號擴展）
//
// 一條指令做完規格書 §3 列的整條鏈：
//   fixed-point multiply -> rounding -> shift -> zero-point addition
//   -> saturation
//
// rd 只寫不讀（與 xqmac8 不同）—— 這是純轉換，不累加，
// 所以硬體上少一個讀取埠。
//
// ======================================================================
// 捨入語意
// ======================================================================
// 必須與 TFLM 的 MultiplyByQuantizedMultiplier **位元相同**。
// 那個函式有兩套實作，差 1 個 LSB：
//
//   單次捨入（TFLITE_SINGLE_ROUNDING=1）：64 位元乘、加 round、算術右移
//   雙次捨入（gemmlowp）：SRDHM 在 2^31 處捨入，再 RoundingDivideByPOT
//
// 本專案實測是**雙次捨入** —— bare/t_xqrq 啟動時會自己印出 TFLM 編譯成
// 哪一種，不必靠 grep 猜。單次捨入版見 docs/X2_xqrq_design.md §2.1。
//
// 兩個實測逼出來的細節（都由 t_xqrq 的 directed 案例守住）：
//
//   1. `x * (1 << left)` 是 **int32** 乘法，溢位會環繞。不能改成
//      64 位元 —— 大的累加值配上正的 shift 正好走這條路，兩者會分岔。
//      （t_xqrq 的 DBLROUND1/3/4）
//
//   2. SRDHM 的最後一步 gemmlowp 寫的是 `/ (1ll << 31)` —— 整數除法，
//      對負數**往零截斷**。寫成 `>> 31`（往下取整）在負數上會差 1。
//      （t_xqrq 的 TRUNCDIV1..3）
//
// 所有區域變數都加 rq_ 前綴，避免跟 spike 的 insn 巨集撞名。

// ---------------------------------------------------------------- 解碼
const int32_t rq_acc = (int32_t)(RS1 & 0xffffffffULL);
const int32_t rq_mult = (int32_t)(uint32_t)(RS2 >> 32);
const int32_t rq_shift = (int32_t)(int8_t)((RS2 >> 24) & 0xff);
const int32_t rq_off = (int32_t)(int8_t)((RS2 >> 16) & 0xff);
const int32_t rq_lo = (int32_t)(int8_t)((RS2 >> 8) & 0xff);
const int32_t rq_hi = (int32_t)(int8_t)(RS2 & 0xff);

// ------------------------------------------- MultiplyByQuantizedMultiplier
const int rq_left = rq_shift > 0 ? rq_shift : 0;
const int rq_right = rq_shift > 0 ? 0 : -rq_shift;

// 見上面細節 1：int32 乘法，溢位環繞
const int32_t rq_x = (int32_t)((uint32_t)rq_acc << rq_left);

// SaturatingRoundingDoublingHighMul
const int64_t rq_ab = (int64_t)rq_x * (int64_t)rq_mult;
const int32_t rq_nudge = (rq_ab >= 0) ? (1 << 30) : (1 - (1 << 30));
// 見上面細節 2：除法而非 >> 31
int32_t rq_v =
    (int32_t)((rq_ab + (int64_t)rq_nudge) / (int64_t)(1LL << 31));
// gemmlowp 的飽和特例：兩個運算元都是 INT32_MIN。
// TFLM 的 DCHECK 保證 multiplier 非負，所以實際走不到，
// 但照抄以免語意有缺口。
if (rq_x == (-2147483647 - 1) && rq_mult == (-2147483647 - 1)) {
  rq_v = 2147483647;
}

// RoundingDivideByPOT：四捨五入、離零方向
if (rq_right > 0) {
  const int32_t rq_mask = (int32_t)((1LL << rq_right) - 1);
  const int32_t rq_rem = rq_v & rq_mask;
  const int32_t rq_thr = (rq_mask >> 1) + (rq_v < 0 ? 1 : 0);
  rq_v = (rq_v >> rq_right) + (rq_rem > rq_thr ? 1 : 0);
}

// ------------------------------------------- zero-point addition + saturation
//
// 這一步**必須在 int32 裡做、溢位環繞**，不能用 int64。
//
// kernel 的軟體參考是：
//     acc = MultiplyByQuantizedMultiplier(...);   // int32
//     acc += out_off;                              // int32，溢位環繞
// 用 int64 相加會在 |rq| 接近 INT32_MAX 時與參考分岔 —— 實測有 9/10000
// 筆隨機案例落在那裡（全部是 mult = INT32_MAX），差值可達 ±179。
//
// 這是個刻意的設計決定，不是疏漏：真實的硬體再量化單元會選飽和而非環繞，
// 但期中的驗收條件是與軟體參考位元相同，所以這裡照軟體的行為。
// 兩者的差異範圍可以精確界定 —— 需要再量化結果接近 INT32_MAX，而
// multiplier/shift 的整個設計目的就是讓結果落在 int8 附近，
// 所以真實模型碰不到。（t_xqrq 的 OFFWRAP1..3 守住這個行為）
//
// 用 unsigned 相加取得定義明確的環繞語意。
int32_t rq_out = (int32_t)((uint32_t)rq_v + (uint32_t)rq_off);
if (rq_out < rq_lo) rq_out = rq_lo;
if (rq_out > rq_hi) rq_out = rq_hi;

WRITE_RD(sext_xlen((int64_t)rq_out));