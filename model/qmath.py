#!/usr/bin/env python3
"""
qmath.py — 定點量化數學的唯一真實來源。

convert / reference 都從這裡 import，C kernel 與 Spike 的 xqrq 指令
也必須逐位元符合這裡的語意。改這個檔 = 改規格。

支援 int8 (qmin=-128, qmax=127) 與 uint8 (qmin=0, qmax=255) 兩種輸出域；
核心數學完全相同，只有最後的飽和範圍不同。

全程只用 Python 的任意精度整數。quantize_multiplier 的輸入是浮點，
但那是離線一次性的轉換，不進 RISC-V。
"""

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def quantize_multiplier(m):
    """
    實數 multiplier -> (int32 q, shift)，語意同 TFLite 的 QuantizeMultiplier。
    q 落在 [2^30, 2^31)，shift 為負代表右移：

        real_M  ~=  q * 2^(shift - 31)
    """
    if m == 0.0:
        return 0, 0
    if m < 0.0:
        raise ValueError("multiplier must be non-negative, got %r" % m)
    s = 0
    while m < 0.5:
        m *= 2.0
        s -= 1
    while m >= 1.0:
        m /= 2.0
        s += 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        s += 1
    assert 0 < q <= INT32_MAX, q
    return q, s


def requantize(acc, mult, shift, z_out, qmin=-128, qmax=127):
    """
    int32 accumulator -> 量化輸出。這就是 xqrq 指令要實作的語意。

    採 TFLite 的 single-rounding 路徑 (TFLITE_SINGLE_ROUNDING)：

        total = 31 - shift
        r     = (acc * mult + 2^(total-1)) >> total      # 算術右移
        out   = clamp(r + z_out, qmin, qmax)

    一次 32x32->64 乘法、一次加法、一次算術右移。硬體上只需要
    一顆乘法器、一個加法器、一個可變移位器與一組 clamp。

    uint8 模型傳 qmin=0, qmax=255；int8 模型用預設值。
    """
    assert -31 <= shift <= 30, shift
    total = 31 - shift
    r = (int(acc) * int(mult) + (1 << (total - 1))) >> total
    r = max(INT32_MIN, min(INT32_MAX, r))
    return max(qmin, min(qmax, r + int(z_out)))


# ---------------------------------------------------------------------------
# 以下為 gemmlowp 的舊雙重捨入路徑，保留供報告對照用。
# 實測對 LiteRT 的不符率比 single rounding 高，且硬體成本較大。
# ---------------------------------------------------------------------------

def sat_round_dbl_high_mul(a, b):
    """
    gemmlowp 的 SaturatingRoundingDoublingHighMul(int32, int32) -> int32

    注意：gemmlowp 原文用 C 的整數除法（往零截斷），不是算術右移。
    對負數兩者差 1 —— 這是個容易寫錯的地方。
    """
    assert INT32_MIN <= a <= INT32_MAX, a
    assert INT32_MIN <= b <= INT32_MAX, b
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    t = ab + nudge
    r = (t // (1 << 31)) if t >= 0 else -((-t) // (1 << 31))
    return max(INT32_MIN, min(INT32_MAX, r))


def rounding_div_by_pot(x, exp):
    """gemmlowp 的 RoundingDivideByPOT(x, exp) = round(x / 2^exp)。"""
    assert exp >= 0
    if exp == 0:
        return x
    mask = (1 << exp) - 1
    rem = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    r = x >> exp
    return r + 1 if rem > threshold else r


def requantize_double_rounding(acc, mult, shift, z_out, qmin=-128, qmax=127):
    """舊的 gemmlowp 雙重捨入路徑，保留供對照。"""
    left = shift if shift > 0 else 0
    right = -shift if shift < 0 else 0
    v = acc << left
    assert INT32_MIN <= v <= INT32_MAX, ("overflow before mul", acc, shift)
    v = rounding_div_by_pot(sat_round_dbl_high_mul(v, int(mult)), right)
    return max(qmin, min(qmax, v + int(z_out)))
