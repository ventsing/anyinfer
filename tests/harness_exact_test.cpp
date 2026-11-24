// B-01 的验证：精确性判据本身必须被验证，否则整个 L1 门都建在假设上。
//
// 本测试刻意包含**两个对照臂**：
//   正向：满足判据的输入 -> 两种归约顺序位级相等
//   负向：违反判据的输入 -> 两种归约顺序位级不等（证明判据不是摆设）

#include "exact.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

uint32_t bits_of(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// 正向对照：判据成立时，求和与顺序无关。
void forward_control()
{
    constexpr uint32_t kCount = 1024;
    constexpr int32_t kShift = 20;
    constexpr int32_t kBound = 512;

    // 判据自检：1024 * 512 = 2^19 <= 2^24-1，应当精确
    check(anyinfer::exact::sum_is_exact(kCount, kBound),
          "sum_is_exact should hold for 1024 terms bounded by 512");

    std::vector<float> values(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        values[i] = anyinfer::exact::to_float(
            anyinfer::exact::generate(0xA5A5u, i, kShift, kBound));
    }

    float forward = 0.0f;
    for (uint32_t i = 0; i < kCount; ++i) {
        forward += values[i];
    }
    float backward = 0.0f;
    for (uint32_t i = kCount; i-- > 0;) {
        backward += values[i];
    }

    check(bits_of(forward) == bits_of(backward),
          "exact inputs must sum identically in both orders");

    // 结果必须仍落在 2^-kShift 网格上，否则说明生成了非 dyadic 值
    const double scaled = static_cast<double>(forward) * std::ldexp(1.0, kShift);
    check(scaled == std::floor(scaled), "sum should lie on the 2^-shift grid");
}

// 负向对照：违反判据时，求和确实依赖顺序 —— 证明判据有判别力。
void negative_control()
{
    // 2^24 处 ulp 已是 2，故 +1 会被舍入掉
    volatile float big = 16777216.0f; // 2^24
    volatile float one = 1.0f;

    volatile float left = (big + one) + one;  // 2^24+1 舍入回 2^24，再加 1 仍为 2^24
    volatile float right = big + (one + one); // 1+1=2，2^24+2 精确

    check(bits_of(left) != bits_of(right),
          "inexact inputs must demonstrate order dependence");
    check(left == 16777216.0f && right == 16777218.0f,
          "inexact control expects 2^24 and 2^24 + 2");

    check(!anyinfer::exact::sum_is_exact(3, int64_t(16777216)),
          "sum_is_exact must reject magnitude 2^24");
    // 单项时 2^24-1 恰好用满 24 位有效位，是精确的；两项就开始溢出
    check(anyinfer::exact::sum_is_exact(1, int64_t(16777215)),
          "single term of magnitude 2^24 - 1 is exact");
    check(!anyinfer::exact::sum_is_exact(2, int64_t(16777215)),
          "two terms of magnitude 2^24 - 1 overflow the significand");
    check(!anyinfer::exact::sum_is_exact(3, int64_t(16777215)),
          "three terms of magnitude 2^24 - 1 overflow the significand");
}

void representability()
{
    using anyinfer::exact::Dyadic;
    using anyinfer::exact::representable;
    using anyinfer::exact::to_float;

    check(representable(Dyadic{1, 0}), "1 * 2^0 representable");
    check(representable(Dyadic{16777215, 0}), "2^24-1 representable");
    check(!representable(Dyadic{16777216, 0}), "2^24 not representable");
    check(!representable(Dyadic{1, -1}), "negative shift rejected");
    check(!representable(Dyadic{1, 127}), "shift beyond 126 rejected (subnormal trap)");

    check(to_float(Dyadic{3, 2}) == 0.75f, "3 * 2^-2 == 0.75");
    check(to_float(Dyadic{-5, 1}) == -2.5f, "-5 * 2^-1 == -2.5");
    check(to_float(Dyadic{0, 30}) == 0.0f, "zero stays zero");
    check(to_float(Dyadic{1, 20}) == std::ldexp(1.0f, -20), "2^-20 exact");
}

void product_and_dot()
{
    using anyinfer::exact::dot_is_exact;
    using anyinfer::exact::product_is_exact;

    check(!product_is_exact(4096, 4096), "4096*4096 == 2^24 is not exact");
    check(product_is_exact(4096, 4095), "4096*4095 < 2^24 is exact");
    check(product_is_exact(0, 12345), "zero product is always exact");
    check(product_is_exact(-4096, 4095), "sign must not affect exactness");

    // 64 项、每项部分积 2^18 -> 恰好用尽 2^24 容量，故不精确
    check(!dot_is_exact(64, 1 << 18), "64 * 2^18 == 2^24 is not exact");
    check(dot_is_exact(63, 1 << 18), "63 * 2^18 < 2^24 is exact");
    check(dot_is_exact(0, 1 << 30), "empty dot is trivially exact");
}

void determinism()
{
    using anyinfer::exact::generate;

    const auto a = generate(1234u, 5678u, 16, 100);
    const auto b = generate(1234u, 5678u, 16, 100);
    check(a.k == b.k && a.shift == b.shift, "generate is deterministic");

    bool within_bound = true;
    bool varied = false;
    const int32_t first = generate(7u, 0u, 16, 32).k;
    for (uint32_t i = 0; i < 4096; ++i) {
        const auto d = generate(7u, i, 16, 32);
        if (d.k < -32 || d.k > 32) {
            within_bound = false;
        }
        if (d.k != first) {
            varied = true;
        }
    }
    check(within_bound, "generate respects |k| <= bound");
    check(varied, "generate produces varied values");

    check(generate(1u, 1u, 8, 0).k == 0, "bound 0 yields zero");
    check(generate(1u, 1u, 8, -5).k == 0, "negative bound yields zero");
}

} // namespace

int main()
{
    forward_control();
    negative_control();
    representability();
    product_and_dot();
    determinism();

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("exact.h: all checks passed (forward + negative controls)\n");
    return 0;
}
