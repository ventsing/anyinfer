// B-02 / B-03 的验证：两个比对器的**语义边界**，尤其是容易被"放宽"的地方。
//
// 这类代码的风险不是算错，而是**判据被悄悄放宽**（比如允许 +0/-0 相等、
// 或把 NaN 当成"反正是 NaN 就算相等"）。放宽一次，之后所有门都不作数。

#include "bit_exact.h"
#include "fp64_reference.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

bool contains(const std::string &haystack, const char *needle)
{
    return haystack.find(needle) != std::string::npos;
}

float from_bits(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ------------------------------------------------------------------ L1
void bit_exact_semantics()
{
    using namespace anyinfer::harness;

    const float expected[4] = {1.0f, -2.5f, 0.0f, 1e-30f};
    const float same[4] = {1.0f, -2.5f, 0.0f, 1e-30f};
    const BitReport equal = compare_bits(expected, same, 4);
    check(equal.gate == Gate::Pass, "identical buffers pass");
    check(equal.bad_count == 0, "identical buffers have no mismatches");

    const float one_off[4] = {1.0f, -2.5f, 0.0f, -1e-30f};
    const BitReport mismatched = compare_bits(expected, one_off, 4);
    check(mismatched.gate == Gate::BitMismatch, "one differing element is BitMismatch");
    check(mismatched.bad_count == 1, "bad_count counts differing elements");
    check(mismatched.first_bad == 3, "first_bad points at the differing index");

    // 语义边界 1：+0.0 与 -0.0 数值相等但位模式不同 -> 必须判不同
    const float positive_zero[1] = {0.0f};
    const float negative_zero[1] = {-0.0f};
    check(compare_bits(positive_zero, negative_zero, 1).gate == Gate::BitMismatch,
          "+0.0 vs -0.0 must be a mismatch (bit-exact, not value-equal)");

    // 语义边界 2：不同负载的 NaN -> 必须判不同
    const float nan_a[1] = {std::numeric_limits<float>::quiet_NaN()};
    const float nan_b[1] = {std::numeric_limits<float>::quiet_NaN()};
    const float other_nan[1] = {from_bits(float_bits(nan_b[0]) ^ 0x1u)};
    check(float_bits(nan_b[0]) != float_bits(other_nan[0]),
          "NaN payloads differ by construction");
    check(compare_bits(nan_a, other_nan, 1).gate == Gate::BitMismatch,
          "different NaN payloads must be a mismatch");
    // 同一 NaN 位模式则应通过：位级门看的是位模式
    check(compare_bits(nan_a, nan_b, 1).gate == Gate::Pass,
          "identical NaN bit patterns pass");

    // 语义边界 3：期望有限、实际 Inf -> 归为 NonFinite（便于定位写越界）
    const float infinite[4] = {1.0f, -2.5f, 0.0f, std::numeric_limits<float>::infinity()};
    check(compare_bits(expected, infinite, 4).gate == Gate::NonFinite,
          "finite expected vs Inf actual is NonFinite");

    // 空区间必须通过（0 个元素没有可失配的东西）
    check(compare_bits(expected, same, 0).gate == Gate::Pass, "empty range passes");
    // 空指针是调用方 bug，不得判通过
    check(compare_bits(nullptr, same, 4).gate == Gate::LengthMismatch,
          "null pointer is LengthMismatch, not pass");

    const std::string json = bit_report_json("hc_post_l1", equal);
    check(contains(json, "\"name\":\"hc_post_l1\""), "bit json has name");
    check(contains(json, "\"gate\":\"pass\""), "bit json has gate");
    check(contains(json, "\"bad_count\":0"), "bit json has bad_count");
    check(json.front() == '{' && json.back() == '}', "bit json is an object");
}

// ------------------------------------------------------------------ L2
void fp64_reference_semantics()
{
    using namespace anyinfer::harness;

    const double reference[4] = {1.0, -2.0, 0.0, 1e6};
    const float exact[4] = {1.0f, -2.0f, 0.0f, 1e6f};

    const ReferenceReport ok = compare_reference(reference, exact, 4, Tolerance{1e-6, 1e-6});
    check(ok.gate == Gate::Pass, "identical values pass tolerance gate");
    check(ok.violations == 0, "no violations on identical values");

    // atol 兜底：ref 为 0 处的绝对偏差由 atol 承担
    const float near_zero[4] = {1.0f, -2.0f, 1e-9f, 1e6f};
    check(compare_reference(reference, near_zero, 4, Tolerance{1e-6, 0.0}).gate
              == Gate::Pass,
          "atol covers deviation near a zero reference");

    // 超出容差：1e6 上 100 的绝对误差
    const float off[4] = {1.0f, -2.0f, 0.0f, 1.0e6f + 100.0f};
    const ReferenceReport exceeded =
        compare_reference(reference, off, 4, Tolerance{1e-6, 1e-6});
    check(exceeded.gate == Gate::ToleranceExceeded, "large deviation is ToleranceExceeded");
    check(exceeded.violations == 1, "violations counts exceeding elements");
    check(exceeded.first_violation == 3, "first_violation points at index 3");
    check(exceeded.max_abs > 99.0, "max_abs captures the deviation magnitude");

    // rtol 尺度检查：rtol=1e-7 时 1e6 上允许 ~0.1，故 0.5 必须违规
    const float big_off[4] = {1.0f, -2.0f, 0.0f, 1.0e6f + 0.5f};
    check(compare_reference(reference, big_off, 4, Tolerance{1e-6, 1e-7}).gate
              == Gate::ToleranceExceeded,
          "rtol 1e-7 makes 0.5 absolute error on 1e6 a violation");
    // 同一偏差在 rtol=1e-6（允许 ~1.0）下应当通过 —— 证明容差确实被使用
    check(compare_reference(reference, big_off, 4, Tolerance{1e-6, 1e-6}).gate
              == Gate::Pass,
          "rtol 1e-6 tolerates 0.5 absolute error on 1e6");

    // 参考值非法 -> 归因到参考实现，而不是被测件
    const double bad_reference[4] = {1.0, std::nan(""), 0.0, 1e6};
    check(compare_reference(bad_reference, exact, 4, Tolerance{1e-6, 1e-6}).gate
              == Gate::InvalidReference,
          "non-finite reference is InvalidReference");

    // 被测件 NaN -> NonFinite，且优先于 ToleranceExceeded
    const float nan_actual[4] = {1.0f, -2.0f, 0.0f,
                                 std::numeric_limits<float>::quiet_NaN()};
    check(compare_reference(reference, nan_actual, 4, Tolerance{1e-6, 1e-6}).gate
              == Gate::NonFinite,
          "NaN actual is NonFinite");

    // 零容差即"按位"的确定性自检：任何差异都必须被抓住
    const float tiny[4] = {1.0f, -2.0f, 0.0f, 1e6f + 1.0f};
    check(compare_reference(reference, tiny, 4, bitwise_tolerance()).gate
              == Gate::ToleranceExceeded,
          "zero tolerance rejects any difference");

    // 空指针
    check(compare_reference(nullptr, exact, 4, Tolerance{0.0, 0.0}).gate
              == Gate::LengthMismatch,
          "null reference is LengthMismatch");

    const std::string json =
        reference_report_json("gemm_l2", exceeded, Tolerance{1e-6, 1e-6});
    check(contains(json, "\"gate\":\"tolerance_exceeded\""), "ref json has gate");
    check(contains(json, "\"atol\":"), "ref json records atol");
    check(contains(json, "\"violations\":1"), "ref json records violations");
    check(json.front() == '{' && json.back() == '}', "ref json is an object");
}

} // namespace

int main()
{
    bit_exact_semantics();
    fp64_reference_semantics();

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("bit_exact.h + fp64_reference.h: all checks passed\n");
    return 0;
}
