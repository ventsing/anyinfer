#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "report.h"

// B-02：L1 位级比对。
//
// 语义严格是「位模式相同」，不是「数值相等」：
//   +0.0f 与 -0.0f   -> 不同（Mismatch）
//   两个 NaN 的负载不同 -> 不同（Mismatch）
// 这是刻意的。RL 要求的确定性是**逐位复现**，不是「差不多」；
// 一旦允许 +0/-0 或 NaN 负载的差异通过，未来某个真实的确定性问题就会藏在里面。

namespace anyinfer::harness {

struct BitReport {
    Gate gate = Gate::Pass;
    uint64_t count = 0;
    uint64_t bad_count = 0;
    uint64_t first_bad = 0;
    uint32_t expected_bits = 0;
    uint32_t actual_bits = 0;
};

inline uint32_t float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline bool finite_bits(uint32_t bits)
{
    return ((bits >> 23) & 0xFFu) != 0xFFu;
}

inline BitReport compare_bits(const uint32_t *expected, const uint32_t *actual,
                              size_t count)
{
    BitReport report;
    if (expected == nullptr || actual == nullptr) {
        report.gate = Gate::LengthMismatch;
        return report;
    }
    report.count = count;
    for (size_t i = 0; i < count; ++i) {
        if (expected[i] == actual[i]) {
            continue;
        }
        if (report.bad_count == 0) {
            report.first_bad = i;
            report.expected_bits = expected[i];
            report.actual_bits = actual[i];
        }
        ++report.bad_count;
    }
    if (report.bad_count == 0) {
        return report;
    }
    // 期望有限而实际非有限，几乎总意味着写越界/未初始化，独立归类便于定位。
    report.gate = (finite_bits(report.expected_bits) && !finite_bits(report.actual_bits))
                      ? Gate::NonFinite
                      : Gate::BitMismatch;
    return report;
}

inline BitReport compare_bits(const float *expected, const float *actual, size_t count)
{
    return compare_bits(reinterpret_cast<const uint32_t *>(expected),
                        reinterpret_cast<const uint32_t *>(actual), count);
}

inline std::string bit_report_json(const std::string &name, const BitReport &report)
{
    std::string body;
    json_kv(body, "name", name);
    json_kv(body, "gate", gate_name(report.gate));
    json_kv(body, "count", report.count);
    json_kv(body, "bad_count", report.bad_count);
    json_kv(body, "first_bad", report.first_bad);
    json_kv(body, "expected_bits", uint64_t(report.expected_bits));
    json_kv(body, "actual_bits", uint64_t(report.actual_bits));
    return json_close(body);
}

} // namespace anyinfer::harness
