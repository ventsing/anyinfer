#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "report.h"

// B-03：L2「FP64 参考 + 容差」比对。
//
// 用于 GEMM / attention / MoE 等无法构造精确输入的算子 —— 对这些算子，
// 位级相等**在数学上就不成立**（归约阶数、tile 划分都会改变舍入），
// 所以第一道门只能是「与高精度参考的相对误差」。
//
// 判据（逐元素）：
//     |actual - ref| <= atol + rtol * |ref|
//
// 为什么不是单纯的相对误差：ref 接近 0 时相对误差会爆炸，
// 这是所有量化/归一化算子的常见处（rmsnorm 的输出、残差的小分量）。
// atol 承担这个地板，rtol 承担大值区的比例要求，两者缺一不可。

namespace anyinfer::harness {

struct Tolerance {
    double atol = 0.0;
    double rtol = 0.0;
};

// 位级近似的 L2 退化形式：atol = rtol = 0。用于「同代码同输入复算」的确定性自检。
inline Tolerance bitwise_tolerance()
{
    return Tolerance{0.0, 0.0};
}

struct ReferenceReport {
    Gate gate = Gate::Pass;
    uint64_t count = 0;
    uint64_t violations = 0;
    uint64_t first_violation = 0;
    double max_abs = 0.0;
    uint64_t max_abs_index = 0;
    double max_rel = 0.0;
    uint64_t max_rel_index = 0;
    // 最严重那个元素的原值，便于直接复现
    double worst_ref = 0.0;
    double worst_actual = 0.0;
};

inline ReferenceReport compare_reference(const double *ref, const float *actual,
                                         size_t count, Tolerance tolerance)
{
    ReferenceReport report;
    if (ref == nullptr || actual == nullptr) {
        report.gate = Gate::LengthMismatch;
        return report;
    }
    report.count = count;

    bool invalid_reference = false;
    bool non_finite_actual = false;

    for (size_t i = 0; i < count; ++i) {
        const double reference = ref[i];
        const double observed = static_cast<double>(actual[i]);

        if (!std::isfinite(reference)) {
            invalid_reference = true;
            continue;
        }
        if (!std::isfinite(observed)) {
            non_finite_actual = true;
            if (report.violations == 0) {
                report.first_violation = i;
                report.worst_ref = reference;
                report.worst_actual = observed;
            }
            ++report.violations;
            continue;
        }

        const double diff = std::fabs(observed - reference);
        const double allowed = tolerance.atol + tolerance.rtol * std::fabs(reference);

        if (diff > report.max_abs) {
            report.max_abs = diff;
            report.max_abs_index = i;
        }
        // ref 为 0 时相对误差无定义，跳过统计（不参与判据，判据由 atol 兜底）。
        if (reference != 0.0) {
            const double rel = diff / std::fabs(reference);
            if (rel > report.max_rel) {
                report.max_rel = rel;
                report.max_rel_index = i;
            }
        }
        if (diff > allowed) {
            if (report.violations == 0) {
                report.first_violation = i;
                report.worst_ref = reference;
                report.worst_actual = observed;
            }
            ++report.violations;
        }
    }

    // 归因顺序：参考实现非法 > 被测件非有限 > 超容差。
    if (invalid_reference) {
        report.gate = Gate::InvalidReference;
    } else if (non_finite_actual) {
        report.gate = Gate::NonFinite;
    } else if (report.violations != 0) {
        report.gate = Gate::ToleranceExceeded;
    } else {
        report.gate = Gate::Pass;
    }
    return report;
}

inline std::string reference_report_json(const std::string &name,
                                         const ReferenceReport &report,
                                         Tolerance tolerance)
{
    std::string body;
    json_kv(body, "name", name);
    json_kv(body, "gate", gate_name(report.gate));
    json_kv(body, "count", report.count);
    json_kv(body, "violations", report.violations);
    json_kv(body, "first_violation", report.first_violation);
    json_kv(body, "atol", tolerance.atol);
    json_kv(body, "rtol", tolerance.rtol);
    json_kv(body, "max_abs", report.max_abs);
    json_kv(body, "max_abs_index", report.max_abs_index);
    json_kv(body, "max_rel", report.max_rel);
    json_kv(body, "max_rel_index", report.max_rel_index);
    json_kv(body, "worst_ref", report.worst_ref);
    json_kv(body, "worst_actual", report.worst_actual);
    return json_close(body);
}

} // namespace anyinfer::harness
