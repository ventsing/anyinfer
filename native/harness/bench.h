#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "report.h"

// B-04：微基准框架（配对 A/B + 分位数）。
//
// 三条测量纪律，每条都对应一种常见的"测出假结论"：
//
//  1. **分组计时**：一个样本测 I 次迭代的总时长再除以 I，而不是每次迭代取一次时钟。
//     单次取时钟的开销（几十 ns）在微秒级算子上是百分之几的系统性偏差。
//
//  2. **配对而非独立**：A 与 B 在**同一样本内**背靠背测，比较的是逐样本差值。
//     GPU/NPU 的频率漂移、邻居租户、温度爬升是慢变量，配对差值能把它们抵掉；
//     两次独立的"中位数比中位数"抵不掉。
//
//  3. **ABBA 交错**：偶数样本先 A 后 B，奇数样本先 B 后 A。
//     这样 A 与 B 在时间轴上的平均位置相同，残余的线性漂移不会偏向某一方。
//
// 以及一条诚实性约束：**样本数只有 7 时，"p99" 就是最大值**，
// 报它只会制造虚假的确定性。本文件用 nearest-rank 且显式标记退化，
// 并用**精确符号检验**代替小样本的百分位比较。

namespace anyinfer::bench {

// 与原型 native/acl_backend/bench.cpp 保持一致：warmup 5 / samples 7 / iterations 8。
// 5 次预热足以让频率与缓存进入稳态；7 个配对样本正好让"7:0 全胜"达到 p≈0.016。
inline constexpr uint32_t kWarmup = 5;
inline constexpr uint32_t kSamples = 7;
inline constexpr uint32_t kIterations = 8;

struct Protocol {
    uint32_t warmup = kWarmup;
    uint32_t samples = kSamples;
    uint32_t iterations = kIterations;

    // 返回空字符串表示合法；否则是错误原因。
    std::string validate() const
    {
        if (samples < 3) {
            return "samples must be >= 3 (fewer cannot support any decision)";
        }
        if (iterations == 0) {
            return "iterations must be >= 1";
        }
        if (warmup == 0) {
            return "warmup must be >= 1 (cold frequency makes the first samples junk)";
        }
        if (samples > 255) {
            return "samples must be <= 255";
        }
        return std::string();
    }
};

using Clock = std::function<uint64_t()>; // 纳秒
using Sync = std::function<void()>;      // 设备计时必须提供：读时钟前要等设备真正完成

// ---------------------------------------------------------------- 统计原语
inline std::vector<double> sorted_copy(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values;
}

inline double median_of_sorted(const std::vector<double> &sorted)
{
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t n = sorted.size();
    if (n % 2 == 1) {
        return sorted[n / 2];
    }
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

// nearest-rank：返回第 ceil(p*n) 小的值（p ∈ (0,1]）
inline double percentile_nearest_rank(const std::vector<double> &sorted, double p)
{
    if (sorted.empty()) {
        return 0.0;
    }
    if (p <= 0.0) {
        return sorted.front();
    }
    if (p >= 1.0) {
        return sorted.back();
    }
    const size_t n = sorted.size();
    size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
    if (rank == 0) {
        rank = 1;
    }
    if (rank > n) {
        rank = n;
    }
    return sorted[rank - 1];
}

/* 该百分位在样本数为 n 时是否退化（等于最大值）。
 * n=7 时 p95 与 p99 都落在 rank 7 = 最大值 —— 报出来就是假的确定性。 */
inline bool percentile_is_degenerate(size_t n, double p)
{
    if (n == 0) {
        return true;
    }
    const size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
    return rank >= n;
}

/* 精确双尾符号检验。high/low 是"谁更快"的胜场数（平局不计入 n）。
 * 用于小样本：n=7 且 7:0 时 p = 2/128 = 0.015625。 */
inline double sign_test_p(uint32_t high, uint32_t low)
{
    const uint32_t n = high + low;
    if (n == 0) {
        return 1.0;
    }
    const uint32_t threshold = high > low ? high : low;
    // 从 k=n 往下累加 C(n,k)/2^n，用递推避免大中间量
    double term = std::ldexp(1.0, -static_cast<int>(n)); // C(n,n)/2^n
    double tail = term;
    for (uint32_t k = n; k > threshold; --k) {
        term *= static_cast<double>(k) / static_cast<double>(n - k + 1);
        tail += term;
    }
    const double p = 2.0 * tail;
    return p > 1.0 ? 1.0 : p;
}

struct ArmStats {
    std::string name;
    std::vector<double> per_iter_ns; // 每个样本的"每迭代纳秒"
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double p95 = 0.0;
    bool p95_degenerate = true;
    double mean = 0.0;
};

inline ArmStats summarize(const std::string &name, std::vector<double> samples)
{
    ArmStats stats;
    stats.name = name;
    stats.per_iter_ns = samples;
    const std::vector<double> sorted = sorted_copy(samples);
    stats.median = median_of_sorted(sorted);
    stats.minimum = sorted.empty() ? 0.0 : sorted.front();
    stats.maximum = sorted.empty() ? 0.0 : sorted.back();
    stats.p95 = percentile_nearest_rank(sorted, 0.95);
    stats.p95_degenerate = percentile_is_degenerate(sorted.size(), 0.95);
    double total = 0.0;
    for (double value : samples) {
        total += value;
    }
    stats.mean = samples.empty() ? 0.0 : total / static_cast<double>(samples.size());
    return stats;
}

struct PairedResult {
    Protocol protocol;
    ArmStats a;
    ArmStats b;
    std::vector<double> deltas_ns; // 逐样本 b - a
    double median_delta_ns = 0.0;
    uint32_t a_wins = 0; // b 更慢的样本数
    uint32_t b_wins = 0; // b 更快的样本数
    uint32_t ties = 0;
    double sign_test_p = 1.0;
    double speedup = 1.0; // a.median / b.median，>1 表示 B 更快
    std::string verdict;  // b_faster / a_faster / indistinguishable
};

/* 跑一次配对测量。
 *
 * run_a / run_b：执行**一次迭代**。用模板而非 std::function，
 *   让测试里的 lambda 能被内联（本框架会被用来测亚微秒算子）。
 * sync：可为空；设备计时必须提供，否则读到的是下发时刻而不是完成时刻。
 */
template <typename RunA, typename RunB>
PairedResult run_paired(const Protocol &protocol, const Clock &now_ns, const Sync &sync,
                        const std::string &name_a, RunA &&run_a, const std::string &name_b,
                        RunB &&run_b, double alpha = 0.05)
{
    PairedResult result;
    result.protocol = protocol;

    auto settle = [&sync]() {
        if (sync) {
            sync();
        }
    };

    // 预热：A 与 B 各跑 warmup 次，不计入样本
    for (uint32_t i = 0; i < protocol.warmup; ++i) {
        run_a();
        run_b();
    }
    settle();

    std::vector<double> per_iter_a;
    std::vector<double> per_iter_b;
    per_iter_a.reserve(protocol.samples);
    per_iter_b.reserve(protocol.samples);

    auto measure = [&](auto &&run) -> double {
        const uint64_t start = now_ns();
        for (uint32_t i = 0; i < protocol.iterations; ++i) {
            run();
        }
        if (sync) {
            sync();
        }
        const uint64_t stop = now_ns();
        const double elapsed = static_cast<double>(stop - start);
        return elapsed / static_cast<double>(protocol.iterations);
    };

    for (uint32_t sample = 0; sample < protocol.samples; ++sample) {
        // ABBA：偶数样本 A 先，奇数样本 B 先
        double a_ns;
        double b_ns;
        if (sample % 2 == 0) {
            a_ns = measure(run_a);
            b_ns = measure(run_b);
        } else {
            b_ns = measure(run_b);
            a_ns = measure(run_a);
        }
        per_iter_a.push_back(a_ns);
        per_iter_b.push_back(b_ns);
        result.deltas_ns.push_back(b_ns - a_ns);
        if (b_ns < a_ns) {
            ++result.b_wins;
        } else if (b_ns > a_ns) {
            ++result.a_wins;
        } else {
            ++result.ties;
        }
    }

    result.a = summarize(name_a, per_iter_a);
    result.b = summarize(name_b, per_iter_b);
    result.median_delta_ns = median_of_sorted(sorted_copy(result.deltas_ns));
    result.sign_test_p = sign_test_p(result.b_wins, result.a_wins);
    result.speedup = result.b.median > 0.0 ? result.a.median / result.b.median : 0.0;

    // 只有符号检验显著时才给方向性结论；否则一律"不可区分"。
    // 这是刻意保守的：微基准的噪声足以制造 5% 的假提升。
    if (result.sign_test_p <= alpha && result.b_wins > result.a_wins) {
        result.verdict = "b_faster";
    } else if (result.sign_test_p <= alpha && result.a_wins > result.b_wins) {
        result.verdict = "a_faster";
    } else {
        result.verdict = "indistinguishable";
    }
    return result;
}

inline std::string paired_report_json(const std::string &name, const PairedResult &result)
{
    std::string body;
    harness::json_kv(body, "name", name);
    harness::json_kv(body, "verdict", result.verdict);
    harness::json_kv(body, "warmup", uint64_t(result.protocol.warmup));
    harness::json_kv(body, "samples", uint64_t(result.protocol.samples));
    harness::json_kv(body, "iterations", uint64_t(result.protocol.iterations));

    harness::json_kv(body, "a_name", result.a.name);
    harness::json_kv(body, "a_median_ns", result.a.median);
    harness::json_kv(body, "b_name", result.b.name);
    harness::json_kv(body, "b_median_ns", result.b.median);

    harness::json_kv(body, "median_delta_ns", result.median_delta_ns);
    harness::json_kv(body, "speedup", result.speedup);
    harness::json_kv(body, "a_wins", uint64_t(result.a_wins));
    harness::json_kv(body, "b_wins", uint64_t(result.b_wins));
    harness::json_kv(body, "ties", uint64_t(result.ties));
    harness::json_kv(body, "sign_test_p", result.sign_test_p);
    // 显式记录百分位是否退化，避免下游把 p95 当成分位数
    harness::json_kv(body, "p95_degenerate", result.a.p95_degenerate ? std::string("true")
                                                            : std::string("false"));
    return harness::json_close(body);
}

} // namespace anyinfer::bench
