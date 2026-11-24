// B-04 的验证：微基准框架本身必须被测，否则所有"这次比上次快 3%"的结论都建在假设上。
//
// 关键手法：**注入假时钟**。计时是唯一让微基准无法在无卡机器上测试的部分，
// 把时钟做成可注入的之后，整个协议（warmup / 分组除算 / ABBA 交错 / 配对差值 /
// 符号检验）都变成完全确定的算术，可以在 host 上逐位断言。
//
// 两个必须成立的负向性质：
//   1. 噪声导致的 4:3 微弱优势**不许**被判为改进（否则 RSI 会追逐噪声）
//   2. 完全相同的两臂必须判"不可区分"，而不是靠浮点相等侥幸通过

#include "bench.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace anyinfer::bench;

int failures = 0;

void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

void check_near(double actual, double expected, double tolerance, const char *what)
{
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::printf("FAIL %s: %.17g != %.17g (tol %.3g)\n", what, actual, expected,
                    tolerance);
        ++failures;
    }
}

/* 假时钟：与真实的 steady_clock 只差一个实现细节 —— 价格由被测代码自己推动。
 * 这样"每迭代多少纳秒"变成可精确预期的数，测试就能断言算术而不是断言噪声。 */
struct FakeTime {
    uint64_t now_ns = 1'000'000'000; // 起点非零：能抓住"忘了相减"这类错误
    uint64_t calls = 0;

    Clock clock()
    {
        return [this]() {
            ++calls;
            return now_ns;
        };
    }
};

struct Arm {
    FakeTime *time;
    uint64_t ns_per_iter;
    uint64_t *call_counter;
    std::vector<char> *log;
    char tag;

    void operator()() const
    {
        time->now_ns += ns_per_iter;
        if (call_counter != nullptr) {
            ++*call_counter;
        }
        if (log != nullptr) {
            log->push_back(tag);
        }
    }
};

Protocol small_protocol()
{
    Protocol protocol;
    protocol.warmup = 2;
    protocol.samples = 7;
    protocol.iterations = 4;
    return protocol;
}

// ---------------------------------------------------------------- 协议校验
void protocol_validation()
{
    Protocol defaults;
    check(defaults.validate().empty(), "default protocol is valid");
    check(defaults.warmup == 5 && defaults.samples == 7 && defaults.iterations == 8,
          "defaults are warmup 5 / samples 7 / iterations 8");

    Protocol too_few;
    too_few.samples = 2;
    check(!too_few.validate().empty(), "samples < 3 is rejected");

    Protocol no_iterations;
    no_iterations.iterations = 0;
    check(!no_iterations.validate().empty(), "iterations = 0 is rejected");

    Protocol no_warmup;
    no_warmup.warmup = 0;
    check(!no_warmup.validate().empty(), "warmup = 0 is rejected");
}

// ---------------------------------------------------------------- 统计原语
void median_and_percentile()
{
    check_near(median_of_sorted({1.0, 2.0, 3.0}), 2.0, 0.0, "median odd");
    check_near(median_of_sorted({1.0, 2.0, 3.0, 4.0}), 2.5, 0.0, "median even");
    check_near(median_of_sorted({}), 0.0, 0.0, "median empty");

    const std::vector<double> sorted = {10, 20, 30, 40, 50, 60, 70};
    // nearest-rank：rank = ceil(p*n)
    check_near(percentile_nearest_rank(sorted, 0.95), 70.0, 0.0, "p95 n=7 is max");
    check_near(percentile_nearest_rank(sorted, 0.50), 40.0, 0.0, "p50 n=7");
    check_near(percentile_nearest_rank(sorted, 1.00), 70.0, 0.0, "p100 is max");
    check_near(percentile_nearest_rank(sorted, 0.0), 10.0, 0.0, "p0 is min");

    // 诚实性：n=7 时 p95/p99 退化成最大值，必须被标记出来
    check(percentile_is_degenerate(7, 0.95), "p95 degenerate at n=7");
    check(percentile_is_degenerate(7, 0.99), "p99 degenerate at n=7");
    check(!percentile_is_degenerate(100, 0.95), "p95 meaningful at n=100");
    check(!percentile_is_degenerate(100, 0.99), "p99 meaningful at n=100");
    check(percentile_is_degenerate(99, 0.99), "p99 degenerate at n=99");
}

void sign_test_table()
{
    // n=7 且 7:0 全胜 -> 2/128，这是本协议默认样本数的"可判定门槛"
    check_near(sign_test_p(7, 0), 0.015625, 0.0, "sign test 7:0");
    check_near(sign_test_p(0, 7), 0.015625, 0.0, "sign test 0:7 symmetric");
    // 6:1 -> 2*8/128
    check_near(sign_test_p(6, 1), 0.125, 0.0, "sign test 6:1");
    // 5:2 -> 2*29/128
    check_near(sign_test_p(5, 2), 0.453125, 0.0, "sign test 5:2");
    // 4:3 -> 尾部已经是 1/2，双尾截断到 1
    check_near(sign_test_p(4, 3), 1.0, 0.0, "sign test 4:3 collapses to 1");
    check_near(sign_test_p(0, 0), 1.0, 0.0, "sign test no samples");

    // 单调性：胜得越干净 p 越小，且永不超过 1
    check(sign_test_p(7, 0) < sign_test_p(6, 1), "p decreases with margin");
    check(sign_test_p(6, 1) < sign_test_p(5, 2), "p decreases with margin");
    check(sign_test_p(30, 1) <= 1.0 && sign_test_p(30, 1) > 0.0,
          "sign test stays in (0,1] for larger n");
}

// ---------------------------------------------------------------- 分组与除法
void group_timing_divides_by_iterations()
{
    FakeTime time;
    Arm slow{&time, 1000, nullptr, nullptr, 'A'};
    Arm slow_b{&time, 1000, nullptr, nullptr, 'B'};

    Protocol protocol = small_protocol();
    protocol.iterations = 8;
    const PairedResult result =
        run_paired(protocol, time.clock(), Sync(), "a", slow, "b", slow_b);

    // 每迭代 1000 ns，不是每次分组 8000 ns
    check_near(result.a.median, 1000.0, 0.0, "group time is divided by iterations");
    check_near(result.b.median, 1000.0, 0.0, "group time is divided by iterations (b)");
    check(result.a.per_iter_ns.size() == protocol.samples, "one sample per iteration group");
}

void warmup_is_not_counted()
{
    FakeTime time;
    uint64_t calls_a = 0;
    uint64_t calls_b = 0;
    Arm arm_a{&time, 500, &calls_a, nullptr, 'A'};
    Arm arm_b{&time, 500, &calls_b, nullptr, 'B'};

    Protocol protocol = small_protocol(); // warmup 2, samples 7, iterations 4
    const PairedResult result =
        run_paired(protocol, time.clock(), Sync(), "a", arm_a, "b", arm_b);

    // 每臂总调用 = warmup + samples*iterations
    const uint64_t expected = protocol.warmup + protocol.samples * protocol.iterations;
    check(calls_a == expected, "arm a ran warmup + samples*iterations times");
    check(calls_b == expected, "arm b ran warmup + samples*iterations times");
    // 但样本数只等于 samples（warmup 没进统计）
    check(result.a.per_iter_ns.size() == protocol.samples, "warmup excluded from samples");
}

void abba_interleaving()
{
    FakeTime time;
    std::vector<char> log;
    Arm arm_a{&time, 500, nullptr, &log, 'A'};
    Arm arm_b{&time, 500, nullptr, &log, 'B'};

    Protocol protocol = small_protocol(); // warmup 2, samples 7, iterations 4
    run_paired(protocol, time.clock(), Sync(), "a", arm_a, "b", arm_b);

    // 预热每臂只调用 1 次（预热单位是"一次迭代"，不是一个分组）
    const size_t group = protocol.iterations;
    const size_t warmup_len = static_cast<size_t>(protocol.warmup) * 2;
    check(log.size() == warmup_len + static_cast<size_t>(protocol.samples) * 2 * group,
          "log length matches protocol");

    std::string order;
    for (size_t start = warmup_len; start + group <= log.size(); start += group) {
        bool uniform = true;
        for (size_t i = start; i < start + group; ++i) {
            if (log[i] != log[start]) {
                uniform = false;
            }
        }
        check(uniform, "each group is a single arm");
        order.push_back(log[start]);
    }
    check(order == "ABBAABBAABBAAB", "ABBA interleaving across samples");
}

// ---------------------------------------------------------------- 配对裁决
void clear_winner_is_detected()
{
    FakeTime time;
    Arm fast{&time, 1000, nullptr, nullptr, 'A'};
    Arm slow{&time, 1100, nullptr, nullptr, 'B'};

    const PairedResult result =
        run_paired(small_protocol(), time.clock(), Sync(), "fast", fast,
                          "slow", slow);

    check_near(result.a.median, 1000.0, 0.0, "a median");
    check_near(result.b.median, 1100.0, 0.0, "b median");
    check_near(result.median_delta_ns, 100.0, 0.0, "paired median delta is exact");
    check(result.a_wins == 7 && result.b_wins == 0 && result.ties == 0,
          "all 7 samples favour a");
    check_near(result.sign_test_p, 0.015625, 0.0, "sign test p for 7:0");
    check(result.verdict == "a_faster", "verdict names a as faster");
    // speedup = a.median / b.median < 1 表示 B 更慢，方向必须一致
    check_near(result.speedup, 1000.0 / 1100.0, 1e-12, "speedup is a/b median ratio");

    // 反向：B 变快时结论必须翻转，速度比也必须 > 1
    FakeTime time2;
    Arm fast2{&time2, 1000, nullptr, nullptr, 'A'};
    Arm faster{&time2, 900, nullptr, nullptr, 'B'};
    const PairedResult flipped =
        run_paired(small_protocol(), time2.clock(), Sync(), "fast", fast2,
                          "faster", faster);
    check(flipped.verdict == "b_faster", "verdict flips when b is faster");
    check_near(flipped.speedup, 1000.0 / 900.0, 1e-12, "speedup flips above 1");
}

void noise_is_not_an_improvement()
{
    /* 关键负向用例：B 在 7 个样本里只有 4 次略快、3 次略慢。
     * 中位数看起来"更好"，但符号检验 p=1.0 —— 必须判不可区分。
     * 若这里放行，RSI 会开始追噪声，而且每次都能"测出"1~3% 的提升。 */
    FakeTime time;
    Arm arm_a{&time, 1000, nullptr, nullptr, 'A'};
    const Protocol protocol = small_protocol(); // warmup 2, samples 7, iterations 4
    uint32_t call_index = 0;
    auto jitter_run = [&]() {
        uint64_t cost = 1000; // 预热用中性值，不参与统计
        if (call_index >= protocol.warmup) {
            // 按样本序号抖动：偶数样本快、奇数样本慢 -> 4:3
            const uint32_t sample = (call_index - protocol.warmup) / protocol.iterations;
            cost = (sample % 2 == 0) ? 900 : 1100;
        }
        time.now_ns += cost;
        ++call_index;
    };

    const PairedResult result =
        run_paired(protocol, time.clock(), Sync(), "a", arm_a, "jitter", jitter_run);

    check(result.verdict == "indistinguishable", "4:3 split must not be called an improvement");
    check(result.sign_test_p > 0.05, "sign test p above alpha for 4:3");
    check(result.b_wins == 4 && result.a_wins == 3, "wins recorded as 4:3");
}

void identical_arms_are_indistinguishable()
{
    FakeTime time;
    Arm arm_a{&time, 700, nullptr, nullptr, 'A'};
    Arm arm_b{&time, 700, nullptr, nullptr, 'B'};

    const PairedResult result =
        run_paired(small_protocol(), time.clock(), Sync(), "a", arm_a, "b", arm_b);

    check(result.verdict == "indistinguishable", "identical arms are indistinguishable");
    check(result.ties == result.protocol.samples, "identical arms produce only ties");
    check_near(result.median_delta_ns, 0.0, 0.0, "identical arms have zero delta");
    check_near(result.speedup, 1.0, 1e-12, "identical arms have unit speedup");
    check_near(result.sign_test_p, 1.0, 0.0, "all ties means no evidence");
}

void sync_is_invoked_for_device_timing()
{
    FakeTime time;
    Arm arm_a{&time, 300, nullptr, nullptr, 'A'};
    Arm arm_b{&time, 300, nullptr, nullptr, 'B'};
    uint64_t sync_calls = 0;
    Sync sync = [&sync_calls]() { ++sync_calls; };

    Protocol protocol = small_protocol();
    run_paired(protocol, time.clock(), sync, "a", arm_a, "b", arm_b);

    // 预热后 1 次 + 每个样本每臂各 1 次
    const uint64_t expected = 1 + 2ull * protocol.samples;
    check(sync_calls == expected, "sync invoked once after warmup and before each stop");
}

void report_is_machine_readable()
{
    FakeTime time;
    Arm fast{&time, 1000, nullptr, nullptr, 'A'};
    Arm slow{&time, 1200, nullptr, nullptr, 'B'};
    const PairedResult result =
        run_paired(small_protocol(), time.clock(), Sync(), "fast", fast,
                          "slow", slow);
    const std::string json = paired_report_json("hc_post", result);

    check(!json.empty() && json.front() == '{' && json.back() == '}', "json is fenced");
    check(json.find("\"verdict\":\"a_faster\"") != std::string::npos, "json carries verdict");
    check(json.find("\"sign_test_p\":") != std::string::npos, "json carries sign test");
    check(json.find("\"p95_degenerate\":\"true\"") != std::string::npos,
          "json flags degenerate percentiles");
    // 报告里不能出现平均值冒充分位数：median 必须单独出现
    check(json.find("\"a_median_ns\":") != std::string::npos, "json carries median");
}

} // namespace

int main()
{
    protocol_validation();
    median_and_percentile();
    sign_test_table();

    group_timing_divides_by_iterations();
    warmup_is_not_counted();
    abba_interleaving();

    clear_winner_is_detected();
    noise_is_not_an_improvement();
    identical_arms_are_indistinguishable();
    sync_is_invoked_for_device_timing();
    report_is_machine_readable();

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("bench.h: all checks passed (injected clock, ABBA paired, sign test)\n");
    return 0;
}
