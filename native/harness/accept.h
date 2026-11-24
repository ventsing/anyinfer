#pragma once
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "report.h"

// B-04b：算子验收的**裁决策略**。
//
// 为什么要把策略单独抽成一个纯头文件：
//   RSI 闭环里"这次改动能不能晋升"这个判断必须只有**一个**实现。
//   设备驱动、host 测试、跨语言的 Python 入口若各写一遍聚合逻辑，
//   迟早会出现"本地通过、流水线不通过"这类无法归因的分歧。
//   所以策略在这里，其他所有入口都调用它。
//
// 一个刻意的设计：三态而不是布尔。
//   只用 true/false 表示一道门，会让"这道门今天根本没跑"和"这道门通过了"
//   变成同一个值 —— 于是第一个版本的工具就会永远输出 l4_gate: pass，
//   而没有任何人会注意到它其实从未运行过。
//   因此 NotRun 是独立状态，且**阻断晋升**。

namespace anyinfer::accept {

// ---------------------------------------------------------------- 门状态
enum class GateState {
    Pass,
    Fail,
    NotRun, // 未执行（尚未接线 / 缺少设备 / 跳过）—— 绝不等同于通过
};

inline const char *gate_state_name(GateState state)
{
    switch (state) {
    case GateState::Pass:
        return "pass";
    case GateState::Fail:
        return "fail";
    case GateState::NotRun:
        return "not_run";
    }
    return "unknown";
}

inline bool parse_gate_state(const std::string &text, GateState &out)
{
    if (text == "pass") {
        out = GateState::Pass;
        return true;
    }
    if (text == "fail") {
        out = GateState::Fail;
        return true;
    }
    if (text == "not_run") {
        out = GateState::NotRun;
        return true;
    }
    return false;
}

struct GateOutcome {
    GateState state = GateState::NotRun;
    std::string reason; // 失败/未跑时必须非空，否则报告不可审计
};

inline GateOutcome not_run(std::string reason)
{
    GateOutcome outcome;
    outcome.state = GateState::NotRun;
    outcome.reason = std::move(reason);
    return outcome;
}

inline GateOutcome pass()
{
    GateOutcome outcome;
    outcome.state = GateState::Pass;
    return outcome;
}

inline GateOutcome fail(std::string reason)
{
    GateOutcome outcome;
    outcome.state = GateState::Fail;
    outcome.reason = std::move(reason);
    return outcome;
}

// 小的、确定性的数字格式化（报告里要能直接比对文本）
inline std::string format_p(double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return std::string(buffer);
}

/* 把 harness::Gate 映射成门状态。
 *
 * 两处刻意的归因区分：
 *   - InvalidReference：坏的是**参考实现**，不是被测算子。必须能看出来，
 *     否则有人会去优化一个本来就没问题的 kernel。
 *   - LengthMismatch：坏的是**调用方**（缓冲区/长度传错），不是 kernel 算错。
 * 这两种情况都阻断晋升，但责任人不同，报告里必须分开。 */
inline GateOutcome from_gate(harness::Gate gate, const std::string &what)
{
    if (gate == harness::Gate::Pass) {
        return pass();
    }
    std::string reason = what + "=" + harness::gate_name(gate);
    if (gate == harness::Gate::InvalidReference) {
        reason += " (reference itself is non-finite: the oracle is broken, not the candidate)";
    } else if (gate == harness::Gate::LengthMismatch) {
        reason += " (buffer/shape mismatch: harness caller bug, not a kernel bug)";
    }
    return fail(reason);
}

// ---------------------------------------------------------------- 微基准门
/* 微基准的门槛不是"更快"，而是"不劣化"：
 *   b_faster        -> pass  候选更快
 *   indistinguishable -> pass 没有证据表明更差（n=7 时 4:3 噪声也落在这里）
 *   a_faster        -> fail  候选显著更慢
 * 再加一条安全网：回归幅度超过 max_regression 时，即使未达统计显著也判失败。
 * 理由：**测不出显著性的巨大回归，说明测量本身不可信**，不可信就不许晋升。 */
struct BenchPolicy {
    double max_regression = 0.10; // a.median/b.median < 0.90 即视为超过守护线
};

inline GateOutcome from_bench(const std::string &verdict, double speedup, double sign_test_p,
                              const BenchPolicy &policy)
{
    if (verdict == "not_run" || verdict.empty()) {
        return not_run("microbench not run");
    }
    if (verdict == "a_faster") {
        return fail("microbench=a_faster (candidate slower, p=" + format_p(sign_test_p) + ")");
    }
    if (verdict != "b_faster" && verdict != "indistinguishable") {
        return fail("microbench=unknown verdict '" + verdict + "'");
    }
    const double guard = 1.0 - policy.max_regression;
    if (speedup < guard) {
        return fail("microbench regression: speedup=" + format_p(speedup) + " < guard " +
                    format_p(guard) + " while not significant (p=" + format_p(sign_test_p) +
                    "); measurement is untrustworthy, refusing to promote");
    }
    return pass();
}

struct AcceptInput {
    std::string op = "unknown";
    GateOutcome correctness;
    GateOutcome microbench;
    GateOutcome l4;
    // 微基准细节（可为空字符串；仅用于报告，不参与裁决）
    std::string bench_verdict;
    double bench_speedup = 0.0;
    double bench_sign_test_p = 1.0;
    BenchPolicy bench_policy;
};

struct AcceptanceReport {
    std::string op;
    GateState correctness = GateState::NotRun;
    GateState microbench = GateState::NotRun;
    GateState l4 = GateState::NotRun;
    std::string verdict; // pass / fail / incomplete
    bool promotable = false;
    std::vector<std::string> blocking;
    // 裁决时使用的策略参数：没有规则记录的判决是不可审计的
    double max_regression = 0.10;
    std::string bench_verdict;
    double bench_speedup = 0.0;
    double bench_sign_test_p = 1.0;
    std::string correctness_reason;
    std::string microbench_reason;
    std::string l4_reason;
};

/* 聚合规则（顺序即优先级）：
 *   1. 任一 Fail        -> verdict = "fail"，不可晋升
 *   2. 否则任一 NotRun  -> verdict = "incomplete"，不可晋升
 *   3. 否则             -> verdict = "pass"，可晋升
 */
inline AcceptanceReport evaluate(const AcceptInput &input)
{
    AcceptanceReport report;
    report.op = input.op;
    report.correctness = input.correctness.state;
    report.microbench = input.microbench.state;
    report.l4 = input.l4.state;
    report.max_regression = input.bench_policy.max_regression;
    report.bench_verdict = input.bench_verdict;
    report.bench_speedup = input.bench_speedup;
    report.bench_sign_test_p = input.bench_sign_test_p;
    report.correctness_reason = input.correctness.reason;
    report.microbench_reason = input.microbench.reason;
    report.l4_reason = input.l4.reason;

    struct Named {
        const char *key;
        GateState state;
    };
    const Named gates[] = {
        {"correctness_gate", report.correctness},
        {"microbench", report.microbench},
        {"l4_gate", report.l4},
    };

    bool any_fail = false;
    bool any_not_run = false;
    for (const Named &gate : gates) {
        if (gate.state == GateState::Fail) {
            any_fail = true;
            report.blocking.push_back(std::string(gate.key) + "=fail");
        } else if (gate.state == GateState::NotRun) {
            any_not_run = true;
            report.blocking.push_back(std::string(gate.key) + "=not_run");
        }
    }

    if (any_fail) {
        report.verdict = "fail";
    } else if (any_not_run) {
        report.verdict = "incomplete";
    } else {
        report.verdict = "pass";
    }
    report.promotable = report.verdict == "pass";
    return report;
}

/* 退出码：0 可晋升 / 1 失败 / 2 证据不全。
 * incomplete 与 fail 分开，是因为处置方式不同：
 *   fail        -> 回退候选，继续改
 *   incomplete  -> 补上缺失的门，禁止基于残缺证据做决定 */
inline int exit_code(const AcceptanceReport &report)
{
    if (report.verdict == "pass") {
        return 0;
    }
    if (report.verdict == "fail") {
        return 1;
    }
    return 2;
}

// ---------------------------------------------------------------- JSON
inline std::string gate_object(const GateOutcome &outcome)
{
    std::string body;
    harness::json_kv(body, "state", gate_state_name(outcome.state));
    harness::json_kv(body, "reason", outcome.reason);
    return harness::json_close(body);
}

inline std::string acceptance_json(const AcceptInput &input, const AcceptanceReport &report)
{
    std::string body;
    harness::json_kv(body, "format", "anyinfer-op-acceptance-v1");
    harness::json_kv(body, "op", report.op);
    harness::json_kv(body, "verdict", report.verdict);
    harness::json_kv(body, "promotable", report.promotable ? "true" : "false");

    // 三段门：统一 pass/fail/not_run 三态，便于机器消费
    harness::json_kv(body, "correctness_gate", gate_state_name(report.correctness));
    harness::json_kv(body, "microbench", gate_state_name(report.microbench));
    harness::json_kv(body, "l4_gate", gate_state_name(report.l4));

    std::string blocking;
    for (const std::string &entry : report.blocking) {
        if (!blocking.empty()) {
            blocking += ',';
        }
        blocking += '"' + entry + '"';
    }
    body += "\"blocking\":[" + blocking + "],";

    // 细节：每一段都带原因，且带上裁决时用的策略参数
    body += "\"correctness\":" + gate_object(input.correctness) + ',';
    {
        std::string bench;
        harness::json_kv(bench, "state", gate_state_name(report.microbench));
        harness::json_kv(bench, "verdict", report.bench_verdict);
        harness::json_kv(bench, "speedup", report.bench_speedup);
        harness::json_kv(bench, "sign_test_p", report.bench_sign_test_p);
        harness::json_kv(bench, "max_regression", report.max_regression);
        harness::json_kv(bench, "reason", report.microbench_reason);
        body += "\"microbench_detail\":" + harness::json_close(bench) + ',';
    }
    body += "\"l4\":" + gate_object(input.l4);

    return harness::json_close(body);
}

} // namespace anyinfer::accept
