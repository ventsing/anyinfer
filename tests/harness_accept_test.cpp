// B-04b 的验证：裁决策略本身必须被测。
//
// 这条测试守的是一个特定的失败模式：**"永远绿"的验收门**。
// 如果 NotRun 被当成 Pass，那么第一个版本的验收工具就会在 L4 从未接线的情况下
// 一直输出 promotable=true，而没有人会发现。所以这里最核心的断言是
// "默认状态必须是 NotRun，且 NotRun 必须阻断晋升"。

#include "accept.h"

#include <cstdio>
#include <string>

namespace {

using namespace anyinfer::accept;

int failures = 0;

void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

void check_contains(const std::string &haystack, const char *needle, const char *what)
{
    if (haystack.find(needle) == std::string::npos) {
        std::printf("FAIL %s: '%s' not found in '%s'\n", what, needle, haystack.c_str());
        ++failures;
    }
}

AcceptInput all_pass()
{
    AcceptInput input;
    input.op = "hc_post";
    input.correctness = pass();
    input.microbench = pass();
    input.l4 = pass();
    input.bench_verdict = "b_faster";
    input.bench_speedup = 1.10;
    return input;
}

// ---------------------------------------------------------------- 默认安全
void defaults_are_not_run()
{
    // 关键：默认构造出来必须是"未跑过"，而不是"通过"
    GateOutcome outcome;
    check(outcome.state == GateState::NotRun, "default GateOutcome is NotRun, not Pass");

    const AcceptInput input;
    const AcceptanceReport report = evaluate(input);
    check(report.correctness == GateState::NotRun, "default correctness is NotRun");
    check(report.microbench == GateState::NotRun, "default microbench is NotRun");
    check(report.l4 == GateState::NotRun, "default l4 is NotRun");
    check(report.verdict == "incomplete", "default verdict is incomplete");
    check(!report.promotable, "default is not promotable");
    check(exit_code(report) == 2, "incomplete exits with 2");
}

// ---------------------------------------------------------------- 聚合规则
void three_states()
{
    {
        const AcceptInput input = all_pass();
        const AcceptanceReport report = evaluate(input);
        check(report.verdict == "pass", "all pass -> verdict pass");
        check(report.promotable, "all pass -> promotable");
        check(report.blocking.empty(), "all pass -> nothing blocking");
        check(exit_code(report) == 0, "pass exits with 0");
    }
    {
        AcceptInput input = all_pass();
        input.correctness = fail("bit_mismatch");
        const AcceptanceReport report = evaluate(input);
        check(report.verdict == "fail", "any fail -> verdict fail");
        check(!report.promotable, "any fail -> not promotable");
        check(exit_code(report) == 1, "fail exits with 1");
        check(report.blocking.size() == 1 && report.blocking[0] == "correctness_gate=fail",
              "blocking names the failing gate");
    }
    {
        AcceptInput input = all_pass();
        input.l4 = not_run("engine not wired (C-05)");
        const AcceptanceReport report = evaluate(input);
        check(report.verdict == "incomplete", "not_run -> verdict incomplete");
        check(!report.promotable, "not_run blocks promotion");
        check(exit_code(report) == 2, "incomplete exits with 2");
        check(report.blocking.size() == 1 && report.blocking[0] == "l4_gate=not_run",
              "blocking names the unrun gate");
    }
}

void fail_takes_precedence_over_not_run()
{
    AcceptInput input = all_pass();
    input.correctness = fail("bit_mismatch");
    input.l4 = not_run("engine not wired");
    const AcceptanceReport report = evaluate(input);
    check(report.verdict == "fail", "fail beats incomplete");
    check(report.blocking.size() == 2, "both blockers are reported");
    check_contains(report.blocking[0], "correctness_gate=fail", "fail listed first");
    check_contains(report.blocking[1], "l4_gate=not_run", "not_run also listed");
}

void multiple_failures_all_reported()
{
    AcceptInput input;
    input.correctness = fail("bit_mismatch");
    input.microbench = fail("regression");
    input.l4 = fail("non_finite");
    const AcceptanceReport report = evaluate(input);
    check(report.blocking.size() == 3, "three blockers reported");
    check(exit_code(report) == 1, "still exits with 1");
}

// ---------------------------------------------------------------- 归因
void gate_attribution()
{
    const GateOutcome ok = from_gate(anyinfer::harness::Gate::Pass, "correctness");
    check(ok.state == GateState::Pass, "Pass maps to Pass");
    check(ok.reason.empty(), "Pass carries no reason");

    const GateOutcome bits = from_gate(anyinfer::harness::Gate::BitMismatch, "correctness");
    check(bits.state == GateState::Fail, "BitMismatch maps to Fail");
    check_contains(bits.reason, "bit_mismatch", "reason names the gate");

    // 参考实现坏掉时，说清楚"坏的不是候选"
    const GateOutcome oracle = from_gate(anyinfer::harness::Gate::InvalidReference, "correctness");
    check(oracle.state == GateState::Fail, "InvalidReference maps to Fail");
    check_contains(oracle.reason, "oracle is broken", "InvalidReference blames the oracle");

    // 调用方 bug 与 kernel bug 必须分开
    const GateOutcome shape = from_gate(anyinfer::harness::Gate::LengthMismatch, "correctness");
    check(shape.state == GateState::Fail, "LengthMismatch maps to Fail");
    check_contains(shape.reason, "caller bug", "LengthMismatch blames the caller");
}

// ---------------------------------------------------------------- 微基准门
void bench_policy()
{
    const BenchPolicy policy;

    const GateOutcome faster = from_bench("b_faster", 1.20, 0.0156, policy);
    check(faster.state == GateState::Pass, "b_faster passes");

    // 4:3 噪声判 indistinguishable -> 通过：证据不足不等于更差
    const GateOutcome no_evidence = from_bench("indistinguishable", 1.00, 1.0, policy);
    check(no_evidence.state == GateState::Pass, "indistinguishable passes (not worse)");

    const GateOutcome slower = from_bench("a_faster", 0.80, 0.0156, policy);
    check(slower.state == GateState::Fail, "a_faster fails");
    check_contains(slower.reason, "candidate slower", "reason explains the direction");

    const GateOutcome skipped = from_bench("not_run", 0.0, 1.0, policy);
    check(skipped.state == GateState::NotRun, "not_run stays NotRun");

    const GateOutcome bogus = from_bench("much_faster", 5.0, 0.01, policy);
    check(bogus.state == GateState::Fail, "unknown verdict fails closed");
    check_contains(bogus.reason, "unknown verdict", "reason names the problem");
}

void regression_guard()
{
    const BenchPolicy policy; // max_regression = 0.10

    // 未达统计显著，但幅度远超守护线：测量不可信，不许晋升
    const GateOutcome huge = from_bench("indistinguishable", 0.50, 1.0, policy);
    check(huge.state == GateState::Fail, "large non-significant regression fails");
    check_contains(huge.reason, "untrustworthy", "reason says why it is refused");

    // 边界：恰好等于守护线不触发（speedup == 0.90）
    const GateOutcome boundary = from_bench("indistinguishable", 0.90, 1.0, policy);
    check(boundary.state == GateState::Pass, "speedup exactly at guard does not trip");

    // 只差一点就越线
    const GateOutcome just_over = from_bench("indistinguishable", 0.899, 1.0, policy);
    check(just_over.state == GateState::Fail, "speedup just below guard trips");

    // 策略可配置：放宽到 50% 后同一输入应当通过
    BenchPolicy loose;
    loose.max_regression = 0.50;
    const GateOutcome relaxed = from_bench("indistinguishable", 0.60, 1.0, loose);
    check(relaxed.state == GateState::Pass, "guard is configurable");
}

// ---------------------------------------------------------------- 解析与报告
void state_parsing()
{
    GateState state;
    check(parse_gate_state("pass", state) && state == GateState::Pass, "parse pass");
    check(parse_gate_state("fail", state) && state == GateState::Fail, "parse fail");
    check(parse_gate_state("not_run", state) && state == GateState::NotRun, "parse not_run");
    check(!parse_gate_state("passed", state), "parse rejects near-miss");
    check(!parse_gate_state("", state), "parse rejects empty");

    check(std::string(gate_state_name(GateState::NotRun)) == "not_run", "name not_run");
}

void json_report()
{
    AcceptInput input = all_pass();
    input.l4 = not_run("engine not wired (C-05)");
    const AcceptanceReport report = evaluate(input);
    const std::string json = acceptance_json(input, report);

    check(!json.empty() && json.front() == '{' && json.back() == '}', "json is fenced");
    check_contains(json, "\"format\":\"anyinfer-op-acceptance-v1\"", "format tag");
    // TODO 要求的三段键必须字面存在
    check_contains(json, "\"correctness_gate\":\"pass\"", "correctness_gate present");
    check_contains(json, "\"microbench\":\"pass\"", "microbench present");
    check_contains(json, "\"l4_gate\":\"not_run\"", "l4_gate present");
    check_contains(json, "\"verdict\":\"incomplete\"", "verdict present");
    check_contains(json, "\"promotable\":\"false\"", "promotable false when incomplete");
    // 策略参数必须写进报告：没有规则记录的判决不可审计
    check_contains(json, "\"max_regression\":", "policy recorded");
    check_contains(json, "engine not wired", "not_run reason carried through");

    // 括号平衡（简单的结构检查，能抓住漏闭合这类错误）
    int depth = 0;
    bool balanced = true;
    for (char c : json) {
        if (c == '{' || c == '[') {
            ++depth;
        } else if (c == '}' || c == ']') {
            if (--depth < 0) {
                balanced = false;
            }
        }
    }
    check(balanced && depth == 0, "json braces are balanced");
}

void full_pipeline_verdict()
{
    // 端到端走一遍：三段门都从各自的原始产物映射过来
    AcceptInput input;
    input.op = "hc_post";
    input.correctness = from_gate(anyinfer::harness::Gate::Pass, "correctness");
    input.bench_verdict = "b_faster";
    input.bench_speedup = 1.05;
    input.bench_sign_test_p = 0.015625;
    input.microbench =
        from_bench(input.bench_verdict, input.bench_speedup, input.bench_sign_test_p,
                   input.bench_policy);
    input.l4 = from_gate(anyinfer::harness::Gate::ToleranceExceeded, "l4");

    const AcceptanceReport report = evaluate(input);
    check(report.verdict == "fail", "L4 tolerance failure blocks promotion");
    check(report.blocking.size() == 1 && report.blocking[0] == "l4_gate=fail",
          "only L4 blocks; the other two passed");
    check(exit_code(report) == 1, "pipeline failure exits 1");
}

} // namespace

int main()
{
    defaults_are_not_run();
    three_states();
    fail_takes_precedence_over_not_run();
    multiple_failures_all_reported();
    gate_attribution();
    bench_policy();
    regression_guard();
    state_parsing();
    json_report();
    full_pipeline_verdict();

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("accept.h: all checks passed (three-state gates, NotRun blocks promotion)\n");
    return 0;
}
