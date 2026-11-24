// 无 NPU 也能跑的裁决逻辑自测。
//
// 为什么单独抽出来测：在里程碑 A 里，最容易写错的不是 kernel，也不是 ACL 调用，
// 而是**从四臂结果到最终裁决的映射**。一旦映射写错，一次"看起来通过"的判决
// 会把整个 Phase 2 的内存管理器带到错的方向上。所以这部分必须在没有设备的 CI 里被覆盖。

#include "graph_probe_verdict.h"

#include <cstdio>
#include <cstring>

using graph_probe::ArmResults;
using graph_probe::Outcome;
using graph_probe::Verdict;

namespace {

int failures = 0;

void expect_verdict(const char *case_name, const ArmResults &results, Verdict want)
{
    const Verdict got = graph_probe::decide(results);
    if (got != want) {
        std::printf("FAIL %-24s want=%-24s got=%s\n", case_name,
                    graph_probe::verdict_name(want), graph_probe::verdict_name(got));
        ++failures;
        return;
    }
    std::printf("ok   %-24s -> %s (%s)\n", case_name, graph_probe::verdict_name(got),
                graph_probe::remedy(got));
}

ArmResults healthy()
{
    ArmResults r{};
    r.arm0_kernel = true;
    r.arm1_control = true;
    r.arm2_first = Outcome::Match;
    r.arm2_second = Outcome::Match;
    r.arm3_reentrant = true;
    r.nodes = 1;
    return r;
}

} // namespace

int main()
{
    // ---- classify：三种基本形态 ----
    const uint32_t expected[4] = {1, 2, 3, 4};
    const uint32_t match[4] = {1, 2, 3, 4};
    const uint32_t sentinel[4] = {graph_probe::SENTINEL, graph_probe::SENTINEL,
                                  graph_probe::SENTINEL, graph_probe::SENTINEL};
    const uint32_t stale[4] = {1, 2, 3, 9};
    const uint32_t partial[4] = {1, 2, 3, graph_probe::SENTINEL};

    if (graph_probe::classify(match, expected, 4) != Outcome::Match) {
        std::printf("FAIL classify(match)\n");
        ++failures;
    }
    if (graph_probe::classify(sentinel, expected, 4) != Outcome::NotExecuted) {
        std::printf("FAIL classify(sentinel)\n");
        ++failures;
    }
    if (graph_probe::classify(stale, expected, 4) != Outcome::Stale) {
        std::printf("FAIL classify(stale)\n");
        ++failures;
    }
    // 部分匹配只能是 Stale：只要有一个元素不匹配就不能判成功。
    if (graph_probe::classify(partial, expected, 4) != Outcome::Stale) {
        std::printf("FAIL classify(partial)\n");
        ++failures;
    }
    // 空区间不能误判为成功（否则零长度缓冲会给出假阳性）。
    if (graph_probe::classify(match, expected, 0) != Outcome::Match) {
        std::printf("FAIL classify(empty)\n");
        ++failures;
    }

    // ---- decide：七种裁决路径 ----
    expect_verdict("all good", healthy(), Verdict::IndirectionSupported);

    {
        ArmResults r = healthy();
        r.arm0_kernel = false;
        // kernel 坏时，即使后面各臂"看起来正常"也不许判通过。
        expect_verdict("kernel broken", r, Verdict::KernelBroken);
    }
    {
        ArmResults r = healthy();
        r.nodes = 0;
        expect_verdict("empty graph", r, Verdict::EmptyGraph);
    }
    {
        // nodes == -1 表示 API 不可用，不能据此判空图。
        ArmResults r = healthy();
        r.nodes = -1;
        expect_verdict("nodes unavailable", r, Verdict::IndirectionSupported);
    }
    {
        ArmResults r = healthy();
        r.arm1_control = false;
        expect_verdict("control arm failed", r, Verdict::CapturePathBroken);
    }
    {
        ArmResults r = healthy();
        r.arm2_first = Outcome::Stale;
        expect_verdict("first replay stale", r, Verdict::FirstReplayStale);
    }
    {
        ArmResults r = healthy();
        r.arm2_first = Outcome::NotExecuted;
        expect_verdict("first replay no-run", r, Verdict::ReplayNotExecuting);
    }
    {
        // 核心判决：换了表却仍读到旧块内容。
        ArmResults r = healthy();
        r.arm2_second = Outcome::Stale;
        expect_verdict("content folded", r, Verdict::ContentFolded);
    }
    {
        ArmResults r = healthy();
        r.arm2_second = Outcome::NotExecuted;
        expect_verdict("second replay no-run", r, Verdict::ReplayNotExecuting);
    }
    {
        // 二次重放对、三次不对 -> 仍有问题，不能判通过。
        ArmResults r = healthy();
        r.arm3_reentrant = false;
        expect_verdict("not reentrant", r, Verdict::ContentFolded);
    }

    // ---- is_pass：只有一种裁决放行 ----
    if (!graph_probe::is_pass(Verdict::IndirectionSupported)) {
        std::printf("FAIL is_pass(IndirectionSupported)\n");
        ++failures;
    }
    if (graph_probe::is_pass(Verdict::ContentFolded)) {
        std::printf("FAIL is_pass(ContentFolded)\n");
        ++failures;
    }

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\ngraph probe verdict logic: all checks passed\n");
    return 0;
}
