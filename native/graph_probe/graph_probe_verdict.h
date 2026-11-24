#pragma once
#include <cstdint>
#include <cstring>

namespace graph_probe {

// 哨兵：aclrtMemset(ptr, n, 0xEF, n) 逐字节写入，故 32 位位模式为 0xEFEFEFEF。
inline constexpr uint32_t SENTINEL = 0xEFEFEFEFu;

enum class Outcome {
    Match,       // 与本次期望的块内容一致 -> 间接寻址在当前重放中生效
    NotExecuted, // 全部仍为哨兵 -> 图没有执行（空图 / 未启动 / 重放无效）
    Stale,       // 既非期望也非哨兵 -> 读到的不是本次写入的表（折叠或表未更新）
};

inline uint32_t bits(float value)
{
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

inline const char *name(Outcome outcome)
{
    switch (outcome) {
    case Outcome::Match:
        return "match";
    case Outcome::NotExecuted:
        return "not_executed";
    case Outcome::Stale:
        return "stale";
    }
    return "unknown";
}

// 单臂比对。count 是 uint32 元素个数（rows * elems）。
inline Outcome classify(const uint32_t *got, const uint32_t *expected, uint32_t count)
{
    bool match = true;
    bool sentinel = true;
    for (uint32_t i = 0; i < count; ++i) {
        match = match && got[i] == expected[i];
        sentinel = sentinel && got[i] == SENTINEL;
    }
    if (match) {
        return Outcome::Match;
    }
    if (sentinel) {
        return Outcome::NotExecuted;
    }
    return Outcome::Stale;
}

// 四臂结果。把裁决表从测试驱动里抽出来，是为了让它在没有 NPU 的 CI 中可验证：
// 裁决逻辑写错比 kernel 写错更隐蔽。
struct ArmResults {
    bool arm0_kernel;      // 不捕获，直接下发
    bool arm1_control;     // 捕获 + 直接地址（标量固化）
    Outcome arm2_first;    // 捕获 + 间接寻址，首次重放
    Outcome arm2_second;   // 捕获 + 间接寻址，换表后二次重放  <- 主判决
    bool arm3_reentrant;   // 再次换表重放
    int nodes;             // 图节点数，-1 表示该 API 不可用
};

enum class Verdict {
    KernelBroken,
    EmptyGraph,
    CapturePathBroken,
    ContentFolded,
    ReplayNotExecuting,
    FirstReplayStale,
    IndirectionSupported,
};

inline const char *verdict_name(Verdict verdict)
{
    switch (verdict) {
    case Verdict::KernelBroken:
        return "kernel_broken";
    case Verdict::EmptyGraph:
        return "empty_graph";
    case Verdict::CapturePathBroken:
        return "capture_path_broken";
    case Verdict::ContentFolded:
        return "content_folded";
    case Verdict::ReplayNotExecuting:
        return "replay_not_executing";
    case Verdict::FirstReplayStale:
        return "first_replay_stale";
    case Verdict::IndirectionSupported:
        return "indirection_supported";
    }
    return "unknown";
}

// 判定顺序即归因顺序：越靠前的失败越"上游"，先修上游再谈下游。
inline Verdict decide(const ArmResults &r)
{
    if (!r.arm0_kernel) {
        return Verdict::KernelBroken; // 与捕获无关
    }
    if (r.nodes == 0) {
        return Verdict::EmptyGraph; // 该下发路径未进图
    }
    if (!r.arm1_control) {
        return Verdict::CapturePathBroken; // 直接地址都不过
    }
    if (r.arm2_first != Outcome::Match) {
        return r.arm2_first == Outcome::NotExecuted ? Verdict::ReplayNotExecuting
                                                    : Verdict::FirstReplayStale;
    }
    if (r.arm2_second == Outcome::Match && r.arm3_reentrant) {
        return Verdict::IndirectionSupported;
    }
    if (r.arm2_second == Outcome::NotExecuted) {
        return Verdict::ReplayNotExecuting;
    }
    return Verdict::ContentFolded;
}

// 只有这一个裁决允许继续按 DESIGN §5 的间接寻址方案推进。
inline bool is_pass(Verdict verdict)
{
    return verdict == Verdict::IndirectionSupported;
}

// 失败时给出的下一步（后备梯队 A' -> B -> C）。
inline const char *remedy(Verdict verdict)
{
    switch (verdict) {
    case Verdict::IndirectionSupported:
        return "proceed";
    case Verdict::KernelBroken:
        return "fix kernel";
    case Verdict::EmptyGraph:
        return "switch to A' (explicit graph_task_update)";
    case Verdict::CapturePathBroken:
        return "check stream order and remove sync from capture region";
    case Verdict::FirstReplayStale:
        return "check H2D ordering of the block table";
    case Verdict::ReplayNotExecuting:
        return "check graph instantiate/launch";
    case Verdict::ContentFolded:
        return "switch to A' / B / C";
    }
    return "unknown";
}

} // namespace graph_probe
