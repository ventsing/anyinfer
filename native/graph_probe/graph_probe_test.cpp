// 里程碑 A：图捕获 + 间接寻址判决。
//
// 四臂设计的目的不是"测出能不能跑"，而是让**每一种失败都能归因**：
//   Arm 0  不捕获直接下发    -> kernel 本身对不对
//   Arm 1  捕获 + 直接地址   -> 捕获/重放通路本身通不通（标量在此被固化）
//   Arm 2  捕获 + 间接寻址   -> 主判决：换表内容能否被重放看到
//   Arm 3  再次换表重放      -> 可重入性
//
// 退出码：0 = indirection_supported；1 = 其它任何裁决（含"不支持"）。
// 判决会同时写入 --output 指定的 JSON（.part + rename 原子发布）。

#include "graph_compat.h"
#include "graph_probe.h"
#include "graph_probe_verdict.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using graph_probe::ArmResults;
using graph_probe::Outcome;

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kBlocks = 4;
constexpr uint32_t kElems = 256; // 8 的倍数，32B 对齐
constexpr float kBase = 1000.0f; // 块 b 的填充值 = kBase + b
// 控制臂固定读块；与判决臂使用的块刻意不同
constexpr int32_t kControlBlock = 3;

std::string g_json;

#define CHECK_ACL(expr)                                                              \
    do {                                                                             \
        aclError code__ = (expr);                                                    \
        if (code__ != ACL_SUCCESS) {                                                 \
            const char *msg__ = aclGetRecentErrMsg();                                \
            std::printf("{\"verdict\":\"acl_error\",\"at\":\"%s\",\"code\":%d,"      \
                        "\"msg\":\"%s\"}\n",                                         \
                        #expr, static_cast<int>(code__), msg__ ? msg__ : "");         \
            return 1;                                                                \
        }                                                                            \
    } while (0)

// table 指向的各块内容的位模式期望
std::vector<uint32_t> expect(const int32_t *table)
{
    std::vector<uint32_t> out(size_t(kRows) * kElems);
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t pattern = graph_probe::bits(kBase + float(table[r]));
        for (uint32_t i = 0; i < kElems; ++i) {
            out[size_t(r) * kElems + i] = pattern;
        }
    }
    return out;
}

std::vector<uint32_t> expect_block(int32_t block)
{
    const int32_t table[kRows] = {block, block};
    return expect(table);
}

} // namespace

int main(int argc, char **argv)
{
    std::string output;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--output") {
            output = argv[i + 1];
        }
    }

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t arena_bytes = size_t(kBlocks) * kElems * sizeof(float);
    const size_t out_bytes = size_t(kRows) * kElems * sizeof(float);
    const size_t table_bytes = size_t(kRows) * sizeof(int32_t);

    void *arena = nullptr;
    void *out = nullptr;
    int32_t *table = nullptr;
    CHECK_ACL(aclrtMalloc(&arena, arena_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&table), table_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // arena 初始化：块 b 全填 kBase + b（小整数，float 精确可比）
    std::vector<float> host_arena(size_t(kBlocks) * kElems);
    for (uint32_t b = 0; b < kBlocks; ++b) {
        for (uint32_t i = 0; i < kElems; ++i) {
            host_arena[size_t(b) * kElems + i] = kBase + float(b);
        }
    }
    CHECK_ACL(aclrtMemcpy(arena, arena_bytes, host_arena.data(), arena_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<uint32_t> got(size_t(kRows) * kElems);
    auto read_out = [&]() -> int {
        CHECK_ACL(aclrtMemcpy(got.data(), out_bytes, out, out_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        return 0;
    };

    // 每次重放前写哨兵：否则"读了旧块"与"图根本没跑"都是旧值，不可区分。
    // host_table == nullptr 表示沿用设备上已有的表内容。
    auto arm = [&](graph_probe::GraphExec exec, const int32_t *host_table) -> int {
        CHECK_ACL(aclrtMemset(out, out_bytes, 0xEF, out_bytes));
        if (host_table != nullptr) {
            CHECK_ACL(aclrtMemcpyAsync(table, table_bytes, host_table, table_bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
        }
        CHECK_ACL(graph_probe::launch(exec, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        return read_out();
    };

    ArmResults result{};
    result.nodes = -1;

    // ---------------- Arm 0：不捕获，验证 kernel 本身 ----------------
    const int32_t table_01[kRows] = {0, 1};
    CHECK_ACL(aclrtMemcpy(table, table_bytes, table_01, table_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(out, out_bytes, 0xEF, out_bytes));
    CHECK_ACL(anyinfer_graph_probe_launch(stream, 1, kRows, kElems, 1, table, arena, out));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    read_out();
    const std::vector<uint32_t> want_01 = expect(table_01);
    result.arm0_kernel =
        std::memcmp(got.data(), want_01.data(), out_bytes) == 0;
    if (!result.arm0_kernel) {
        const graph_probe::Verdict verdict = graph_probe::decide(result);
        std::printf("{\"verdict\":\"%s\",\"remedy\":\"%s\"}\n",
                    graph_probe::verdict_name(verdict), graph_probe::remedy(verdict));
        return 1;
    }

    // ---------------- Arm 1：控制臂，捕获直接地址（use_table=0 -> 恒读块 3）--------
    graph_probe::Graph graph_ctrl{};
    graph_probe::GraphExec exec_ctrl{};
    CHECK_ACL(graph_probe::capture_begin(stream));
    CHECK_ACL(anyinfer_graph_probe_launch(stream, 1, kRows, kElems, 0, nullptr, arena, out));
    CHECK_ACL(graph_probe::capture_end(stream, &graph_ctrl));
    CHECK_ACL(graph_probe::instantiate(graph_ctrl, &exec_ctrl));
    result.nodes = graph_probe::node_count(graph_ctrl);

    arm(exec_ctrl, nullptr);
    const std::vector<uint32_t> want_ctrl = expect_block(kControlBlock);
    result.arm1_control =
        std::memcmp(got.data(), want_ctrl.data(), out_bytes) == 0;

    if (!result.arm1_control || result.nodes == 0) {
        const graph_probe::Verdict verdict = graph_probe::decide(result);
        std::printf("{\"verdict\":\"%s\",\"nodes\":%d,\"remedy\":\"%s\"}\n",
                    graph_probe::verdict_name(verdict), result.nodes,
                    graph_probe::remedy(verdict));
        return 1;
    }

    // ---------------- Arm 2：判决臂，间接寻址 ----------------
    graph_probe::Graph graph_ind{};
    graph_probe::GraphExec exec_ind{};
    CHECK_ACL(graph_probe::capture_begin(stream));
    CHECK_ACL(anyinfer_graph_probe_launch(stream, 1, kRows, kElems, 1, table, arena, out));
    CHECK_ACL(graph_probe::capture_end(stream, &graph_ind));
    CHECK_ACL(graph_probe::instantiate(graph_ind, &exec_ind));

    arm(exec_ind, table_01);
    result.arm2_first = graph_probe::classify(got.data(), want_01.data(), kRows * kElems);

    // 换表：若间接寻址生效，重放应读到块 2/3 而不是块 0/1
    const int32_t table_23[kRows] = {2, 3};
    const std::vector<uint32_t> want_23 = expect(table_23);
    arm(exec_ind, table_23);
    result.arm2_second = graph_probe::classify(got.data(), want_23.data(), kRows * kElems);

    // ---------------- Arm 3：可重入，再换回 ----------------
    const int32_t table_10[kRows] = {1, 0};
    const std::vector<uint32_t> want_10 = expect(table_10);
    arm(exec_ind, table_10);
    result.arm3_reentrant =
        graph_probe::classify(got.data(), want_10.data(), kRows * kElems) == Outcome::Match;

    // ---------------- 裁决 ----------------
    const graph_probe::Verdict verdict = graph_probe::decide(result);
    const char *verdict_text = graph_probe::verdict_name(verdict);

    std::printf("{\"verdict\":\"%s\",\"nodes\":%d,\"arm0_kernel\":\"%s\","
                "\"arm1_control\":\"%s\",\"arm2_first\":\"%s\",\"arm2_second\":\"%s\","
                "\"arm3_reentrant\":\"%s\",\"remedy\":\"%s\"}\n",
                verdict_text, result.nodes, result.arm0_kernel ? "pass" : "fail",
                result.arm1_control ? "pass" : "fail",
                graph_probe::name(result.arm2_first), graph_probe::name(result.arm2_second),
                result.arm3_reentrant ? "pass" : "fail", graph_probe::remedy(verdict));

    if (!output.empty()) {
        std::ofstream file(output + ".part", std::ios::binary);
        file << "{\"verdict\":\"" << verdict_text << "\",\"nodes\":" << result.nodes
             << ",\"arm0_kernel\":\"" << (result.arm0_kernel ? "pass" : "fail")
             << "\",\"arm1_control\":\"" << (result.arm1_control ? "pass" : "fail")
             << "\",\"arm2_first\":\"" << graph_probe::name(result.arm2_first)
             << "\",\"arm2_second\":\"" << graph_probe::name(result.arm2_second)
             << "\",\"arm3_reentrant\":\"" << (result.arm3_reentrant ? "pass" : "fail")
             << "\",\"remedy\":\"" << graph_probe::remedy(verdict) << "\"}\n";
        file.close();
        std::rename((output + ".part").c_str(), output.c_str());
    }

    graph_probe::destroy_exec(exec_ind);
    graph_probe::destroy(graph_ind);
    graph_probe::destroy_exec(exec_ctrl);
    graph_probe::destroy(graph_ctrl);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return graph_probe::is_pass(verdict) ? 0 : 1;
}
