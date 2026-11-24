#include "kernel_operator.h"

using namespace AscendC;

namespace {
// 控制臂固定读取的块号。与判决臂使用的块（0/1/2/3 轮换）刻意不同，
// 一旦控制臂与判决臂互相串位，输出会立刻对不上。
constexpr int32_t CONTROL_BLOCK = 3;

class Probe {
public:
    __aicore__ inline void Run(GM_ADDR table, GM_ADDR arena, GM_ADDR out,
                               uint32_t rows, uint32_t elems, uint32_t use_table)
    {
        // 探针只用一个核：单核时序简单，失败即可归因，不做并行调优。
        if (GetBlockIdx() != 0) {
            return;
        }
        table_.SetGlobalBuffer((__gm__ int32_t *)table);
        arena_.SetGlobalBuffer((__gm__ float *)arena);
        out_.SetGlobalBuffer((__gm__ float *)out);
        pipe_.InitBuffer(buf_, 1, elems * sizeof(float));

        for (uint32_t r = 0; r < rows; ++r) {
            // use_table 是**标量**参数，捕获后被固化为 1。
            // 这正是本探针要暴露的区分：标量会被固化，指针所指缓冲的内容不会。
            const int64_t block = use_table ? table_.GetValue(r) : CONTROL_BLOCK;
            if (block < 0) {
                continue;
            }
            auto local = buf_.AllocTensor<float>();
            // 运行时算出的 GM 偏移。原型 hc_post_kernel.cpp 已在用同一写法。
            DataCopy(local, arena_[static_cast<uint64_t>(block) * elems], elems);
            buf_.EnQue(local);
            local = buf_.DeQue<float>();
            DataCopy(out_[static_cast<uint64_t>(r) * elems], local, elems);
            buf_.FreeTensor(local);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> buf_;
    GlobalTensor<int32_t> table_;
    GlobalTensor<float> arena_;
    GlobalTensor<float> out_;
};
} // namespace

extern "C" __global__ __aicore__ void anyinfer_graph_probe_kernel(
    GM_ADDR table, GM_ADDR arena, GM_ADDR out, uint32_t rows, uint32_t elems,
    uint32_t use_table)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Probe op;
    op.Run(table, arena, out, rows, elems, use_table);
}
