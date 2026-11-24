#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 图捕获 / 间接寻址探针（里程碑 A）。
 *
 * 单核探针：按 table 的间接值从 arena 取块写入 out。它唯一要证明的事是：
 *   「图重放时，kernel 读到的是固定地址缓冲的**当前内容**，而不是捕获时的内容。」
 *
 * table   [rows]   int32，物理块编号；负值跳过该行（out 保持原值）
 * arena   [blocks, elems] float32，连续行主序，行首需 32B 对齐
 * out     [rows, elems]   float32
 * use_table = 0 时忽略 table，恒定读块 3
 *   —— 控制臂：地址在捕获期即已知，用来区分「捕获通路坏」与「间接寻址失败」
 * elems   必须是 8 的倍数（32B），否则 DataCopy 对齐不满足
 * blocks  参与计算的 AI Core 数；探针固定用 1，时序简单便于归因
 *
 * 返回码是 aclError。缓冲区必须互不重叠且存活到流结束。
 */
int32_t anyinfer_graph_probe_launch(void *stream, uint32_t blocks, uint32_t rows,
                                    uint32_t elems, uint32_t use_table,
                                    void *table, void *arena, void *out);

#ifdef __cplusplus
}
#endif
