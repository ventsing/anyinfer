/* AnyInfer 的 C ABI 边界（D-01）。
 *
 * 设计约束见 docs/DECISIONS.md：
 *   ADR-004  边界上只允许 POD、裸指针、标量。禁止 std::vector / std::string / bool。
 *
 * 为什么不用 C 的 enum 表示状态码：
 *   enum 的底层类型由实现决定（C 里通常是 int，但也可能被压成 char），
 *   Rust 侧的 #[repr(C)] enum 与之匹配只是"通常成立"。用 int32_t + 具名常量
 *   就把这层猜测彻底去掉 —— 编译器没得选，也就没得分歧。
 *
 * 三条自缚的规则：
 *   1. **结构体里不允许出现隐式 padding**。offsetof 必须首尾相接（abi_layout_test 会检查）。
 *      隐式 padding 里的字节未定义，会让"两次构造的 step 逐位相等"这种断言失效。
 *   2. **sizeof 被硬钉住**。加字段必须同时改这里的常量与 abi_layout.txt，
 *      否则两边各自编译通过、运行时才错位。这是刻意制造的摩擦。
 *   3. 只支持 LP64（指针 8 字节）。32 位平台直接编译失败，而不是悄悄错位。
 */
#ifndef ANYINFER_ABI_H
#define ANYINFER_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANYINFER_ABI_VERSION 1u

#if defined(__cplusplus)
#define ANYINFER_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ANYINFER_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#error "AnyInfer ABI requires C11 or C++11"
#endif

ANYINFER_STATIC_ASSERT(sizeof(void *) == 8,
                       "AnyInfer ABI is LP64-only; 32-bit targets are not supported");

/* ---------------------------------------------------------------- 状态 */
/* 状态码。0 成功，负值失败。 */
enum {
    ANYINFER_OK = 0,
    ANYINFER_ERR_INVALID_ARG = -1,  /* 调用方传了非法参数（空指针、版本不符等） */
    ANYINFER_ERR_NOT_PLANNED = -2,  /* plan 阶段没覆盖这个形状（桶未捕获） */
    ANYINFER_ERR_SHAPE = -3,        /* 运行期形状与 plan 不符 —— 会破坏图重放 */
    ANYINFER_ERR_DEVICE = -4,       /* 设备/ACL 调用失败 */
    ANYINFER_ERR_GRAPH = -5,        /* 图捕获/实例化/重放失败 */
    ANYINFER_ERR_UNSUPPORTED = -6,  /* 未实现的能力，显式拒绝而不是算错 */
    ANYINFER_ERR_INTERNAL = -7,
};

typedef int32_t AnyInferStatusCode;

/* `failed_op_id` 的哨兵：没有失败的算子。
 *
 * 用宏而不是 enum：C23 之前，enum 的取值必须落在 int 范围内，
 * 0xFFFFFFFF 会触发 -Wpedantic。这类"编译器方言差异"正是 ABI 头文件要避免的东西。 */
#define ANYINFER_OP_NONE 0xFFFFFFFFu

/* 失败时返回的信息。成功时 code == ANYINFER_OK，failed_op_id == ANYINFER_OP_NONE。
 *
 * failed_op_id 是 RSI 闭环的关键：不确定"是哪一层的哪个算子算错了"，
 * 就无法把误差归因到具体的优化动作上，闭环退化成"整体重跑一遍试试"。
 * step_index 让同一份失败能在轨迹回放里精确复现。
 */
typedef struct AnyInferStatus {
    AnyInferStatusCode code;
    uint32_t failed_op_id;
    int32_t detail;      /* 算子私有细节码，例如 aclnn 返回码的低位 */
    uint32_t reserved0;  /* 显式占位：不留隐式 padding */
    uint64_t step_index; /* 出错的那一步；未执行过则为 0 */
} AnyInferStatus;

/* ---------------------------------------------------------------- 步 */
/* flags */
enum {
    /* 本步是 decode（否则按 prefill 处理）。两者捕获的图不同，必须显式区分。 */
    ANYINFER_STEP_FLAG_DECODE = 1u << 0,
    /* 批内含有哑元行（桶填充）。读侧要裁剪、写侧要截断 slot_mapping，
     * 否则哑元行的 KV 会永久污染轨迹（见 DESIGN §4）。 */
    ANYINFER_STEP_FLAG_BUCKET_PADDED = 1u << 1,
    /* 要求引擎导出逐层张量，供 C-05 的对拍器直接消费。 */
    ANYINFER_STEP_FLAG_TRACE = 1u << 2,
};

/* 一次 step 的全部输入与输出。
 *
 * 形状字段（batch_size / num_tokens / max_blocks / vocab_size / topk）在同一个
 * 捕获图内**必须保持不变** —— 变了就是另一个桶，必须换图。引擎会检查并返回
 * ANYINFER_ERR_SHAPE，而不是让图带着错的形状重放。
 *
 * 所有指针都是**设备地址**，除了 stream（aclrtStream 句柄）。
 * 每步只允许一次 FFI 跨界（ADR-003）。
 */
typedef struct AnyInferStep {
    uint32_t abi_version; /* 必须等于 ANYINFER_ABI_VERSION */
    uint32_t flags;
    uint32_t batch_size;
    uint32_t num_tokens;
    uint32_t max_blocks;
    uint32_t vocab_size;
    uint32_t topk;
    uint32_t reserved0; /* 显式占位：不留隐式 padding */

    uint64_t step_index;     /* 单调递增；用于回放、去重与失败归因 */
    uint64_t weight_version; /* 权重 epoch；换版后旧图引用的权重已失效 */

    void *stream;

    const int32_t *token_ids;    /* [batch_size] */
    const int32_t *positions;    /* [batch_size] */
    const int32_t *block_table;  /* [batch_size * max_blocks] 设备侧间接寻址（ADR-005） */
    const int32_t *seq_lens;     /* [batch_size] */
    const int32_t *slot_mapping; /* [num_tokens] */

    void *kv_cache; /* 设备基址 */

    void *logits;        /* 输出 [batch_size, vocab_size] */
    void *topk_values;   /* 输出 [batch_size, topk] */
    int32_t *topk_indices; /* 输出 [batch_size, topk] */
    void *logsumexp;     /* 输出 [batch_size] */
    void *sampling_mask; /* 输出位图 [batch_size, ceil(vocab_size/32)] */
} AnyInferStep;

/* 钉死尺寸：加字段必须同时更新这里、abi_layout.txt 与 Rust 侧的定义。
 * 这是刻意制造的摩擦 —— 无声的错位比编译失败贵得多。 */
ANYINFER_STATIC_ASSERT(sizeof(AnyInferStatus) == 24,
                       "AnyInferStatus layout changed; update native/abi/abi_layout.txt");
ANYINFER_STATIC_ASSERT(sizeof(AnyInferStep) == 144,
                       "AnyInferStep layout changed; update native/abi/abi_layout.txt");

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ANYINFER_ABI_H */
