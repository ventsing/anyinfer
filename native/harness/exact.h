#pragma once
#include <cmath>
#include <cstdint>

// B-01：有界二进制有理数与「精确性判据」。
//
// 为什么需要这个：L1 位级验收门声称「逐位相等」，但这个声称有前提 ——
// 只有当输入使得**每一步中间结果在 float32 中都能被精确表示**时，
// 「实现 A 与实现 B、或同一实现的两种归约顺序」才必然给出相同位模式。
// 前提不成立时，位级不等说明不了任何问题（可能是舍入顺序不同，而非实现错）。
//
// 所以：任何声称位级的测试，都必须先用本文件判定「这组输入下位级相等是否本来就成立」。
// 判据不成立时，那个测试应当降级为 L2（FP64 参考 + 容差）。

namespace anyinfer::exact {

// float32 的有效位数（1 隐含位 + 23 显式位）
inline constexpr int kFp32SignificandBits = 24;
// 「严格小于 2^24」写成可直接比较的上界
inline constexpr int64_t kFp32ExactCapacity = (int64_t(1) << kFp32SignificandBits) - 1;
// 允许的最大移位。刻意排除次正规范围：那里每个值只有不到 24 位有效位，
// 是「看起来精确、实际不精确」最经典的陷阱。
inline constexpr int kMaxShift = 126;

// value = k * 2^-shift
struct Dyadic {
    int32_t k;
    int32_t shift;
};

inline int64_t magnitude(int32_t k)
{
    return k < 0 ? -int64_t(k) : int64_t(k);
}

// 该值能否被 float32 精确表示（不做任何舍入）。
inline bool representable(const Dyadic &d)
{
    if (d.shift < 0 || d.shift > kMaxShift) {
        return false;
    }
    return magnitude(d.k) <= kFp32ExactCapacity;
}

inline float to_float(const Dyadic &d)
{
    // 用 double 中转：double 的 53 位有效位足以精确表示 k（≤2^24），
    // 且 ldexp 对 2 的整数次幂是精确的；representable() 已保证 float32 转换不丢位。
    return static_cast<float>(std::ldexp(static_cast<double>(d.k), -d.shift));
}

/* n 个**同一 shift**的 dyadic 求和是否精确。
 * 同 shift 意味着所有项共用同一网格 2^-shift，故和的整数表示最大为 n*max|k|；
 * 只要它不超过 2^24-1，和就能被 float32 精确表示（与相加顺序无关）。
 * 这条正是「位级相等与归约顺序无关」的充要条件。 */
inline bool sum_is_exact(uint64_t count, int64_t max_abs_k)
{
    if (count == 0 || max_abs_k == 0) {
        return true;
    }
    if (max_abs_k > kFp32ExactCapacity) {
        return false;
    }
    return count <= uint64_t(kFp32ExactCapacity) / uint64_t(max_abs_k);
}

/* (k1 * 2^-s1) * (k2 * 2^-s2) = (k1*k2) * 2^-(s1+s2)：精确当且仅当 |k1*k2| <= 2^24-1。 */
inline bool product_is_exact(int32_t k1, int32_t k2)
{
    return magnitude(k1) == 0 || magnitude(k2) == 0
               ? true
               : magnitude(k1) <= kFp32ExactCapacity / magnitude(k2);
}

/* 点积 sum a_i*b_i 的精确性：所有部分积对齐到同一网格 2^-(sa+sb)，
 * 故只要 count * max|k_a*k_b| <= 2^24-1 就精确。 */
inline bool dot_is_exact(uint64_t count, int64_t max_abs_product_k)
{
    return sum_is_exact(count, max_abs_product_k);
}

// 确定性哈希：不依赖 <random> 的实现细节，避免跨编译器/标准库产生不同输入。
inline uint32_t mix(uint32_t seed, uint32_t index)
{
    uint32_t x = (seed + 0x9E3779B1u) ^ (index + 0x85EBCA6Bu);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

/* 生成 |k| <= bound、指定 shift 的 dyadic。同 (seed, index) 必得同值。 */
inline Dyadic generate(uint32_t seed, uint32_t index, int32_t shift, int32_t bound)
{
    if (bound <= 0) {
        return Dyadic{0, shift};
    }
    if (bound > (kFp32ExactCapacity - 1) / 2) {
        bound = int32_t((kFp32ExactCapacity - 1) / 2);
    }
    const uint32_t span = uint32_t(2) * uint32_t(bound) + 1u;
    const int32_t k = int32_t(mix(seed, index) % span) - bound;
    return Dyadic{k, shift};
}

/* 一组输入是否满足「任意归约顺序都位级相等」的前提。
 * 供测试在断言位级相等之前先自检，避免把舍入差异误判为缺陷。 */
inline bool order_independent(uint64_t count, int64_t max_abs_k)
{
    return sum_is_exact(count, max_abs_k);
}

/* ---------------------------------------------------------------- 典型预算
 *
 * bf16 输入、fp32 累加的算子（hc_post / dequant / 各类 elementwise 融合）：
 *   1. bf16 有 8 位有效位 -> 两个 bf16 之积至多 16 位，在 fp32 中**必然精确**；
 *   2. 若所有输入都落在同一网格 2^-s 上，则所有部分积都落在网格 2^-2s 上；
 *   3. 于是「N 项累加精确」<=> N * max|ka*kb| <= 2^24 - 1。
 *
 * 例（hc_post）：1 个 post*x 项 + 4 个 comb*residual 项 = 5 项；
 *   max|k| = 255 时 max|ka*kb| = 65025，5 * 65025 = 325125，远小于 2^24。
 *   余量三个数量级 —— 所以 hc_post 可以理直气壮地声明 L1 位级门，
 *   而 GEMM（K 上千、含量化缩放）通常不行，应降到 L2。
 *
 * 反之：如果算子内部有 split-K、atomic 累加、或按运行时数据决定归约阶数，
 * 那么无论输入多精确都不存在「顺序无关」，**不许声称 L1**。
 */

} // namespace anyinfer::exact
