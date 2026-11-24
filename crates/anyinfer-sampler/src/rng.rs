//! Counter-based（无状态）随机数。
//!
//! **为什么不是一条 RNG 流：**
//! 流式 RNG 的结果依赖于"在这之前抽了多少次"，而抽了多少次取决于批里有多少行、
//! 有没有被抢占重算、padding 行占了多少位置 —— 于是同一请求在不同 batch 组合下
//! 会得到不同 token。那正是 DESIGN §6.3 说的"batch invariance 崩掉"。
//!
//! 这里把均匀数做成 `(seed, position)` 的**纯函数**：
//! 抽第几个与别人无关，重算同一位置得到同一个数，回放天然一致。
//! 代价是失去"连续抽取之间的独立性保证"，但本用途每步只抽一次，不存在这个问题。
//!
//! 只取 24 位是有意的：f32 的尾数就是 24 位，更多位在 f32 里表示不出来，
//! 反而会让映射非均匀（某些值被四舍五入到同一个 f32 上）。

/// Golden ratio 派生常量（SplitMix64 的常用增量）。
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const GOLDEN_2: u64 = 0xBF58_476D_1CE4_E5B9;
const GOLDEN_3: u64 = 0x94D0_49BB_1331_11EB;

/// SplitMix64 的 finalizer：双射、雪崩性好、只用整数运算，跨平台逐位一致。
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 由 `(seed, position)` 构造 64 位键。
///
/// `seed` 是**每请求**的（不是全局的），`position` 是该请求的 token 位置。
/// 两者都不参与批次级状态。
#[inline]
pub fn key(seed: u64, position: u64) -> u64 {
    let a = mix64(seed ^ GOLDEN);
    let b = mix64(position.wrapping_mul(GOLDEN_2));
    mix64(a ^ b ^ GOLDEN_3)
}

/// 均匀分布到 `[0, 1)`，只有 2^24 个可能取值，且每个都精确可表示。
#[inline]
pub fn uniform01(seed: u64, position: u64) -> f32 {
    let bits = (mix64(key(seed, position)) >> 40) as u32; // 高 24 位
    bits as f32 * (1.0_f32 / 16_777_216.0_f32) // 2^-24
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_unit_interval() {
        for position in 0..10_000u64 {
            let value = uniform01(0x1234_5678, position);
            assert!(
                (0.0..1.0).contains(&value),
                "uniform01 out of range: {value}"
            );
        }
    }

    #[test]
    fn uniform_values_are_exactly_representable() {
        // 必须是 2^-24 的整数倍，否则说明做了会丢精度的缩放
        let scale = 16_777_216.0_f32;
        for position in 0..1000u64 {
            let value = uniform01(7, position) * scale;
            assert_eq!(value, value.round(), "not a multiple of 2^-24: {value}");
        }
    }

    #[test]
    fn uniform_is_deterministic() {
        for position in [0u64, 1, 42, u64::MAX] {
            assert_eq!(
                uniform01(99, position).to_bits(),
                uniform01(99, position).to_bits()
            );
        }
    }

    #[test]
    fn different_positions_differ() {
        let mut differing = 0;
        for position in 0..1000u64 {
            if uniform01(5, position) != uniform01(5, position + 1) {
                differing += 1;
            }
        }
        // 相邻位置撞车的概率是 2^-24；1000 次里几乎必然全不同
        assert!(
            differing > 995,
            "too many collisions: {differing}/1000 distinguishable"
        );
    }

    #[test]
    fn different_seeds_differ() {
        let mut differing = 0;
        for position in 0..1000u64 {
            if uniform01(1, position) != uniform01(2, position) {
                differing += 1;
            }
        }
        assert!(differing > 995, "seeds barely differ: {differing}/1000");
    }

    #[test]
    fn mean_is_near_half() {
        const N: u64 = 200_000;
        let mut sum = 0.0_f64;
        for position in 0..N {
            sum += uniform01(0xDEAD_BEEF, position) as f64;
        }
        let mean = sum / N as f64;
        // 期望 0.5，标准差 1/sqrt(12N) ≈ 6.5e-4
        assert!((mean - 0.5).abs() < 5e-3, "mean {mean} too far from 0.5");
    }

    /// 均匀性用卡方检验，而不是"某个桶的最大偏差小于某个拍脑袋的数"。
    ///
    /// 16 个桶里单独一个桶偏 2.3 sigma 属于正常波动（约 8% 的概率会发生）；
    /// 拿它当阈值就是在写一个偶尔会红的测试。
    #[test]
    fn buckets_are_roughly_flat() {
        const N: u64 = 200_000;
        const BUCKETS: usize = 16;
        let mut counts = [0u32; BUCKETS];
        for position in 0..N {
            let bucket = (uniform01(12345, position) * BUCKETS as f32) as usize;
            counts[bucket.min(BUCKETS - 1)] += 1;
        }
        let expected = (N / BUCKETS as u64) as f64;
        let chi_square: f64 = counts
            .iter()
            .map(|count| {
                let delta = *count as f64 - expected;
                delta * delta / expected
            })
            .sum();
        // 15 自由度、上尾 0.1% 的临界值约 37.7
        assert!(
            chi_square < 37.7,
            "chi-square {chi_square:.2} suggests non-uniformity (counts {counts:?})"
        );
    }

    #[test]
    fn mix64_is_a_permutation_on_samples() {
        // 换一个角度确认 finalizer 不是一个退化的函数
        let mut seen = std::collections::HashSet::new();
        for position in 0..5000u64 {
            assert!(
                seen.insert(mix64(position)),
                "mix64 collision at {position}"
            );
        }
    }
}
