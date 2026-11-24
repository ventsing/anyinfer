//! 确定性的单精度指数函数。
//!
//! **为什么不用 `f32::exp`：**
//! 采样决策必须可复现。`exp` 在 Rust 里转发到平台的 libm，而 libm 允许
//! 最后一位 ulp 的实现差异（不同 glibc 版本、不同 CPU 上的不同代码路径都会）。
//! 一次采样的概率边界差 1 ulp，就可能在高熵 token 上翻转选择 ——
//! 于是"同 seed 同轨迹"这条 RSI 的基础契约在换机器后失效，
//! 而且失效方式极难发现：99.999% 的 token 一样，偶尔一个不一样。
//!
//! 本实现只用 IEEE-754 保证正确舍入的运算（`+ - * /`、`floor`）与整数位操作，
//! 因此**跨平台逐位一致**。这正是 DESIGN §6.4「确定性」要求的形态。
//!
//! 算法取自 Cephes 的单精度 `expf`（6 阶 minimax 多项式 + 2 的幂次缩放），
//! 实测相对误差 < 1e-6，对本用途（softmax 权重）绰绰有余。

/// 1 / ln(2)，把以 e 为底换成以 2 为底。
const INV_LN2: f32 = 1.442_695_1_f32;
/// ln(2) 的高位部分，与低位配合在两段相减中减少抵消误差。
const LN2_HI: f32 = 6.931_457_5e-1_f32;
const LN2_LO: f32 = 1.428_606_8e-6_f32;

/// 上溢/下溢截断点：exp(88.7228..) 已超出 f32 范围。
const MAX_LOG: f32 = 88.722_84_f32;
const MIN_LOG: f32 = -103.972_08_f32;

/// 2^-24 附近的多项式系数（Cephes）。
const P: [f32; 6] = [
    1.987_569_2e-4_f32,
    1.398_199_9e-3_f32,
    8.333_451_9e-3_f32,
    4.166_579_6e-2_f32,
    1.666_666_5e-1_f32,
    5.000_000_1e-1_f32,
];

/// 确定性 `exp`。输入 ±inf / NaN 时返回 IEEE 语义的结果，不做静默兜底。
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > MAX_LOG {
        return f32::INFINITY;
    }
    if x < MIN_LOG {
        return 0.0;
    }

    // n = round(x / ln2)，用 +0.5 后向下取整实现（round-half-up，行为固定）
    let n = (x * INV_LN2 + 0.5_f32).floor();
    // 两段相减保留精度：一次减 n*ln2 会在 |x| 较大时吃掉有效位
    let r = x - n * LN2_HI - n * LN2_LO;

    // Horner 求值 Q(r) = P[0] + P[1]·r + ... + P[5]·r^5，其中
    //   exp(r) = 1 + r + r²·Q(r)
    // 所以末项必须是 poly * r² —— 写成 poly * r 会得到
    //   poly·r + r + 1 ≈ 0.55 而不是 0.7357（exp(-0.307)），误差 27%。
    // 注意：Horner 是对 r 迭代（不是 r²），最后才乘 r²。
    // P 是按次数从高到低排的：r⁵, r⁴, r³, r², r, 1
    let rr = r * r;
    let mut poly = P[0];
    poly = poly * r + P[1];
    poly = poly * r + P[2];
    poly = poly * r + P[3];
    poly = poly * r + P[4];
    poly = poly * r + P[5];
    let y = poly * rr + r + 1.0_f32;

    // 乘上 2^n。
    //
    // 这里刻意升到 f64 再落回 f32，而不是直接改 f32 的指数域：
    //   - n < -126 时结果是**次正规** f32。改指数域会静默丢掉这些值
    //     （e^{-88} ≈ 6e-39 是能表示的，却被退成 0），于是 exp 在
    //     -87.7 附近不再单调。
    //   - f64 乘、f64→f32 转换都是 IEEE 规定正确舍入的，因此这条路径
    //     仍然跨平台逐位一致，没有引入 libm。
    if n == 0.0_f32 {
        return y;
    }
    let exponent = n as i32;
    let scaled = (y as f64) * pow2_f64(exponent);
    scaled as f32
}

/// 精确的 2^n（n 为整数）。用位构造而不是 `powi`，避免依赖其实现。
/// n 落在 f64 可精确表示的范围内时结果精确。
#[inline]
fn pow2_f64(n: i32) -> f64 {
    if n > 1023 {
        return f64::INFINITY;
    }
    if n >= -1022 {
        return f64::from_bits(((n + 1023) as u64) << 52);
    }
    if n >= -1074 {
        // f64 次正规区间：2^n 仍可精确表示
        return f64::from_bits(1_u64 << ((n + 1074) as u32));
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_exp(x: f64) -> f64 {
        // 测试里用标准库算"真值"是可以的：被测代码不依赖它
        x.exp()
    }

    #[test]
    fn matches_libm_within_tolerance() {
        // 采样只用 x <= 0 的区间（logits 减去最大值），但两侧都覆盖一下
        let mut worst_rel = 0.0_f64;
        let mut worst_at = 0.0_f32;
        let mut x = -103.0_f32;
        while x <= 0.0 {
            let got = super::exp_f32(x) as f64;
            let want = f64_exp(x as f64);
            if want > 1e-30 {
                let rel = ((got - want) / want).abs();
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_at = x;
                }
            }
            x += 0.01;
        }
        assert!(
            worst_rel < 1e-6,
            "worst relative error {worst_rel:e} at x={worst_at}"
        );
    }

    #[test]
    fn exact_at_zero() {
        assert_eq!(super::exp_f32(0.0), 1.0);
        // -0.0 必须与 0.0 同结果
        assert_eq!(super::exp_f32(-0.0), 1.0);
    }

    #[test]
    fn saturation() {
        assert_eq!(super::exp_f32(MIN_LOG - 1.0), 0.0);
        assert!(super::exp_f32(MAX_LOG + 1.0).is_infinite());
        assert!(super::exp_f32(f32::NAN).is_nan());
    }

    #[test]
    fn monotone_non_decreasing() {
        let mut previous = -1.0_f32;
        let mut x = -100.0_f32;
        while x <= 0.0 {
            let value = super::exp_f32(x);
            assert!(value >= previous, "exp not monotone at x={x}");
            previous = value;
            x += 0.01;
        }
    }

    #[test]
    fn deterministic() {
        for x in [-50.0_f32, -1.0, -0.5, -1e-8, 0.0] {
            assert_eq!(super::exp_f32(x).to_bits(), super::exp_f32(x).to_bits());
        }
    }

    #[test]
    fn all_results_are_probabilities() {
        // exp(x) for x <= 0 必须落在 (0, 1]
        let mut x = -100.0_f32;
        while x <= 0.0 {
            let value = super::exp_f32(x);
            assert!((0.0..=1.0).contains(&value), "exp({x}) = {value} out of (0,1]");
            x += 0.01;
        }
    }
}
