//! 采样器（D-04）：确定性 + top-k/top-p + logprob 导出。
//!
//! # 本模块最重要的一件事：两个量必须分开
//!
//! | 量 | 定义 | 用途 |
//! | --- | --- | --- |
//! | 采样分布 | `softmax(logits / T)` 截断到 top-k 再截断到 top-p，在**候选集**上归一 | 决定出哪个 token |
//! | 导出的 logprob | `logit[chosen] - logsumexp(全词表)` | 训练端 / PPO 的 KL 与重要性比 |
//!
//! 这两个量**不能**互相代替：
//!   - 用采样分布算 logprob，会把温度和 top-p 的裁剪混进训练信号里，
//!     PPO 的重要性比立刻算错，而且是"看起来只是差一点点"的那种错。
//!   - 用 logprob 的归一化去采样，则 top-p 等于没做。
//!
//! 所以本模块导出的 `logprob` 恒等式是 `values[chosen] - full_logsumexp`，
//! **与 temperature 和 top_p 无关**。这一条有专门的测试盯着。
//!
//! # 确定性
//!
//! 决策路径上不出现 libm：`exp` 用 `detmath::exp_f32`（纯 IEEE 运算），
//! 均匀数用 `rng::uniform01`（计数器式，纯整数）。因此
//! **同 seed + 同输入 ⇒ 跨平台逐位相同的 token 序列**。
//!
//! 诊断量（`candidate_mass`）允许用 libm，因为它不参与决策。

pub mod detmath;
pub mod rng;

use detmath::exp_f32;

/// 候选数上限。超出说明调用方把词表当候选了 —— 那是 host 侧的设计错误，
/// 宁可直接拒绝，也不要在这里悄悄分配一块大缓冲。
pub const MAX_CANDIDATES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// 0 表示贪心（不消耗随机数）。负数非法。
    pub temperature: f32,
    /// (0, 1]；1.0 表示不裁剪。
    pub top_p: f32,
    /// **每请求**的种子。不是全局种子：全局种子会破坏 batch invariance。
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleError {
    EmptyCandidates,
    TooManyCandidates,
    LengthMismatch,
    /// 候选必须按 logit 降序给出（设备侧 top-k 的天然输出）。
    /// 这里选择拒绝而不是内部排序：内部排序要么需要分配，要么改变候选顺序，
    /// 两者都会让"设备与 host 对候选集的理解"出现无声分歧。
    NotSortedDescending,
    InvalidTemperature,
    InvalidTopP,
    /// 出现 NaN 或 +inf。`-inf` 是允许的（pad 槽位的天然表示），权重为 0。
    NonFiniteCandidate,
    /// 至少需要一个有限候选，否则采样无意义。
    NoFiniteCandidate,
    NonFiniteLogsumexp,
}

/// 候选集视图。`values` 是 logits（未归一化），`indices` 是对应 token id。
#[derive(Debug, Clone, Copy)]
pub struct Candidates<'a> {
    pub values: &'a [f32],
    pub indices: &'a [i32],
    /// 全词表的 logsumexp。导出 logprob 时用它，而不是候选集上的归一化。
    pub full_logsumexp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleOutcome {
    pub token: i32,
    /// `values[chosen] - full_logsumexp`。与 T / top_p 无关。
    pub logprob: f32,
    /// top-p 之后保留的候选个数（诊断：越小说明分布越尖，对 T 越敏感）。
    pub kept: u32,
    /// 实际用到的均匀数。贪心时为 0.0 且未被使用。
    pub uniform: f32,
    /// 保留部分占候选集总权重的比例（温度缩放后）。1.0 表示 top-p 没起裁剪作用。
    pub retained_fraction: f32,
}

fn validate(params: &SamplingParams, candidates: &Candidates<'_>) -> Result<(), SampleError> {
    if candidates.values.len() != candidates.indices.len() {
        return Err(SampleError::LengthMismatch);
    }
    if candidates.values.is_empty() {
        return Err(SampleError::EmptyCandidates);
    }
    if candidates.values.len() > MAX_CANDIDATES {
        return Err(SampleError::TooManyCandidates);
    }
    if !params.temperature.is_finite() || params.temperature < 0.0 {
        return Err(SampleError::InvalidTemperature);
    }
    if !params.top_p.is_finite() || params.top_p <= 0.0 || params.top_p > 1.0 {
        return Err(SampleError::InvalidTopP);
    }
    if !candidates.full_logsumexp.is_finite() {
        return Err(SampleError::NonFiniteLogsumexp);
    }

    let mut finite_seen = false;
    let mut previous = f32::INFINITY;
    for &value in candidates.values {
        if value.is_nan() || value == f32::INFINITY {
            return Err(SampleError::NonFiniteCandidate);
        }
        if value.is_finite() {
            finite_seen = true;
        }
        if value > previous {
            return Err(SampleError::NotSortedDescending);
        }
        previous = value;
    }
    if !finite_seen {
        return Err(SampleError::NoFiniteCandidate);
    }
    Ok(())
}

/// 从候选集采一个 token。
///
/// `position` 是该请求当前要生成的 token 位置。它与 `params.seed` 一起唯一确定
/// 这一次抽取 —— 所以同一请求在任何批次组合下都得到同一个结果。
pub fn sample_row(
    params: &SamplingParams,
    candidates: &Candidates<'_>,
    position: u64,
) -> Result<SampleOutcome, SampleError> {
    validate(params, candidates)?;

    let values = candidates.values;
    let indices = candidates.indices;

    // ---- 贪心：不消耗随机数，热路径最短
    if params.temperature == 0.0 {
        return Ok(SampleOutcome {
            token: indices[0],
            logprob: values[0] - candidates.full_logsumexp,
            kept: 1,
            uniform: 0.0,
            retained_fraction: 1.0,
        });
    }

    let inv_t = 1.0 / params.temperature;
    // values 降序 + inv_t > 0 ⇒ 缩放后仍降序，故首元素仍是最大值
    let scaled_max = values[0] * inv_t;

    let weight = |index: usize| -> f32 {
        // -inf 候选（pad 槽位）在这里得到 0；首元素的权重恒为 exp(0) = 1.0
        exp_f32(values[index] * inv_t - scaled_max)
    };

    // 第 1 遍：候选集总权重
    let mut total = 0.0_f32;
    for index in 0..values.len() {
        total += weight(index);
    }
    debug_assert!(
        total >= 1.0,
        "max candidate always has weight 1.0, so total >= 1; got {total}"
    );

    // 第 2 遍：按 top-p 找截断点。break 时的 acc 就是保留权重之和。
    let threshold = params.top_p * total;
    let mut acc = 0.0_f32;
    let mut kept = values.len();
    let mut retained = total;
    for index in 0..values.len() {
        acc += weight(index);
        if acc >= threshold {
            kept = index + 1;
            retained = acc;
            break;
        }
    }

    // 第 3 遍：在保留前缀上按权重走 CDF
    let uniform = rng::uniform01(params.seed, position);
    let target = uniform * retained;
    let mut walk = 0.0_f32;
    let mut chosen = kept - 1; // 兜底：舍入导致走不到时取最后一个保留项
    for index in 0..kept {
        walk += weight(index);
        if walk >= target {
            chosen = index;
            break;
        }
    }

    Ok(SampleOutcome {
        token: indices[chosen],
        logprob: values[chosen] - candidates.full_logsumexp,
        kept: kept as u32,
        uniform,
        retained_fraction: retained / total,
    })
}

/// 候选集覆盖了全词表多少概率质量（`T = 1`）。
///
/// 用途：top-k 截断太多时，采样分布与真实分布已经不是一回事，而这件事
/// **不会报错**，只会让 RL 训练信号悄悄偏掉。所以要么报数、要么别裁。
///
/// 本函数允许用 libm：它是诊断量，不参与决策，因此不需要跨平台逐位一致。
pub fn candidate_mass(values: &[f32], full_logsumexp: f32) -> f32 {
    let Some(&max) = values.first() else {
        return 0.0;
    };
    if !max.is_finite() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for &value in values {
        if value == f32::NEG_INFINITY {
            continue;
        }
        sum += f64::exp((value - max) as f64);
    }
    if sum <= 0.0 {
        return 0.0;
    }
    let log_topk = (max as f64) + sum.ln();
    let mass = (log_topk - full_logsumexp as f64).exp();
    mass.clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates<'a>(values: &'a [f32], indices: &'a [i32], logsumexp: f32) -> Candidates<'a> {
        Candidates {
            values,
            indices,
            full_logsumexp: logsumexp,
        }
    }

    #[test]
    fn greedy_picks_the_max() {
        let values = [3.0, 1.0, 0.5];
        let indices = [10, 20, 30];
        let outcome = sample_row(
            &SamplingParams::greedy(),
            &candidates(&values, &indices, 4.0),
            0,
        )
        .unwrap();
        assert_eq!(outcome.token, 10);
        assert_eq!(outcome.logprob, 3.0 - 4.0);
        assert_eq!(outcome.kept, 1);
    }

    #[test]
    fn validation_rejects_bad_input() {
        let params = SamplingParams::default();
        let values = [1.0, 0.0];

        assert_eq!(
            sample_row(&params, &candidates(&values, &[1], 0.0), 0),
            Err(SampleError::LengthMismatch)
        );
        assert_eq!(
            sample_row(&params, &candidates(&[], &[], 0.0), 0),
            Err(SampleError::EmptyCandidates)
        );
        // 升序：违反降序契约
        let ascending = [0.0, 1.0];
        assert_eq!(
            sample_row(&params, &candidates(&ascending, &[1, 2], 0.0), 0),
            Err(SampleError::NotSortedDescending)
        );
        // NaN 必须拒绝，+inf 必须拒绝
        let nan = [f32::NAN, 0.0];
        assert_eq!(
            sample_row(&params, &candidates(&nan, &[1, 2], 0.0), 0),
            Err(SampleError::NonFiniteCandidate)
        );
        let inf = [f32::INFINITY, 0.0];
        assert_eq!(
            sample_row(&params, &candidates(&inf, &[1, 2], 0.0), 0),
            Err(SampleError::NonFiniteCandidate)
        );
        // 全是 -inf：没有可用候选
        let neg_inf = [f32::NEG_INFINITY; 2];
        assert_eq!(
            sample_row(&params, &candidates(&neg_inf, &[1, 2], 0.0), 0),
            Err(SampleError::NoFiniteCandidate)
        );
        // 温度 / top_p 边界
        for bad in [-1.0_f32, f32::NAN] {
            let p = SamplingParams { temperature: bad, ..Default::default() };
            assert_eq!(
                sample_row(&p, &candidates(&values, &[1, 2], 0.0), 0),
                Err(SampleError::InvalidTemperature)
            );
        }
        for bad in [0.0_f32, -0.5, 1.001, f32::NAN] {
            let p = SamplingParams { top_p: bad, ..Default::default() };
            assert_eq!(
                sample_row(&p, &candidates(&values, &[1, 2], 0.0), 0),
                Err(SampleError::InvalidTopP)
            );
        }
    }

    #[test]
    fn neg_inf_candidates_are_never_selected() {
        let values = [2.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let indices = [7, 8, 9];
        for position in 0..500u64 {
            let outcome = sample_row(
                &SamplingParams { temperature: 1.0, top_p: 1.0, seed: 42 },
                &candidates(&values, &indices, 5.0),
                position,
            )
            .unwrap();
            assert_eq!(outcome.token, 7, "padded slot was selected");
        }
    }

    #[test]
    fn single_candidate_always_wins() {
        for temperature in [0.0_f32, 0.5, 1.0, 7.0] {
            let params = SamplingParams { temperature, top_p: 0.9, seed: 1 };
            for position in 0..100u64 {
                let outcome =
                    sample_row(&params, &candidates(&[1.5], &[99], 2.0), position).unwrap();
                assert_eq!(outcome.token, 99);
                assert_eq!(outcome.logprob, 1.5 - 2.0);
            }
        }
    }

    /// 本模块的核心契约：logprob 是训练端的量，不随采样参数变化。
    #[test]
    fn logprob_is_independent_of_sampling_params() {
        let values = [2.0, 1.5, 1.0, 0.0];
        let indices = [100, 200, 300, 400];
        let full_logsumexp = 3.75;

        let mut seen = std::collections::HashMap::<i32, f32>::new();
        for temperature in [0.0_f32, 0.25, 0.7, 1.0, 2.0, 50.0] {
            for top_p in [0.05_f32, 0.3, 0.75, 1.0] {
                for position in 0..200u64 {
                    let params = SamplingParams { temperature, top_p, seed: 12345 };
                    let outcome = sample_row(&params, &candidates(&values, &indices, full_logsumexp), position)
                        .unwrap();
                    // 恒等式必须严格成立，且只依赖被选中的那个候选
                    let expected = values[indices.iter().position(|&i| i == outcome.token).unwrap()]
                        - full_logsumexp;
                    assert_eq!(
                        outcome.logprob.to_bits(),
                        expected.to_bits(),
                        "logprob drifted: T={temperature} top_p={top_p} pos={position}"
                    );
                    // 同一个 token 在不同参数下导出的 logprob 必须完全相同
                    if let Some(&previous) = seen.get(&outcome.token) {
                        assert_eq!(previous.to_bits(), outcome.logprob.to_bits());
                    } else {
                        seen.insert(outcome.token, outcome.logprob);
                    }
                }
            }
        }
        assert!(seen.len() > 1, "test never exercised more than one token");
    }

    #[test]
    fn top_p_never_keeps_low_mass_tokens() {
        // [0, -10, -20] 在 T=1 下第一个候选独占 ~99.995% 的质量
        let values = [0.0, -10.0, -20.0];
        let indices = [1, 2, 3];
        let params = SamplingParams { temperature: 1.0, top_p: 0.5, seed: 7 };
        for position in 0..1000u64 {
            let outcome = sample_row(&params, &candidates(&values, &indices, -0.000_045), position)
                .unwrap();
            assert_eq!(outcome.token, 1, "top-p kept a negligible-mass token");
            assert_eq!(outcome.kept, 1);
            // retained_fraction 是"保留部分占候选集总权重"的比例，
            // 因为另外两个候选仍占 ~4.5e-5 的权重，所以它略小于 1。
            assert!(
                outcome.retained_fraction > 0.999,
                "retained_fraction {} unexpectedly small",
                outcome.retained_fraction
            );
        }
    }

    /// 均匀分布上 top-p 的截断点必须精确：4 个等概率候选、top_p=0.6 时
    /// 累计概率在第三个处达到 0.75 >= 0.6，所以只保留前三个。
    /// 第四个候选在 10000 次抽样里**一次都不许出现**。
    #[test]
    fn top_p_cut_point_is_exact() {
        let values = [0.0_f32; 4];
        let indices = [1, 2, 3, 4];
        let params = SamplingParams { temperature: 1.0, top_p: 0.6, seed: 999 };

        for position in 0..10_000u64 {
            let outcome = sample_row(&params, &candidates(&values, &indices, 4.0_f32.ln()), position)
                .unwrap();
            assert_eq!(outcome.kept, 3, "cut point should keep exactly three");
            assert_ne!(outcome.token, 4, "token beyond the top-p cut was sampled");
        }

        // 对照组：top_p = 1.0 时第四个候选必须能被抽到，否则上一条测试没有意义
        let loose = SamplingParams { temperature: 1.0, top_p: 1.0, seed: 999 };
        let mut saw_last = false;
        for position in 0..10_000u64 {
            let outcome = sample_row(&loose, &candidates(&values, &indices, 4.0_f32.ln()), position)
                .unwrap();
            if outcome.token == 4 {
                saw_last = true;
                break;
            }
        }
        assert!(saw_last, "with top_p = 1.0 the last candidate must be reachable");
    }

    #[test]
    fn top_p_one_keeps_everything() {
        let values = [0.0, -1.0, -2.0];
        let indices = [1, 2, 3];
        let params = SamplingParams { temperature: 1.0, top_p: 1.0, seed: 7 };
        let outcome = sample_row(&params, &candidates(&values, &indices, 0.0), 3).unwrap();
        assert_eq!(outcome.kept, 3, "top_p = 1.0 must not truncate");
    }

    #[test]
    fn empirical_distribution_matches_analytic() {
        // 温度 1、top_p 1、logits 使得概率为 [0.5, 0.25, 0.25]
        let values = [0.0_f32, -std::f32::consts::LN_2, -std::f32::consts::LN_2];
        let indices = [10, 20, 30];
        let params = SamplingParams { temperature: 1.0, top_p: 1.0, seed: 0xC0FFEE };

        const N: u64 = 60_000;
        let mut counts = [0u64; 3];
        for position in 0..N {
            let outcome =
                sample_row(&params, &candidates(&values, &indices, 0.0), position).unwrap();
            let slot = match outcome.token {
                10 => 0,
                20 => 1,
                30 => 2,
                other => panic!("unexpected token {other}"),
            };
            counts[slot] += 1;
        }

        let expected = [0.5, 0.25, 0.25];
        for (index, want) in expected.iter().enumerate() {
            let got = counts[index] as f64 / N as f64;
            // 标准差 sqrt(p(1-p)/N) ≈ 2e-3（p=0.5）；给 4 sigma
            assert!(
                (got - want).abs() < 0.01,
                "slot {index}: empirical {got:.4} vs analytic {want}"
            );
        }
    }

    #[test]
    fn determinism_across_repeated_calls() {
        let values = [1.0_f32, 0.5, 0.25, 0.125];
        let indices = [1, 2, 3, 4];
        let params = SamplingParams { temperature: 0.8, top_p: 0.95, seed: 0xABCD_EF01 };
        let first: Vec<i32> = (0..1000)
            .map(|position| {
                sample_row(&params, &candidates(&values, &indices, 1.0), position)
                    .unwrap()
                    .token
            })
            .collect();
        let second: Vec<i32> = (0..1000)
            .map(|position| {
                sample_row(&params, &candidates(&values, &indices, 1.0), position)
                    .unwrap()
                    .token
            })
            .collect();
        assert_eq!(first, second);
    }

    /// batch invariance：同一行的结果与"同批里还有谁"完全无关。
    #[test]
    fn batch_composition_does_not_matter() {
        let values = [1.0_f32, 0.5, 0.25];
        let indices = [11, 22, 33];
        let params = SamplingParams { temperature: 1.0, top_p: 0.9, seed: 4242 };

        let alone: Vec<i32> = (0..200)
            .map(|position| {
                sample_row(&params, &candidates(&values, &indices, 1.2), position)
                    .unwrap()
                    .token
            })
            .collect();

        // 模拟"同一行被放在一个大 batch 的不同位置"：其他行用完全不同的候选，
        // 但本行的调用参数一字未变，因此结果必须逐个相同。
        let mut row_outcomes = Vec::new();
        for position in 0..200u64 {
            for other_row in 0..8u64 {
                let other_values = [other_row as f32, 0.0];
                let other_indices = [900 + other_row as i32, 901];
                let other = sample_row(
                    &SamplingParams { temperature: 1.3, top_p: 0.8, seed: 999 + other_row },
                    &candidates(&other_values, &other_indices, 0.0),
                    position,
                );
                assert!(other.is_ok());
            }
            row_outcomes.push(
                sample_row(&params, &candidates(&values, &indices, 1.2), position)
                    .unwrap()
                    .token,
            );
        }
        assert_eq!(alone, row_outcomes);
    }

    #[test]
    fn candidate_mass_reports_coverage() {
        // 只有一个候选且它就是最大值：logsumexp 恰好等于它 ⇒ mass = 1
        assert!((candidate_mass(&[2.0], 2.0) - 1.0).abs() < 1e-6);
        // 全词表 logsumexp 明显更大 ⇒ 候选集只覆盖部分质量
        let mass = candidate_mass(&[0.0, -1.0], 1.5);
        assert!(mass > 0.0 && mass < 1.0, "mass = {mass}");
        // 空候选
        assert_eq!(candidate_mass(&[], 0.0), 0.0);
    }

    #[test]
    fn total_weight_is_at_least_one() {
        // 首元素权重恒为 exp(0) = 1，因此 total >= 1，除法不会退化
        let values = [0.0_f32, -1000.0, -1000.0];
        let outcome = sample_row(
            &SamplingParams { temperature: 1.0, top_p: 1.0, seed: 1 },
            &candidates(&values, &[1, 2, 3], 0.5),
            0,
        )
        .unwrap();
        assert_eq!(outcome.token, 1);
        // -1000 的候选权重下溢到 0，于是保留部分就是全部权重
        assert_eq!(outcome.retained_fraction, 1.0);
    }
}
