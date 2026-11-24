//! E-04：三级回收的决策 + 抢占率记账。
//!
//! # 三级的代价不在一个量级上
//!
//! | 级别 | 代价 | 可逆 | 触发条件 |
//! | --- | --- | --- | --- |
//! | 驱逐 | 0（丢掉没人引用的缓存条目） | 可重算，但只是"以后可能要" | 总是先做 |
//! | 换出 | 一次 D2H + 一次 H2D 的带宽 | 可逆，KV 还在 | 驱逐不够时 |
//! | 抢占 | **整段 prefill 重算** | 不可逆，KV 没了 | 换出空间也不够时 |
//!
//! 所以顺序是硬顺序，不是启发式：**只要还有一个可驱逐的缓存条目，就绝不允许抢占**。
//! 4k token 的 prefill 是毫秒级计算，而换出 100 MB 在 2 TB/s 上是几十微秒 —— 差两个数量级。
//!
//! # 一个不显然的结论：换出要保护"贵的"，抢占要牺牲"便宜的"
//!
//! 换出本身不花计算，所以应该用它去**保住重算代价高的序列**；
//! 反过来说，真要抢占时，应该挑重算代价最低的。
//!
//! 于是两级的挑选顺序是**相反**的：
//!
//! - 换出：按 `tokens` **降序**（先保护最贵的）
//! - 抢占：按 `tokens` **升序**（先牺牲最便宜的）
//!
//! 如果两级用同一个顺序（比如都按 LRU），换出就会先把便宜的保护起来，
//! 留下贵的等着被抢占 —— 恰好和抢占总代价最小化的目标相反。
//!
//! # 抢占率必须按窗口看，累计率会撒谎
//!
//! 判据是 `Recompute Tokens / Total Decoded Tokens < 5%`。
//! 但累计率天然掩盖尖峰：一次 10% 的抖动放进 10 倍长的历史里就是 1%，
//! "看起来达标"。所以这里同时给累计率与**最差窗口率**，后者才是会让人半夜被叫起来的那个数。

use std::fmt;

/// DESIGN §5.2 的目标。
pub const TARGET_PREEMPT_RATE: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(pub u64);

impl fmt::Display for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// 一个可以被换出或抢占的活跃序列。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub sequence: SequenceId,
    /// 这个序列已占用的 token 数 = 抢占后要重算的量。
    pub tokens: u32,
    /// 释放它能腾出的块数。0 表示腾不出东西，不该被选中。
    pub blocks: u32,
}

/// 此刻有哪些资源可以回收。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budget {
    /// 驱逐能腾出的块数（缓存树里 `refcount == 0` 的叶节点）。
    pub evictable_blocks: u32,
    /// 换出能腾出的块数上限（受 pinned host 内存限制）。
    pub swappable_blocks: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReclaimPlan {
    pub evict_blocks: u32,
    pub swap_blocks: u32,
    pub swapped: Vec<SequenceId>,
    pub preempted: Vec<SequenceId>,
    pub preempted_blocks: u32,
    /// 这个计划要付的重算代价，直接喂给 [`RecomputeLedger`]。
    pub recompute_tokens: u32,
}

impl ReclaimPlan {
    /// 还差多少块。
    pub fn shortfall(&self, needed: u32) -> u32 {
        needed.saturating_sub(
            self.evict_blocks
                .saturating_add(self.swap_blocks)
                .saturating_add(self.preempted_blocks),
        )
    }

    /// 是否真的够用。**不够必须报出来**，不能悄悄返回一个看起来成功的计划。
    pub fn is_sufficient(&self, needed: u32) -> bool {
        self.shortfall(needed) == 0
    }

    /// 三级是否按顺序使用：先驱逐、再换出、最后才抢占。
    ///
    /// 用 `>=` 而不是 `==`：换出与抢占都是整序列操作，会溢出所需块数。
    pub fn respects_escalation(&self, budget: &Budget, needed: u32) -> bool {
        if self.preempted.is_empty() {
            return true;
        }
        let evict_exhausted = self.evict_blocks >= budget.evictable_blocks.min(needed);
        let after_evict = needed.saturating_sub(self.evict_blocks);
        let swap_exhausted = self.swap_blocks >= budget.swappable_blocks.min(after_evict);
        evict_exhausted && swap_exhausted
    }
}

/// 制定回收计划。
///
/// 排序带确定的 id 平局裁决，所以同样的输入永远得到同样的计划 ——
/// 否则"同 seed 同轨迹"这条契约会被回收策略悄悄破坏。
pub fn plan_reclaim(needed: u32, budget: Budget, candidates: &[Candidate]) -> ReclaimPlan {
    let mut plan = ReclaimPlan::default();
    let mut remaining = needed;

    // ---- 第 1 级：驱逐
    let evict = remaining.min(budget.evictable_blocks);
    plan.evict_blocks = evict;
    remaining -= evict;
    if remaining == 0 {
        return plan;
    }

    // ---- 第 2 级：换出。按重算代价降序 —— 保住贵的。
    let mut by_cost_desc: Vec<usize> = (0..candidates.len())
        .filter(|&index| candidates[index].blocks > 0)
        .collect();
    by_cost_desc.sort_by(|&a, &b| {
        candidates[b]
            .tokens
            .cmp(&candidates[a].tokens)
            .then(candidates[a].sequence.cmp(&candidates[b].sequence))
    });

    let mut swapped = vec![false; candidates.len()];
    let mut swap_budget = budget.swappable_blocks;
    for index in by_cost_desc {
        if remaining == 0 || swap_budget == 0 {
            break;
        }
        let candidate = &candidates[index];
        if candidate.blocks > swap_budget {
            // 换出只能整序列做，塞不下就跳过（留给后面的级别或下一轮）
            continue;
        }
        swap_budget -= candidate.blocks;
        remaining = remaining.saturating_sub(candidate.blocks);
        plan.swap_blocks += candidate.blocks;
        plan.swapped.push(candidate.sequence);
        swapped[index] = true;
    }
    if remaining == 0 {
        return plan;
    }

    // ---- 第 3 级：抢占。按重算代价升序 —— 先牺牲便宜的。
    let mut by_cost_asc: Vec<usize> = (0..candidates.len())
        .filter(|&index| candidates[index].blocks > 0 && !swapped[index])
        .collect();
    by_cost_asc.sort_by(|&a, &b| {
        candidates[a]
            .tokens
            .cmp(&candidates[b].tokens)
            .then(candidates[a].sequence.cmp(&candidates[b].sequence))
    });

    for index in by_cost_asc {
        if remaining == 0 {
            break;
        }
        let candidate = &candidates[index];
        remaining = remaining.saturating_sub(candidate.blocks);
        plan.preempted_blocks += candidate.blocks;
        plan.recompute_tokens += candidate.tokens;
        plan.preempted.push(candidate.sequence);
    }

    plan
}

/// 一个记账窗口。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Window {
    pub decoded: u64,
    pub recompute: u64,
}

impl Window {
    /// 窗口内没发生解码时返回 `None`。
    ///
    /// 返回 `0.0` 会是撒谎：分母为零的"完美达标"和真正的零抢占长得一模一样，
    /// 而前者只说明这一窗什么都没跑。
    pub fn rate(&self) -> Option<f64> {
        (self.decoded > 0).then(|| self.recompute as f64 / self.decoded as f64)
    }
}

/// 抢占率记账：累计 + 滑动窗口。
#[derive(Debug, Clone)]
pub struct RecomputeLedger {
    windows: Vec<Window>,
    cursor: usize,
    cumulative: Window,
}

impl RecomputeLedger {
    pub fn new(windows: usize) -> Self {
        assert!(windows > 0, "need at least one window");
        Self {
            windows: vec![Window::default(); windows],
            cursor: 0,
            cumulative: Window::default(),
        }
    }

    pub fn record_decoded(&mut self, tokens: u32) {
        self.cumulative.decoded += u64::from(tokens);
        self.windows[self.cursor].decoded += u64::from(tokens);
    }

    pub fn record_recompute(&mut self, tokens: u32) {
        self.cumulative.recompute += u64::from(tokens);
        self.windows[self.cursor].recompute += u64::from(tokens);
    }

    /// 把一次回收计划的重算代价记进来。
    pub fn record_plan(&mut self, plan: &ReclaimPlan) {
        self.record_recompute(plan.recompute_tokens);
    }

    /// 翻到下一个窗口（上一个窗口的数据被覆盖）。
    pub fn advance_window(&mut self) {
        self.cursor = (self.cursor + 1) % self.windows.len();
        self.windows[self.cursor] = Window::default();
    }

    pub fn cumulative(&self) -> Window {
        self.cumulative
    }

    pub fn cumulative_rate(&self) -> Option<f64> {
        self.cumulative.rate()
    }

    /// 最差窗口的抢占率。**这才是会让人半夜被叫起来的那个数。**
    pub fn worst_window_rate(&self) -> Option<f64> {
        self.windows
            .iter()
            .filter_map(Window::rate)
            .fold(None, |worst: Option<f64>, rate| {
                Some(worst.map_or(rate, |current| current.max(rate)))
            })
    }

    /// 累计与最差窗口都在目标内。
    pub fn within_target(&self, target: f64) -> bool {
        self.cumulative_rate().is_none_or(|rate| rate < target)
            && self.worst_window_rate().is_none_or(|rate| rate < target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: u64, tokens: u32, blocks: u32) -> Candidate {
        Candidate {
            sequence: SequenceId(id),
            tokens,
            blocks,
        }
    }

    #[test]
    fn eviction_alone_satisfies_the_request() {
        let budget = Budget {
            evictable_blocks: 10,
            swappable_blocks: 10,
        };
        let candidates = [candidate(1, 100, 4)];
        let plan = plan_reclaim(6, budget, &candidates);
        assert_eq!(plan.evict_blocks, 6);
        assert_eq!(plan.swap_blocks, 0);
        assert!(
            plan.preempted.is_empty(),
            "must not preempt while evictable remains"
        );
        assert_eq!(plan.recompute_tokens, 0);
        assert!(plan.is_sufficient(6));
        assert!(plan.respects_escalation(&budget, 6));
    }

    #[test]
    fn swap_only_happens_after_eviction_is_exhausted() {
        let budget = Budget {
            evictable_blocks: 2,
            swappable_blocks: 10,
        };
        let candidates = [candidate(1, 100, 6)];
        let plan = plan_reclaim(5, budget, &candidates);
        assert_eq!(plan.evict_blocks, 2);
        // 换出是整序列操作：候选占 6 块，只缺 3 块也要整块换出（溢出 3 块）
        assert_eq!(plan.swap_blocks, 6);
        assert!(plan.preempted.is_empty());
        assert!(plan.is_sufficient(5));
        assert!(plan.respects_escalation(&budget, 5));
    }

    /// ★ 不显然的那条：换出保护贵的，抢占牺牲便宜的。
    #[test]
    fn swap_protects_expensive_sequences_and_preemption_sacrifices_cheap_ones() {
        let budget = Budget {
            evictable_blocks: 0,
            swappable_blocks: 4, // 只够换出一个
        };
        let candidates = [
            candidate(1, 10, 4),   // 便宜
            candidate(2, 4000, 4), // 贵
            candidate(3, 200, 4),
        ];
        // 需要 8 块：换出 4 块 + 抢占一个 4 块的序列
        let plan = plan_reclaim(8, budget, &candidates);

        assert_eq!(
            plan.swapped,
            vec![SequenceId(2)],
            "the most expensive sequence must be swapped out, not left to be preempted"
        );
        assert_eq!(
            plan.preempted,
            vec![SequenceId(1)],
            "the cheapest sequence must be preempted"
        );
        assert_eq!(
            plan.recompute_tokens, 10,
            "recompute cost must be the cheap one"
        );
        assert!(plan.is_sufficient(8));
    }

    #[test]
    fn preemption_order_is_ascending_by_cost() {
        let budget = Budget::default();
        let candidates = [
            candidate(1, 900, 1),
            candidate(2, 50, 1),
            candidate(3, 300, 1),
        ];
        let plan = plan_reclaim(2, budget, &candidates);
        assert_eq!(plan.preempted, vec![SequenceId(2), SequenceId(3)]);
        assert_eq!(plan.recompute_tokens, 350);
    }

    #[test]
    fn ties_are_broken_by_sequence_id_so_plans_are_deterministic() {
        let budget = Budget::default();
        let candidates = [
            candidate(30, 100, 1),
            candidate(10, 100, 1),
            candidate(20, 100, 1),
        ];
        let first = plan_reclaim(2, budget, &candidates);
        let second = plan_reclaim(2, budget, &candidates);
        assert_eq!(first, second);
        assert_eq!(first.preempted, vec![SequenceId(10), SequenceId(20)]);
    }

    #[test]
    fn shortfall_is_reported_when_everything_runs_out() {
        let budget = Budget {
            evictable_blocks: 1,
            swappable_blocks: 1,
        };
        let candidates = [candidate(1, 100, 2)];
        let plan = plan_reclaim(10, budget, &candidates);
        assert_eq!(plan.evict_blocks, 1);
        // 换出预算只有 1 块，而候选占 2 块 —— 整序列塞不下，于是跳过换出
        assert_eq!(plan.swap_blocks, 0);
        assert_eq!(plan.preempted_blocks, 2);
        assert_eq!(plan.shortfall(10), 7);
        assert!(!plan.is_sufficient(10), "insufficiency must not be silent");
    }

    #[test]
    fn zero_block_candidates_are_ignored() {
        let budget = Budget::default();
        let candidates = [candidate(1, 5, 0), candidate(2, 100, 2)];
        let plan = plan_reclaim(1, budget, &candidates);
        assert_eq!(plan.preempted, vec![SequenceId(2)]);
    }

    #[test]
    fn a_sequence_is_never_both_swapped_and_preempted() {
        let budget = Budget {
            evictable_blocks: 0,
            swappable_blocks: 8,
        };
        let candidates = [candidate(1, 10, 4), candidate(2, 4000, 4)];
        let plan = plan_reclaim(12, budget, &candidates);
        for id in &plan.preempted {
            assert!(
                !plan.swapped.contains(id),
                "{id} was both swapped and preempted"
            );
        }
        assert!(plan.respects_escalation(&budget, 12));
    }

    #[test]
    fn overshoot_is_possible_because_preemption_is_whole_sequence() {
        // 需要 1 块，但候选序列各占 4 块 —— 只能整序列抢占，于是腾出 4 块
        let budget = Budget::default();
        let candidates = [candidate(1, 100, 4)];
        let plan = plan_reclaim(1, budget, &candidates);
        assert_eq!(plan.preempted_blocks, 4);
        assert!(plan.is_sufficient(1));
    }

    // ---------------- 记账 ----------------

    #[test]
    fn rate_needs_a_denominator() {
        let mut ledger = RecomputeLedger::new(4);
        assert_eq!(ledger.cumulative_rate(), None, "no decode yet -> no rate");
        assert_eq!(ledger.worst_window_rate(), None);
        assert!(ledger.within_target(TARGET_PREEMPT_RATE));

        ledger.record_decoded(200);
        assert_eq!(ledger.cumulative_rate(), Some(0.0));
        ledger.record_recompute(10);
        assert_eq!(ledger.cumulative_rate(), Some(0.05));
        assert!(!ledger.within_target(TARGET_PREEMPT_RATE), "5% is not < 5%");
    }

    /// ★ 累计率会撒谎：这一条是本模块存在的第二个理由。
    #[test]
    fn cumulative_rate_hides_a_spike_that_the_window_metric_catches() {
        // 窗口数要多于记录轮数，否则最后一轮 advance 会把尖峰那一窗覆盖掉
        let mut ledger = RecomputeLedger::new(16);
        for window in 0..10 {
            for _ in 0..10 {
                ledger.record_decoded(100); // 每窗 1000 token
            }
            if window == 0 {
                ledger.record_recompute(100); // 第一窗 10%
            }
            ledger.advance_window();
        }

        let cumulative = ledger.cumulative_rate().unwrap();
        assert!(
            cumulative < TARGET_PREEMPT_RATE,
            "cumulative looks fine: {cumulative:.4}"
        );
        let worst = ledger.worst_window_rate().unwrap();
        assert!(
            worst > TARGET_PREEMPT_RATE,
            "worst window must expose the spike: {worst:.4}"
        );
        assert!(
            !ledger.within_target(TARGET_PREEMPT_RATE),
            "a spike inside the target window must fail the gate"
        );
    }

    #[test]
    fn advancing_windows_discards_old_data() {
        let mut ledger = RecomputeLedger::new(2);
        ledger.record_decoded(100);
        ledger.record_recompute(100); // 100%
        assert_eq!(ledger.worst_window_rate(), Some(1.0));

        ledger.advance_window();
        ledger.record_decoded(100);
        // 窗口容量 2：再翻一次才会覆盖掉那个 100% 的窗口
        ledger.advance_window();
        ledger.record_decoded(100);
        assert_eq!(
            ledger.worst_window_rate(),
            Some(0.0),
            "old window should have been recycled"
        );
        // 但累计率永远记得
        assert_eq!(ledger.cumulative_rate(), Some(100.0 / 300.0));
    }

    #[test]
    fn ledger_accepts_a_plan_directly() {
        let mut ledger = RecomputeLedger::new(4);
        ledger.record_decoded(1000);
        let plan = plan_reclaim(1, Budget::default(), &[candidate(7, 20, 4)]);
        ledger.record_plan(&plan);
        assert_eq!(ledger.cumulative().recompute, 20);
    }
}
