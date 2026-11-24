//! E-04 的端到端仿真：把 E-01..E-04 串起来跑，让**指标自己回答**驱逐顶不顶用。
//!
//! 这里能验证的只是决策层；真正的换出（D2H 到 pinned host）需要设备，
//! 所以仿真只建模驱逐与抢占 —— 换出的策略在 `reclaim.rs` 的单元测试里覆盖。
//!
//! # 建模约定（说清楚，否则数字没有意义）
//!
//! **被抢占的请求会重试，而不是被丢掉。**
//! 第一版仿真把抢占当成"请求消失"，结果分母塌了、比率飙到 160%。
//! 而 >100% 的抢占率在真实系统里不可能 —— 它只说明"这份负载根本没被服务"，
//! 说明模型错了，不是系统糟了。
//!
//! 改成重试之后：分子是"算过又作废、需要再算一遍的 token"，分母是"引擎总共算过的
//! token"，比值才是 DESIGN §5.2 想要的那个数，且天然满足 `rate <= 1`
//! （每个作废的 token 都曾经被计入过 decoded）。这条不等式有专门的测试盯着。
//!
//! 重试次数有上限。无限重试在容量严重不足时会变成永不停机的抖动，仿真永远跑不完——
//! 那不是模型的缺陷，是真实系统也该给这类请求一个放弃判据。

use crate::arena::{Arena, ArenaConfig};
use crate::prefix::{NodeId, PrefixCache};
use crate::reclaim::{plan_reclaim, Budget, Candidate, ReclaimPlan, RecomputeLedger, SequenceId};
use crate::sequence::KvSequence;

const WIDTH: u32 = 4;
const PREFIXES: u32 = 3;
const PREFIX_TOKENS: usize = 8;
const JOBS: usize = 40;
const GENERATE: usize = 12;
const MAX_LIVE: usize = 4;
const MAX_ATTEMPTS: u8 = 4;
const MAX_STEPS: usize = 50_000;

struct Live {
    seq: KvSequence,
    remaining: usize,
    job: usize,
}

struct Outcome {
    decoded: u64,
    recompute: u64,
    preemptions: u32,
    evictions: u32,
    finished: u32,
    gave_up: u32,
    /// 步数预算用尽而退出。此时 `finished + gave_up` 必然小于作业数，
    /// 不是记账错了，是这份负载在这个容量下根本没跑完。
    truncated: bool,
    ledger: RecomputeLedger,
}

impl Outcome {
    fn rate(&self) -> Option<f64> {
        self.ledger.cumulative_rate()
    }
}

fn run(blocks: u32, eviction_enabled: bool) -> Outcome {
    let arena = Arena::new(ArenaConfig {
        num_blocks: blocks,
        block_size: WIDTH,
        reserved_blocks: 1,
    })
    .unwrap();
    let mut cache = PrefixCache::new(arena);

    let prefixes: Vec<NodeId> = (0..PREFIXES)
        .map(|k| {
            let tokens: Vec<u32> = (0..PREFIX_TOKENS as u32).map(|i| k * 100 + i).collect();
            cache.insert(&tokens).unwrap().node
        })
        .collect();

    let mut pending: Vec<usize> = (0..JOBS).rev().collect();
    let mut attempts = [0u8; JOBS];
    let mut live: Vec<Live> = Vec::new();

    let mut ledger = RecomputeLedger::new(64);
    let mut decoded = 0u64;
    let mut recompute = 0u64;
    let mut preemptions = 0u32;
    let mut evictions = 0u32;
    let mut finished = 0u32;
    let mut gave_up = 0u32;

    let mut truncated = true;
    for step in 0..MAX_STEPS {
        if pending.is_empty() && live.is_empty() {
            truncated = false;
            break;
        }
        if step % 8 == 0 {
            ledger.advance_window();
        }

        // ---- 准入
        while live.len() < MAX_LIVE {
            let Some(job) = pending.pop() else { break };
            let node = prefixes[job % prefixes.len()];
            match cache.fork(node, PREFIX_TOKENS + GENERATE) {
                Ok(seq) => live.push(Live {
                    seq,
                    remaining: GENERATE,
                    job,
                }),
                Err(_) => {
                    // 连共享前缀都放不下：等下一轮
                    pending.push(job);
                    break;
                }
            }
        }

        // ---- 每个活跃序列推进一个 token
        let mut index = 0;
        while index < live.len() {
            if live[index].remaining == 0 {
                let entry = live.remove(index);
                cache.join(entry.seq).unwrap();
                finished += 1;
                continue;
            }

            match live[index].seq.append_token(cache.arena_mut_for_test()) {
                Ok(_) => {
                    ledger.record_decoded(1);
                    decoded += 1;
                    live[index].remaining -= 1;
                    index += 1;
                }
                Err(_) => {
                    let victims =
                        reclaim(&mut cache, &live, eviction_enabled, &mut evictions);
                    let freed = !victims.is_empty();
                    for slot in victims {
                        if slot >= live.len() {
                            continue;
                        }
                        let entry = live.remove(slot);
                        // ★ 只算**本引擎生成过**的 token。
                        // `seq.len()` 里含着从缓存白拿的共享前缀，那些重试时照样命中，
                        // 一分计算都不用付。把它们算进分子会让指标凭空翻几倍。
                        let lost = (entry.seq.len() - entry.seq.shared_tokens()) as u32;
                        cache.join(entry.seq).unwrap();
                        // 这些 token 已经算过，现在作废 —— 重试时要再算一遍
                        ledger.record_recompute(lost);
                        recompute += u64::from(lost);
                        preemptions += 1;
                        if attempts[entry.job] + 1 < MAX_ATTEMPTS {
                            attempts[entry.job] += 1;
                            pending.push(entry.job);
                        } else {
                            gave_up += 1;
                        }
                    }
                    // 一个都没腾出来就往后走，避免原地死循环
                    index += 1;
                    let _ = freed;
                }
            }
        }
    }

    Outcome {
        decoded,
        recompute,
        preemptions,
        evictions,
        finished,
        gave_up,
        truncated,
        ledger,
    }
}

/// 三级回收：先驱逐，再（换出，仿真不建模），最后抢占。返回要抢占的 `live` 下标。
fn reclaim(
    cache: &mut PrefixCache,
    live: &[Live],
    eviction_enabled: bool,
    evictions: &mut u32,
) -> Vec<usize> {
    let candidates: Vec<Candidate> = live
        .iter()
        .enumerate()
        .map(|(slot, entry)| Candidate {
            sequence: SequenceId(slot as u64),
            tokens: entry.seq.len() as u32,
            blocks: entry.seq.block_count() as u32,
        })
        .collect();

    let plan: ReclaimPlan = plan_reclaim(
        1,
        Budget {
            evictable_blocks: if eviction_enabled {
                cache.evictable_blocks()
            } else {
                0
            },
            swappable_blocks: 0,
        },
        &candidates,
    );

    for _ in 0..plan.evict_blocks {
        if cache.evict_leaf_lru().unwrap().is_some() {
            *evictions += 1;
        }
    }

    plan.preempted.iter().map(|id| id.0 as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reclaim::TARGET_PREEMPT_RATE;

    /// 容量充足时不该发生任何抢占 —— 也顺带证明这套策略不是"总是抢占"。
    #[test]
    fn generous_capacity_preempts_nothing() {
        let outcome = run(256, true);
        assert_eq!(outcome.finished, JOBS as u32, "not every job completed");
        assert_eq!(outcome.decoded, (JOBS * GENERATE) as u64);
        assert_eq!(outcome.preemptions, 0, "preempted with plenty of blocks");
        assert_eq!(outcome.recompute, 0);
        assert_eq!(outcome.rate(), Some(0.0));
        assert!(outcome.ledger.within_target(TARGET_PREEMPT_RATE));
    }

    /// 容量紧张时指标必须**动**起来，否则它就不是在测东西。
    #[test]
    fn tight_capacity_makes_the_metric_move() {
        let tight = run(10, true);
        let generous = run(256, true);
        assert!(
            tight.recompute > generous.recompute,
            "metric did not respond to pressure: tight={} generous={}",
            tight.recompute,
            generous.recompute
        );
    }

    /// ★★ 表征测试：**纯 LRU 驱逐在这份负载下是净负的。**
    ///
    /// 这是仿真写出来之后才发现的，不是我预先想到的：
    /// 三块被缓存的是共享系统提示词，而 LRU 只看"最久没被命中"，
    /// 于是它优先踢掉恰好是**价值最高**的那些节点 ——
    /// 腾出 2 块，却让后续十几个作业失去 8 token 的免费前缀，每个都要多占 2 块。
    ///
    /// 数字（12 块，同容量）：
    /// - 开驱逐：服务 45 token，重算 9 token
    /// - 关驱逐：服务 200 token，重算 32 token
    ///
    /// 所以"驱逐总是好的"不成立。真正该做的是按**复用价值**而不是**最近使用**挑牺牲者，
    /// 那是 E-07（见 TODO）。这条测试把现状钉住：谁改了驱逐策略，这里会告诉他结论变了。
    ///
    /// 顺带也钉住一条方法学前提：两个 run 必须服务同样多的工作，比率才可比。
    #[test]
    fn pure_lru_eviction_is_net_negative_when_the_cache_holds_shared_prefixes() {
        let with = run(12, true);
        let without = run(12, false);
        assert!(
            with.evictions > 0 && without.evictions == 0,
            "eviction flag was not actually wired: with={} without={}",
            with.evictions,
            without.evictions
        );
        assert!(
            with.decoded < without.decoded,
            "LRU eviction stopped being net-negative (with={} without={}) — \
             either the policy got smarter or the workload changed; re-derive E-07",
            with.decoded,
            without.decoded
        );
    }

    /// 保住那两个数字：结论变了要有人知道。
    #[test]
    fn the_net_negative_eviction_measurement_is_pinned() {
        let with = run(12, true);
        let without = run(12, false);
        assert_eq!((with.decoded, with.recompute), (45, 9));
        assert_eq!((without.decoded, without.recompute), (200, 32));
    }

    /// 比率必须恒 <= 1：每个作废的 token 都曾经被计入 decoded。
    /// 超过 1 就说明记账把同一份工作数了两次。
    #[test]
    fn rate_never_exceeds_one() {
        for blocks in [8u32, 10, 12, 14, 16, 18, 20, 24, 32] {
            let outcome = run(blocks, true);
            let rate = outcome.rate().unwrap();
            assert!(
                rate <= 1.0,
                "rate {rate:.3} > 1 at {blocks} blocks: recompute={} decoded={}",
                outcome.recompute,
                outcome.decoded
            );
        }
    }

    #[test]
    fn accounting_is_self_consistent_under_every_pressure() {
        for blocks in [8u32, 12, 16, 24, 64, 256] {
            let outcome = run(blocks, true);
            assert_eq!(
                outcome.ledger.cumulative().recompute,
                outcome.recompute,
                "ledger drifted at {blocks} blocks"
            );
            assert_eq!(outcome.ledger.cumulative().decoded, outcome.decoded);
            if !outcome.truncated {
                assert_eq!(outcome.finished + outcome.gave_up, JOBS as u32);
            }
            if outcome.recompute > 0 {
                assert!(outcome.preemptions > 0, "recompute without preemption");
            }
            if outcome.preemptions == 0 {
                assert_eq!(outcome.recompute, 0);
            }
        }
    }

    /// 容量越大，抢占率不该变差。这条能抓住"回收策略在容量增加时反而更糟"这类回归。
    ///
    /// 只在**没被截断**的容量区间里断言。容量小到跑不完负载时，比率被"饿死"主导 ——
    /// 见 `print_pressure_sweep`：14 块时开驱逐的比率反而比关驱逐更差，
    /// 原因不是策略，而是两个 run 服务的请求数不同。
    #[test]
    fn rate_is_non_increasing_in_capacity() {
        let capacities = [16u32, 18, 20, 24];
        let mut previous = f64::INFINITY;
        for blocks in capacities {
            let outcome = run(blocks, true);
            assert!(!outcome.truncated, "{blocks} blocks truncated; ratio not comparable");
            let rate = outcome.rate().unwrap();
            assert!(
                rate <= previous + 1e-9,
                "rate went up from {previous:.4} to {rate:.4} at {blocks} blocks"
            );
            previous = rate;
        }
        assert_eq!(previous, 0.0, "large capacity should reach zero preemption");
    }

    /// 容量定档：把这条工作负载跑到"零抢占"所需的最小块数钉住。
    ///
    /// 4 路并发 × (8 token 共享前缀 + 12 token 生成) + 3 条缓存前缀。
    /// 数字变了就说明准入/回收/块粒度有一处动了，值得看一眼。
    #[test]
    fn capacity_that_reaches_zero_preemption_is_pinned() {
        let mut smallest = None;
        for blocks in 8u32..=32 {
            if run(blocks, true).rate() == Some(0.0) {
                smallest = Some(blocks);
                break;
            }
        }
        assert_eq!(
            smallest,
            Some(19),
            "zero-preemption capacity moved; re-derive the siting number"
        );
    }
}

/// 容量定档诊断。默认 `#[ignore]`：它的输出是给人看的一张表，不是一个断言。
#[cfg(test)]
mod sweep {
    use super::*;

    #[test]
    #[ignore]
    fn print_pressure_sweep() {
        println!("blocks | decoded | with_evict preempt/recompute/rate | without preempt/recompute/rate | gave_up");
        for blocks in [8u32, 10, 12, 14, 16, 18, 20, 24, 32, 64] {
            let with = run(blocks, true);
            let without = run(blocks, false);
            println!(
                "{blocks:>6} | {:>7} | {:>3}/{:>4}/{:.3} | {:>3}/{:>4}/{:.3} | {}{}",
                with.decoded,
                with.preemptions,
                with.recompute,
                with.rate().unwrap_or(0.0),
                without.preemptions,
                without.recompute,
                without.rate().unwrap_or(0.0),
                with.gave_up,
                if with.truncated { " (truncated)" } else { "" }
            );
        }
    }
}
