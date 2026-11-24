//! E-03：序列的 KV 布局 —— 共享前缀 + 独占尾巴。
//!
//! # 结论先行：块粒度下 **CoW 机制是多余的**
//!
//! TODO 里 E-03 原本写的是"CoW 分叉：分叉后写不污染兄弟；共享前缀只算一次"。
//! 做完 E-02 的粒度决定（块级）之后，这条要求**换个方式就满足了，不需要 CoW**：
//!
//! | | 做法 |
//! | --- | --- |
//! | 共享前缀只算一次 | 前缀树命中 → 直接复用那些块（只读） |
//! | 分叉后写不污染兄弟 | 每个序列的新 token 写进**自己独占的块**，兄弟序列根本拿不到同一块 |
//! | 共享前缀不写 | 共享块的 token 数恒为 `block_size` 的整数倍 ⇒ 新 token 的位置必然落在共享段之后 |
//!
//! 关键在于第三条：**块粒度让"共享段"与"写入点"在结构上不相交**。
//! 如果是 token 粒度共享，两个序列可能共享同一个未填满的块，
//! 于是各自的下一个 token 要写进同一块的同一个 slot —— 那才必须 CoW。
//!
//! 所以这里不实现 CoW，而是把"永远不会写共享块"这条不变量**做成结构性检查**：
//! [`KvSequence::append_token`] 在 `len < shared_tokens` 时直接报错。
//! 这比实现一个用不上的 CoW 路径更有价值 —— 后者会让人以为共享写是允许的。
//!
//! 如果将来引入 token 粒度前缀缓存（或 E-05 的块表快照打破独占），
//! 这个检查会立刻变成一道道红，而不是一堆静默污染的 KV。
//!
//! # 引用计数的不变量
//!
//! 每个块被序列用到时都 retain 一次，所以对任意在用块：
//!
//! ```text
//! arena.refcount(block) == 正在使用这个块的活跃序列数
//! ```
//!
//! 私有块因此恒为 1（只有一个序列独占），共享块等于命中该前缀的活跃序列数。
//! 这条不变量让 `Arena::release` 的"引用计数非零即拒绝"真正兜住写入侧的错误。

use crate::arena::{Arena, ArenaError, Block};
use crate::prefix::PathRefs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvError {
    Arena(ArenaError),
    /// 试图往共享前缀块里写。
    ///
    /// 正常路径**永远**碰不到：共享段的 token 数是块宽的整数倍，
    /// 新 token 的位置必然在它之后。一旦出现，说明有人手工拼了一个长度不对的序列，
    /// 或者前缀树里出现了非整块的节点。
    WouldWriteSharedBlock { position: usize, shared_tokens: usize },
    /// 序列长度超出容量上限。容量在 fork 时定死，因为分页式 KV 也要预留块表列数。
    ExceedsCapacity { position: usize, capacity: usize },
    /// fork 时要求的容量装不下共享前缀本身。
    CapacityBelowPrefix { capacity: usize, shared_tokens: usize },
    /// 块宽非法（只在块宽为 0 时出现，构造 Arena 时就该拦住了）。
    ZeroBlockSize,
    /// fork/join 时前缀树或 arena 报的错，原样透出。
    Cache(crate::prefix::CacheError),
}

impl From<ArenaError> for KvError {
    fn from(error: ArenaError) -> Self {
        KvError::Arena(error)
    }
}

impl From<crate::prefix::CacheError> for KvError {
    fn from(error: crate::prefix::CacheError) -> Self {
        match error {
            crate::prefix::CacheError::Arena(inner) => KvError::Arena(inner),
            other => KvError::Cache(other),
        }
    }
}

/// 一次 KV 写入的位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub block: Block,
    pub offset: u32,
    pub position: usize,
}

impl Slot {
    /// 换算成 `AnyInferStep::slot_mapping` 里的那个整数：
    /// `物理块号 × block_size + 块内偏移`。
    pub fn slot_index(&self, block_size: u32) -> i32 {
        self.block.0 as i32 * block_size as i32 + self.offset as i32
    }
}

/// 一个序列的 KV 布局。**只由 [`crate::PrefixCache::fork`] 构造**，
/// 因为只有它持有前缀树的引用凭据。
pub struct KvSequence {
    pub(crate) shared: Vec<Block>,
    pub(crate) shared_tokens: usize,
    pub(crate) private: Vec<Block>,
    pub(crate) len: usize,
    pub(crate) capacity: usize,
    pub(crate) refs: Option<PathRefs>,
}

impl KvSequence {
    /// 已占用的 token 数（共享前缀 + 已生成的）。
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 共享前缀覆盖的 token 数。恒为 `block_size` 的整数倍。
    pub fn shared_tokens(&self) -> usize {
        self.shared_tokens
    }

    /// 共享的块（只读，永不写入）。
    pub fn shared_blocks(&self) -> &[Block] {
        &self.shared
    }

    /// 序列独占的块（可写）。
    pub fn private_blocks(&self) -> &[Block] {
        &self.private
    }

    /// 完整块表：共享前缀在前，独占尾巴在后。E-05 直接把它拷下去。
    pub fn blocks(&self) -> impl Iterator<Item = Block> + '_ {
        self.shared.iter().chain(self.private.iter()).copied()
    }

    pub fn block_count(&self) -> usize {
        self.shared.len() + self.private.len()
    }

    fn block_size(arena: &Arena) -> Result<usize, KvError> {
        let width = arena.block_size();
        if width == 0 {
            return Err(KvError::ZeroBlockSize);
        }
        Ok(width as usize)
    }

    /// 为下一个 token 分配写入位置。
    ///
    /// 需要新块时向 arena 申请并 retain，所以调用返回后该块的引用计数至少为 1 ——
    /// 它在序列归还之前不可能被回收。
    pub fn append_token(&mut self, arena: &mut Arena) -> Result<Slot, KvError> {
        let width = Self::block_size(arena)?;

        // ★ 本模块的核心检查：写入点绝不能落在共享段里。
        if self.len < self.shared_tokens {
            return Err(KvError::WouldWriteSharedBlock {
                position: self.len,
                shared_tokens: self.shared_tokens,
            });
        }
        if self.len >= self.capacity {
            return Err(KvError::ExceedsCapacity {
                position: self.len,
                capacity: self.capacity,
            });
        }

        let tail = self.len - self.shared_tokens;
        let index = tail / width;
        let offset = (tail % width) as u32;

        // 一次只前进一个 token，所以 index 最多比已有私有块数多一
        debug_assert!(index <= self.private.len(), "private tail skipped a block");
        if index == self.private.len() {
            let block = arena.allocate()?;
            arena.retain(block)?;
            self.private.push(block);
        }

        let slot = Slot {
            block: self.private[index],
            offset,
            position: self.len,
        };
        self.len += 1;
        Ok(slot)
    }

    /// 这个块是否由本序列独占（引用计数为 1）。
    ///
    /// 用于把"写侧只碰独占块"这条不变量变成可断言的量，而不是注释。
    pub fn writes_exclusively(&self, arena: &Arena, block: Block) -> bool {
        arena.refcount(block) == 1
    }

    /// 测试用：手工构造一个长度不对的序列，好让 [`KvError::WouldWriteSharedBlock`] 这条守卫被真的执行一次。
    #[cfg(test)]
    pub(crate) fn force_len(&mut self, len: usize) {
        self.len = len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaConfig;
    use crate::prefix::PrefixCache;

    const WIDTH: u32 = 4;

    fn cache(blocks: u32) -> PrefixCache {
        PrefixCache::new(
            Arena::new(ArenaConfig {
                num_blocks: blocks,
                block_size: WIDTH,
                reserved_blocks: 1,
            })
            .unwrap(),
        )
    }

    const A: [u32; 4] = [1, 2, 3, 4];
    const B: [u32; 4] = [5, 6, 7, 8];

    fn prompt() -> Vec<u32> {
        [A, B].concat()
    }

    #[test]
    fn fork_puts_the_write_point_after_the_shared_prefix() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut seq = cache.fork(node, 16).unwrap();

        assert_eq!(seq.shared_tokens(), 8);
        assert_eq!(seq.len(), 8, "fork 之后起点就是共享前缀的长度");
        assert_eq!(seq.shared_blocks().len(), 2);

        let first = seq.append_token(cache.arena_mut_for_test()).unwrap();
        assert_eq!(first.position, 8);
        assert_eq!(first.offset, 0, "共享段是整块，所以第一个新 token 在块偏移 0");
        assert!(seq.private_blocks().contains(&first.block));
        assert!(!seq.shared_blocks().contains(&first.block));
        cache.join(seq).unwrap();
    }

    /// ★ TODO 里 E-03 的完成判据：分叉后写不污染兄弟。
    #[test]
    fn forked_siblings_never_write_into_each_others_blocks() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;

        let mut left = cache.fork(node, 16).unwrap();
        let mut right = cache.fork(node, 16).unwrap();

        // 共享前缀只算一次：两个序列拿到的是同一批块
        assert_eq!(left.shared_blocks(), right.shared_blocks());
        assert_eq!(cache.arena().in_use_count(), 2, "prefix blocks must be shared, not copied");

        let left_slot = left.append_token(cache.arena_mut_for_test()).unwrap();
        let right_slot = right.append_token(cache.arena_mut_for_test()).unwrap();

        // 写点必须落在不同的块里
        assert_ne!(
            left_slot.block, right_slot.block,
            "siblings must not share a writable block"
        );
        // 而且都是各自独占的
        assert!(left.writes_exclusively(cache.arena(), left_slot.block));
        assert!(right.writes_exclusively(cache.arena(), right_slot.block));
        // 共享块被两个序列同时引用，因此不可回收也不可写
        for &block in left.shared_blocks() {
            assert_eq!(cache.arena().refcount(block), 2);
        }
        cache.join(left).unwrap();
        cache.join(right).unwrap();
    }

    /// 命中同一前缀的两个活跃序列不会共用一个可写块，因此不需要 CoW。
    /// 这个用例把"CoW 不必要"从一句论证变成一个被遍历过的结论。
    #[test]
    fn no_write_ever_targets_a_shared_block_across_a_workload() {
        let mut cache = cache(64);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut sequences: Vec<KvSequence> = Vec::new();

        for round in 0..8 {
            sequences.push(cache.fork(node, 32).unwrap());
            // 每个序列推进若干 token
            for _ in 0..(round % 5) {
                let seq = sequences.last_mut().unwrap();
                let slot = seq.append_token(cache.arena_mut_for_test()).unwrap();
                // ★ 每一次写都必须是独占块
                assert!(
                    seq.writes_exclusively(cache.arena(), slot.block),
                    "write at position {} targeted a shared block {:?}",
                    slot.position,
                    slot.block
                );
                assert!(
                    !seq.shared_blocks().contains(&slot.block),
                    "write landed inside the shared prefix"
                );
            }
        }
        assert_eq!(sequences.len(), 8);

        // 全部归还后不应残留任何引用
        for seq in sequences {
            cache.join(seq).unwrap();
        }
        assert_eq!(cache.arena().total_refcount(), 0);
        while cache.evict_leaf_lru().unwrap().is_some() {}
        assert_eq!(cache.arena().in_use_count(), 0);
    }

    /// 守卫本身必须会响 —— 否则上面那些断言只是在测"代码没跑到那里"。
    #[test]
    fn writing_into_the_shared_prefix_is_refused() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut seq = cache.fork(node, 16).unwrap();

        seq.force_len(3); // 共享 8 个 token，却声称只走到 3
        assert_eq!(
            seq.append_token(cache.arena_mut_for_test()).err(),
            Some(KvError::WouldWriteSharedBlock {
                position: 3,
                shared_tokens: 8
            })
        );
        cache.join(seq).unwrap();
    }

    #[test]
    fn capacity_is_enforced() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut seq = cache.fork(node, 10).unwrap(); // 8 共享 + 2 生成
        assert!(seq.append_token(cache.arena_mut_for_test()).is_ok());
        assert!(seq.append_token(cache.arena_mut_for_test()).is_ok());
        assert_eq!(
            seq.append_token(cache.arena_mut_for_test()).err(),
            Some(KvError::ExceedsCapacity {
                position: 10,
                capacity: 10
            })
        );
        cache.join(seq).unwrap();
    }

    #[test]
    fn capacity_below_prefix_is_refused_at_fork() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        assert_eq!(
            cache.fork(node, 4).err(),
            Some(KvError::CapacityBelowPrefix {
                capacity: 4,
                shared_tokens: 8
            })
        );
    }

    #[test]
    fn block_table_is_shared_prefix_then_private_tail() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut seq = cache.fork(node, 20).unwrap();
        for _ in 0..5 {
            seq.append_token(cache.arena_mut_for_test()).unwrap();
        }
        let table: Vec<Block> = seq.blocks().collect();
        assert_eq!(table.len(), seq.block_count());
        assert_eq!(&table[..2], seq.shared_blocks(), "共享前缀必须在前");
        assert_eq!(&table[2..], seq.private_blocks(), "独占尾巴必须在后");
        // 8 共享 + 5 生成 = 13 token -> 4 个块
        assert_eq!(table.len(), 4);
        cache.join(seq).unwrap();
    }

    #[test]
    fn joining_returns_every_block() {
        let mut cache = cache(32);
        let node = cache.insert(&prompt()).unwrap().node;
        let mut seq = cache.fork(node, 20).unwrap();
        for _ in 0..9 {
            seq.append_token(cache.arena_mut_for_test()).unwrap();
        }
        assert!(cache.arena().in_use_count() > 2);

        cache.join(seq).unwrap();
        // 共享前缀仍被缓存树持有（refcount 归零但块还在用），私有块已释放
        assert_eq!(cache.arena().total_refcount(), 0);
        assert_eq!(cache.arena().in_use_count(), 2, "only the cached prefix remains");

        // 缓存树整体回收后一块不剩
        while cache.evict_leaf_lru().unwrap().is_some() {}
        assert_eq!(cache.arena().in_use_count(), 0);
    }

    /// 随机负载：混合 insert / fork（在不同深度）/ append / join / 回收。
    ///
    /// **每一次写入都断言目标块由本序列独占。** 这是"块粒度下不需要 CoW"这条结论的
    /// 经验证据 —— 不是我推理出来的，是遍历出来的。
    #[test]
    fn randomized_workload_never_writes_a_shared_block() {
        let mut cache = cache(96);
        let width = WIDTH as usize;

        let symbols: Vec<[u32; 4]> = (0..6u32)
            .map(|k| [k * 10, k * 10 + 1, k * 10 + 2, k * 10 + 3])
            .collect();

        let mut state = 0xC0FF_EE12_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };

        // (序列, 已 fork 的深度)
        let mut live: Vec<KvSequence> = Vec::new();
        let mut writes = 0u64;

        for _ in 0..3000 {
            match next() % 8 {
                // 插入一条随机前缀
                0 => {
                    let count = 1 + (next() % 4) as usize;
                    let mut tokens = Vec::new();
                    for _ in 0..count {
                        tokens.extend_from_slice(&symbols[(next() % 6) as usize]);
                    }
                    let _ = cache.insert(&tokens);
                }
                // 在随机深度 fork
                1 | 2 => {
                    if live.len() >= 6 {
                        continue;
                    }
                    let mut tokens = Vec::new();
                    let depth = 1 + (next() % 4) as usize;
                    for _ in 0..depth {
                        tokens.extend_from_slice(&symbols[(next() % 6) as usize]);
                    }
                    let (node, matched) = cache.match_prefix(&tokens);
                    if matched > 0 {
                        if let Ok(seq) = cache.fork(node, matched + 12) {
                            debug_assert!(seq.shared_tokens() % width == 0);
                            live.push(seq);
                        }
                    }
                }
                // 生成若干 token（占 3/8，保证写入量足够）
                3..=5 => {
                    let count = 1 + (next() % 5) as usize;
                    for _ in 0..count {
                        let Some(index) = (!live.is_empty()).then(|| (next() as usize) % live.len()) else {
                            break;
                        };
                        let seq = &mut live[index];
                        let Ok(slot) = seq.append_token(cache.arena_mut_for_test()) else {
                            // 写满或块不够：这个序列就该被释放了，否则它会一直占着位置空转
                            let seq = live.swap_remove(index);
                            cache.join(seq).unwrap();
                            break;
                        };
                        writes += 1;
                        assert!(
                            seq.writes_exclusively(cache.arena(), slot.block),
                            "write to a block with refcount {} at position {}",
                            cache.arena().refcount(slot.block),
                            slot.position
                        );
                        assert!(
                            !seq.shared_blocks().contains(&slot.block),
                            "write landed in the shared prefix"
                        );
                    }
                }
                // 归还一个序列
                6 => {
                    if !live.is_empty() {
                        let index = (next() as usize) % live.len();
                        let seq = live.swap_remove(index);
                        cache.join(seq).unwrap();
                    }
                }
                // 内存压力 -> 回收（没有这一条，块会在前几十步就耗尽，之后 append 全部失败）
                _ => {
                    let _ = cache.evict_leaf_lru();
                }
            }
        }

        // 写入次数决定这条结论的份量：少于几百次就等于没遍历
        assert!(writes > 500, "workload barely wrote anything: {writes}");
        assert!(!live.is_empty(), "workload never kept a live sequence");

        for seq in live {
            cache.join(seq).unwrap();
        }
        assert_eq!(cache.arena().total_refcount(), 0, "attach/join not balanced");
        while cache.evict_leaf_lru().unwrap().is_some() {}
        assert_eq!(cache.arena().in_use_count(), 0, "blocks leaked");
    }

    #[test]
    fn refcount_equals_the_number_of_live_sequences() {
        let mut cache = cache(64);
        let node = cache.insert(&prompt()).unwrap().node;
        let shared_block = cache.blocks_below(node)[0];

        let mut live = 0;
        let mut sequences = Vec::new();
        for _ in 0..5 {
            sequences.push(cache.fork(node, 16).unwrap());
            live += 1;
            assert_eq!(cache.arena().refcount(shared_block), live);
        }
        for seq in sequences {
            cache.join(seq).unwrap();
            live -= 1;
            assert_eq!(cache.arena().refcount(shared_block), live);
        }
        assert_eq!(cache.arena().refcount(shared_block), 0);
    }
}
