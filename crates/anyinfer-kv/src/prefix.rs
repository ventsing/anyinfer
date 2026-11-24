//! E-02：块粒度前缀树 + **祖先链引用计数**。
//!
//! # 为什么是"块粒度"，不是"token 粒度"
//!
//! 这是本模块唯一一个真正的设计取舍，值得写清楚：
//!
//! | 粒度 | 好处 | 代价 |
//! | --- | --- | --- |
//! | token 级（SGLang 风格） | 命中率最高，任意长度前缀都能共享 | 一个物理块会被两个节点共享 → CoW 变成必需，且树的每次分裂都要动块 |
//! | **块级（本实现）** | 一个块**只属于一个节点**，树结构即所有权结构 | 短于块边界的共享前缀不可复用（每序列最多浪费 `block_size - 1` 个 token 的缓存机会） |
//!
//! 本基线选块级，因为"块的所有权 == 树节点的所有权"这条不变量把后面所有
//! bug 的类型都消掉了一类：没有块被两个节点共用，就不存在"兄弟节点把块写脏"、
//! 不存在分裂时的块复制、不存在两把锁之间的顺序问题。
//! 代价（块内不满的部分不共享）在本地小 batch 场景下是可以量的，而不是猜的。
//!
//! CoW 仍然需要（E-03），但那里要 CoW 的是**活跃序列的最后一个不满块**，
//! 与缓存树无关 —— 边界清楚。
//!
//! # 祖先链引用计数：本模块的核心
//!
//! 一个序列命中前缀 `root → A → B` 时，注意力会读 **A 的块和 B 的块**。
//! 所以 `attach(B)` 必须对 `A` 和 `B` 的**所有**块加引用，而不仅仅是 B 自己的。
//!
//! 如果只数"有多少序列直接挂在 A 上"，A 的计数恒为 0（没人把路径停在中间），
//! 于是任何按"计数为 0 即可回收"判断的策略都会把 A 回收掉，
//! 而挂在 B 上的活跃请求仍在读它 —— 读到的是别人写过或已释放的 HBM。
//! 这件事在设备上不崩、不报错，只让结果依赖历史。
//!
//! 两种计数在本模块里都保留并且都暴露，就是为了让这个差别**可断言**
//! （见 `ancestor_refcount_is_what_makes_reclaim_safe`）。

use crate::arena::{Arena, ArenaError, Block};
use std::collections::BTreeMap;

pub type NodeId = u32;
pub const ROOT: NodeId = 0;

struct Node {
    /// 恰好一个 block 宽的 token 段（根节点为空）。
    tokens: Vec<u32>,
    blocks: Vec<Block>,
    parent: Option<NodeId>,
    /// 键是整段 token，不是首 token：两段可以首 token 相同而后面不同。
    children: BTreeMap<Vec<u32>, NodeId>,
    /// **直接**附加在这个节点上的活跃序列数。
    ///
    /// 这个数字**不是**决定能否回收的依据 —— 它只是用来演示"只按它判断会出事"。
    /// 真正的依据是 `Arena::refcount`（祖先链意义下）。
    direct_attachments: u32,
    last_used: u64,
    alive: bool,
}

impl Node {
    fn empty(parent: Option<NodeId>) -> Self {
        Self {
            tokens: Vec::new(),
            blocks: Vec::new(),
            parent,
            children: BTreeMap::new(),
            direct_attachments: 0,
            last_used: 0,
            alive: true,
        }
    }
}

/// 一次命中的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inserted {
    /// 最深的那个节点。
    pub node: NodeId,
    /// 已被缓存覆盖的 token 数（恒为 `block_size` 的整数倍）。
    pub matched_tokens: usize,
}

/// 一次 attach 拿到的引用凭据。**必须**交还给 [`PrefixCache::detach`]。
pub struct PathRefs {
    node: NodeId,
    blocks: Vec<Block>,
    released: bool,
}

impl PathRefs {
    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
}

impl Drop for PathRefs {
    fn drop(&mut self) {
        // 忘了 detach = 引用计数永不归零 = 这些块永远回收不了。
        // 这是内存泄漏里最难注意到的一种（不增长 host 内存，只吃 HBM），
        // 所以在调试构建里直接让它炸出来。
        debug_assert!(
            self.released,
            "PathRefs for node {} dropped without detach ({} blocks leaked)",
            self.node,
            self.blocks.len()
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evicted {
    pub node: NodeId,
    pub nodes: u32,
    pub blocks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    Arena(ArenaError),
    UnknownNode(NodeId),
    CannotEvictRoot,
    /// 子树里还有块被活跃序列引用 —— 拒绝回收。
    SubtreeReferenced {
        node: NodeId,
        block: Block,
        refcount: u32,
    },
}

impl From<ArenaError> for CacheError {
    fn from(error: ArenaError) -> Self {
        CacheError::Arena(error)
    }
}

pub struct PrefixCache {
    arena: Arena,
    nodes: Vec<Node>,
    free_slots: Vec<NodeId>,
    clock: u64,
}

impl PrefixCache {
    pub fn new(arena: Arena) -> Self {
        Self {
            arena,
            nodes: vec![Node::empty(None)],
            free_slots: Vec::new(),
            clock: 0,
        }
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    pub fn block_size(&self) -> u32 {
        self.arena.block_size()
    }

    pub fn alive_nodes(&self) -> u32 {
        self.nodes.iter().filter(|node| node.alive).count() as u32
    }

    /// **直接**附加计数。刻意暴露出来，好让"它不能用来判断可回收"这件事有测试盯着。
    pub fn direct_attachments(&self, node: NodeId) -> u32 {
        self.nodes.get(node as usize).map_or(0, |n| n.direct_attachments)
    }

    fn node(&self, node: NodeId) -> Result<&Node, CacheError> {
        match self.nodes.get(node as usize) {
            Some(n) if n.alive => Ok(n),
            _ => Err(CacheError::UnknownNode(node)),
        }
    }

    fn node_mut(&mut self, node: NodeId) -> Result<&mut Node, CacheError> {
        match self.nodes.get_mut(node as usize) {
            Some(n) if n.alive => Ok(n),
            _ => Err(CacheError::UnknownNode(node)),
        }
    }

    /// 沿 root 向下匹配**整块**。返回最深的命中节点和已覆盖的 token 数。
    pub fn match_prefix(&self, tokens: &[u32]) -> (NodeId, usize) {
        let width = self.block_size() as usize;
        let mut node = ROOT;
        let mut consumed = 0;
        while consumed + width <= tokens.len() {
            let chunk = &tokens[consumed..consumed + width];
            let Some(&child) = self.nodes[node as usize].children.get(chunk) else {
                break;
            };
            node = child;
            consumed += width;
        }
        (node, consumed)
    }

    /// 把 `tokens` 的整块前缀插入缓存。
    ///
    /// 尾部不满一个块的 token **不入缓存**（块粒度设计的直接后果）。
    ///
    /// 块不够时整体失败，不做部分插入：部分插入会让"插入失败"这个结果
    /// 携带一个说不清的副作用，调用方无法据此做任何判断。
    pub fn insert(&mut self, tokens: &[u32]) -> Result<Inserted, CacheError> {
        let width = self.block_size() as usize;
        let (mut node, mut consumed) = self.match_prefix(tokens);
        let needed = (tokens.len() - consumed) / width;
        if self.arena.free_count() < needed as u32 {
            return Err(CacheError::Arena(ArenaError::Exhausted));
        }

        while consumed + width <= tokens.len() {
            let chunk = tokens[consumed..consumed + width].to_vec();
            let block = self.arena.allocate()?;
            let child = self.spawn_node(chunk.clone(), vec![block], node);
            self.nodes[node as usize].children.insert(chunk, child);
            node = child;
            consumed += width;
        }
        self.touch_path(node);
        Ok(Inserted {
            node,
            matched_tokens: consumed,
        })
    }

    /// root → `node` 路径上的所有块。**这就是"祖先链"的具体形式。**
    fn path_blocks(&self, node: NodeId) -> Result<Vec<Block>, CacheError> {
        let mut out = Vec::new();
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            let n = self.node(current)?;
            out.extend_from_slice(&n.blocks);
            cursor = n.parent;
        }
        Ok(out)
    }

    /// 命中一个节点：对**整条路径**加引用。
    pub fn attach(&mut self, node: NodeId) -> Result<PathRefs, CacheError> {
        let blocks = self.path_blocks(node)?;
        for &block in &blocks {
            self.arena.retain(block)?;
        }
        self.node_mut(node)?.direct_attachments += 1;
        self.touch_path(node);
        Ok(PathRefs {
            node,
            blocks,
            released: false,
        })
    }

    /// 与 [`Self::attach`] 严格配平。
    pub fn detach(&mut self, mut refs: PathRefs) {
        for &block in &refs.blocks {
            // 这里不该失败：attach 成功过就意味着这些块还在用。
            // 失败说明调用方把同一个 PathRefs 交了两次，或者中途把块强行回收了。
            debug_assert!(
                self.arena.unretain(block).is_ok(),
                "detach saw an inconsistent block {block:?}"
            );
        }
        if let Some(node) = self.nodes.get_mut(refs.node as usize) {
            node.direct_attachments = node.direct_attachments.saturating_sub(1);
        }
        refs.released = true;
    }

    fn spawn_node(
        &mut self,
        tokens: Vec<u32>,
        blocks: Vec<Block>,
        parent: NodeId,
    ) -> NodeId {
        let node = if let Some(slot) = self.free_slots.pop() {
            let slot_index = slot as usize;
            self.nodes[slot_index] = Node::empty(Some(parent));
            self.nodes[slot_index].tokens = tokens;
            self.nodes[slot_index].blocks = blocks;
            slot
        } else {
            let mut fresh = Node::empty(Some(parent));
            fresh.tokens = tokens;
            fresh.blocks = blocks;
            self.nodes.push(fresh);
            (self.nodes.len() - 1) as NodeId
        };
        node
    }

    fn touch_path(&mut self, node: NodeId) {
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            self.clock += 1;
            let clock = self.clock;
            if let Some(entry) = self.nodes.get_mut(current as usize) {
                entry.last_used = clock;
                cursor = entry.parent;
            } else {
                cursor = None;
            }
        }
    }

    fn subtree_nodes(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            if let Some(entry) = self.nodes.get(current as usize) {
                if entry.alive {
                    out.push(current);
                    stack.extend(entry.children.values().copied());
                }
            }
        }
        out
    }

    /// 回收一整棵子树。
    ///
    /// 拒绝条件是"子树里**任何**一个块的祖先链引用计数不为 0"。
    /// 注意这个检查覆盖整棵子树，而不只是子树的根 —— 因为活跃序列可能挂在深处。
    pub fn evict_subtree(&mut self, node: NodeId) -> Result<Evicted, CacheError> {
        self.node(node)?;
        if node == ROOT {
            return Err(CacheError::CannotEvictRoot);
        }

        let ids = self.subtree_nodes(node);
        let mut blocks = Vec::new();
        for &id in &ids {
            for &block in &self.nodes[id as usize].blocks {
                let refcount = self.arena.refcount(block);
                if refcount != 0 {
                    return Err(CacheError::SubtreeReferenced {
                        node,
                        block,
                        refcount,
                    });
                }
                blocks.push(block);
            }
        }

        // 先从父节点的 children 摘掉，再释放 slot 和块
        let parent = self.nodes[node as usize].parent.ok_or(CacheError::CannotEvictRoot)?;
        let key = self.nodes[node as usize].tokens.clone();
        self.nodes[parent as usize].children.remove(&key);

        for &id in &ids {
            let entry = &mut self.nodes[id as usize];
            entry.alive = false;
            entry.tokens = Vec::new();
            entry.blocks = Vec::new();
            entry.children = BTreeMap::new();
            entry.direct_attachments = 0;
            self.free_slots.push(id);
        }

        let block_count = blocks.len() as u32;
        for block in blocks {
            // 上面已经逐个确认 refcount == 0，这里不该失败
            debug_assert!(self.arena.release(block).is_ok());
            let _ = self.arena.release(block);
        }

        Ok(Evicted {
            node,
            nodes: ids.len() as u32,
            blocks: block_count,
        })
    }

    /// 按逻辑 LRU 回收一个**叶**节点（DESIGN §5.2 第 1 级）。
    ///
    /// 两个条件同时成立才可回收：没有子节点，且所有块的引用计数为 0。
    /// 这两条在纯树结构下是冗余的，但**保留两条**是有意的：
    /// 等 E-03 的 CoW 与 E-05 的块表快照打破"块只属于一个节点"之后，
    /// 叶节点条件不再蕴含引用条件，届时只有引用计数还在兜底。
    pub fn evict_leaf_lru(&mut self) -> Result<Option<Evicted>, CacheError> {
        let mut best: Option<(u64, NodeId)> = None;
        for (index, entry) in self.nodes.iter().enumerate() {
            if !entry.alive || index == ROOT as usize || !entry.children.is_empty() {
                continue;
            }
            let collectable = entry
                .blocks
                .iter()
                .all(|&block| self.arena.refcount(block) == 0);
            if !collectable {
                continue;
            }
            if best.is_none_or(|(stamp, _)| entry.last_used < stamp) {
                best = Some((entry.last_used, index as NodeId));
            }
        }
        match best {
            None => Ok(None),
            Some((_, node)) => Ok(Some(self.evict_subtree(node)?)),
        }
    }

    /// 把一棵子树里的所有块号按 DFS 顺序列出来。供 E-05 拼块表用。
    pub fn blocks_below(&self, node: NodeId) -> Vec<Block> {
        let mut out = Vec::new();
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            if let Some(entry) = self.nodes.get(current as usize) {
                if entry.alive {
                    out.extend(entry.blocks.iter().rev().copied());
                }
                cursor = entry.parent;
            } else {
                cursor = None;
            }
        }
        // 上面是从 node 往 root 走，翻转成 root → node
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaConfig;

    fn cache() -> PrefixCache {
        PrefixCache::new(
            Arena::new(ArenaConfig {
                num_blocks: 16,
                block_size: 4,
                reserved_blocks: 1,
            })
            .unwrap(),
        )
    }

    /// 4 个 token 一块，方便手写。
    const A: [u32; 4] = [1, 2, 3, 4];
    const B: [u32; 4] = [5, 6, 7, 8];
    const C: [u32; 4] = [9, 10, 11, 12];

    fn tokens(parts: &[&[u32]]) -> Vec<u32> {
        parts.iter().flat_map(|part| part.iter().copied()).collect()
    }

    #[test]
    fn insert_then_match() {
        let mut cache = cache();
        let inserted = cache.insert(&tokens(&[&A, &B])).unwrap();
        assert_eq!(inserted.matched_tokens, 8);

        let (node, matched) = cache.match_prefix(&tokens(&[&A, &B]));
        assert_eq!(matched, 8);
        assert_eq!(node, inserted.node);

        // 只匹配到第一个块
        let (node, matched) = cache.match_prefix(&tokens(&[&A, &C]));
        assert_eq!(matched, 4);
        assert_ne!(node, inserted.node);
    }

    #[test]
    fn partial_block_prefix_is_not_cached() {
        let mut cache = cache();
        let inserted = cache.insert(&[1, 2, 3]).unwrap();
        assert_eq!(inserted.matched_tokens, 0);
        assert_eq!(inserted.node, ROOT);
        assert_eq!(cache.alive_nodes(), 1);
    }

    #[test]
    fn divergence_creates_siblings_sharing_the_ancestor() {
        let mut cache = cache();
        let first = cache.insert(&tokens(&[&A, &B])).unwrap();
        let second = cache.insert(&tokens(&[&A, &C])).unwrap();

        // A 只有一个节点，B 和 C 是它的两个兄弟
        assert_eq!(cache.alive_nodes(), 4); // root + A + B + C
        // blocks_below 是 root → node 顺序，所以 [0] 是祖先 A 的块
        let a_block = cache.blocks_below(first.node)[0];
        let a_block_second = cache.blocks_below(second.node)[0];
        assert_eq!(a_block, a_block_second, "shared ancestor block must be identical");
    }

    /// 本模块的核心测试。
    #[test]
    fn ancestor_refcount_is_what_makes_reclaim_safe() {
        let mut cache = cache();
        let deep = cache.insert(&tokens(&[&A, &B])).unwrap();
        let ancestor = cache.match_prefix(&tokens(&[&A])).0;
        assert_ne!(ancestor, ROOT);

        let ancestor_block = cache.blocks_below(ancestor)[0];
        let deep_block = cache.blocks_below(deep.node)[1];

        // attach 之前：两者都空闲
        assert_eq!(cache.arena().refcount(ancestor_block), 0);
        assert_eq!(cache.arena().refcount(deep_block), 0);

        let refs = cache.attach(deep.node).unwrap();

        // ★ 祖先的块也被加上了引用 —— 这就是祖先链计数
        assert_eq!(
            cache.arena().refcount(ancestor_block),
            1,
            "ancestor block must be referenced by the descendant's path"
        );
        assert_eq!(cache.arena().refcount(deep_block), 1);

        // ★ 而"直接附加"计数在祖先上是 0：
        //   只看这个数字的实现会以为祖先可以回收，然后毁掉挂在深处的活跃请求
        assert_eq!(cache.direct_attachments(ancestor), 0);
        assert_eq!(cache.direct_attachments(deep.node), 1);

        // 祖先那条路径不能回收
        assert!(matches!(
            cache.evict_subtree(ancestor),
            Err(CacheError::SubtreeReferenced { .. })
        ));

        // 叶 LRU 是**跳过**被引用的叶（返回 None = 没有可回收的叶），
        // 而不是报错 —— 它的职责是挑一个能回收的，不是抱怨挑不到。
        assert_eq!(
            cache.evict_leaf_lru().unwrap(),
            None,
            "leaf LRU must skip a referenced leaf"
        );

        cache.detach(refs);
        assert_eq!(cache.arena().refcount(ancestor_block), 0);
        assert_eq!(cache.arena().total_refcount(), 0);

        // 现在可以整棵回收
        let evicted = cache.evict_subtree(ancestor).unwrap();
        assert_eq!(evicted.nodes, 2);
        assert_eq!(evicted.blocks, 2);
        assert_eq!(cache.alive_nodes(), 1);
        assert_eq!(cache.arena().in_use_count(), 0);
    }

    #[test]
    fn two_requests_sharing_an_ancestor_both_hold_it() {
        let mut cache = cache();
        let left = cache.insert(&tokens(&[&A, &B])).unwrap();
        let right = cache.insert(&tokens(&[&A, &C])).unwrap();
        let ancestor = cache.match_prefix(&tokens(&[&A])).0;
        let ancestor_block = cache.blocks_below(ancestor)[0];

        let left_refs = cache.attach(left.node).unwrap();
        let right_refs = cache.attach(right.node).unwrap();
        assert_eq!(cache.arena().refcount(ancestor_block), 2);

        cache.detach(left_refs);
        assert_eq!(cache.arena().refcount(ancestor_block), 1);
        assert!(matches!(
            cache.evict_subtree(ancestor),
            Err(CacheError::SubtreeReferenced { .. })
        ));

        cache.detach(right_refs);
        assert_eq!(cache.arena().refcount(ancestor_block), 0);
        assert!(cache.evict_subtree(ancestor).is_ok());
    }

    #[test]
    fn leaf_lru_picks_the_oldest_collectable_leaf() {
        let mut cache = cache();
        let first = cache.insert(&tokens(&[&A, &B])).unwrap();
        // 稍微过一会儿再插入第二条分支，让它更"新"
        for _ in 0..3 {
            cache.touch_path(ROOT);
        }
        let second = cache.insert(&tokens(&[&A, &C])).unwrap();

        // 把 second 的叶 attach 住，于是只有 first 的叶可回收
        let refs = cache.attach(second.node).unwrap();
        let evicted = cache.evict_leaf_lru().unwrap().unwrap();
        assert_eq!(evicted.node, first.node);
        cache.detach(refs);

        // 剩下的叶就是 second
        let evicted = cache.evict_leaf_lru().unwrap().unwrap();
        assert_eq!(evicted.node, second.node);
        // 两次之后只剩 A（现在是叶）和 root
        let evicted = cache.evict_leaf_lru().unwrap().unwrap();
        assert_eq!(evicted.blocks, 1);
        assert!(cache.evict_leaf_lru().unwrap().is_none());
    }

    #[test]
    fn root_cannot_be_evicted() {
        let mut cache = cache();
        assert_eq!(
            cache.evict_subtree(ROOT).err(),
            Some(CacheError::CannotEvictRoot)
        );
        assert!(cache.evict_leaf_lru().unwrap().is_none());
    }

    #[test]
    fn unknown_node_is_reported() {
        let mut cache = cache();
        assert_eq!(
            cache.evict_subtree(999).err(),
            Some(CacheError::UnknownNode(999))
        );
        assert_eq!(cache.attach(999).err(), Some(CacheError::UnknownNode(999)));
    }

    #[test]
    fn insert_is_atomic_when_blocks_run_out() {
        // 3 个可用块（16 总 - 1 保留 = 15，这里单独构造一个小的）
        let mut cache = PrefixCache::new(
            Arena::new(ArenaConfig {
                num_blocks: 4,
                block_size: 4,
                reserved_blocks: 1,
            })
            .unwrap(),
        );
        // 需要 3 个块，只有 3 个可用 —— 刚好够
        assert!(cache.insert(&tokens(&[&A, &B, &C])).is_ok());
        assert_eq!(cache.arena().free_count(), 0);

        // 再来一条完全不同的 3 块前缀：块不够，必须整体失败
        let before = cache.alive_nodes();
        assert_eq!(
            cache.insert(&[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
            Err(CacheError::Arena(ArenaError::Exhausted))
        );
        assert_eq!(
            cache.alive_nodes(),
            before,
            "failed insert must not leave partial nodes behind"
        );
    }

    #[test]
    fn blocks_below_returns_root_to_node_order() {
        let mut cache = cache();
        let deep = cache.insert(&tokens(&[&A, &B, &C])).unwrap();
        let blocks = cache.blocks_below(deep.node);
        assert_eq!(blocks.len(), 3);
        // 顺序必须是逻辑顺序（root → node），块表就按这个顺序拼
        let mut expected = Vec::new();
        let mut cursor = Some(deep.node);
        while let Some(current) = cursor {
            if let Some(&block) = cache.nodes[current as usize].blocks.first() {
                expected.push(block);
            }
            cursor = cache.nodes[current as usize].parent;
        }
        expected.reverse();
        assert_eq!(blocks, expected);
        // 三个块互不相同
        assert_ne!(blocks[0], blocks[1]);
        assert_ne!(blocks[1], blocks[2]);
        assert_ne!(blocks[0], blocks[2]);
    }

    /// 负向用例：**证明"只数直接附加"这条判据会毁掉活跃请求。**
    ///
    /// 这里不执行破坏动作（那需要一个故意写错的实现），而是把朴素判据的判定结果
    /// 和真实后果都算出来摆在一起 —— 于是"祖先链计数不可省"是一个可断言的结论，
    /// 而不是注释里的一句话。
    #[test]
    fn naive_shallow_counting_would_free_a_block_that_is_still_being_read() {
        let mut cache = cache();
        // root → A → B，活跃请求挂在 B 上
        let deep = cache.insert(&tokens(&[&A, &B])).unwrap();
        let ancestor = cache.match_prefix(&tokens(&[&A])).0;
        let ancestor_block = cache.blocks_below(ancestor)[0];

        let refs = cache.attach(deep.node).unwrap();

        // (1) 深度优先的朴素判据：祖先上没有"直接附加的请求"，看起来可以回收
        let naive_says_evictable = cache.direct_attachments(ancestor) == 0;
        assert!(
            naive_says_evictable,
            "前置条件不成立：祖先上居然有直接附加"
        );

        // (2) 但活跃请求的 KV 路径**包含**祖先的块 —— 注意力会读它
        let live_path = cache.blocks_below(deep.node);
        assert!(
            live_path.contains(&ancestor_block),
            "ancestor block is not on the live path; the test premise is wrong"
        );

        // (3) 祖先链判据：这个块正被引用，不许回收
        assert_eq!(cache.arena().refcount(ancestor_block), 1);
        assert!(matches!(
            cache.evict_subtree(ancestor),
            Err(CacheError::SubtreeReferenced { refcount: 1, .. })
        ));

        // (4) 如果真的按朴素判据回收，这个块会进自由队列并被别的请求拿走，
        //     于是活跃请求读到别人的 KV。断言它确实还在原地。
        assert!(cache.arena().is_in_use(ancestor_block));
        assert_eq!(cache.arena().total_refcount(), 2);

        cache.detach(refs);
    }

    /// 2000 步随机负载：每一步都检查块守恒，收尾后必须一滴不剩。
    /// 这是唯一能抓住"attach/detach 不对称"这类错误的东西。
    #[test]
    fn randomized_workload_conserves_blocks_and_refcounts() {
        const BLOCKS: u32 = 64;
        const RESERVED: u32 = 1;
        let mut cache = PrefixCache::new(
            Arena::new(ArenaConfig {
                num_blocks: BLOCKS,
                block_size: 4,
                reserved_blocks: RESERVED,
            })
            .unwrap(),
        );

        let mut state = 0x5EED_1234_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };

        let symbols: Vec<Vec<u32>> = (0..8u32)
            .map(|k| (0..4).map(|i| k * 10 + i).collect())
            .collect();

        let mut held: Vec<PathRefs> = Vec::new();

        for _ in 0..2000 {
            match next() % 5 {
                0 | 1 => {
                    let count = 1 + (next() % 4) as usize;
                    let mut tokens = Vec::new();
                    for _ in 0..count {
                        tokens.extend_from_slice(&symbols[(next() % 8) as usize]);
                    }
                    // 块不够是可接受的，但不许留下半棵树（insert 内部保证原子）
                    let _ = cache.insert(&tokens);
                }
                2 => {
                    let mut tokens = Vec::new();
                    for _ in 0..3 {
                        tokens.extend_from_slice(&symbols[(next() % 8) as usize]);
                    }
                    let (node, matched) = cache.match_prefix(&tokens);
                    if matched > 0 {
                        if let Ok(refs) = cache.attach(node) {
                            held.push(refs);
                        }
                    }
                }
                3 => {
                    if !held.is_empty() {
                        let index = (next() as usize) % held.len();
                        let refs = held.swap_remove(index);
                        cache.detach(refs);
                    }
                }
                _ => {
                    let _ = cache.evict_leaf_lru();
                }
            }

            // 不变量：块守恒
            assert_eq!(
                cache.arena().in_use_count() + cache.arena().free_count(),
                BLOCKS - RESERVED,
                "block accounting drifted"
            );
        }

        // 收尾：还掉所有引用，引用计数必须归零
        for refs in held.drain(..) {
            cache.detach(refs);
        }
        assert_eq!(
            cache.arena().total_refcount(),
            0,
            "attach/detach are not balanced"
        );

        // 全部回收后一块不剩
        while cache.evict_leaf_lru().unwrap().is_some() {}
        assert_eq!(
            cache.arena().in_use_count(),
            0,
            "{} blocks leaked",
            cache.arena().in_use_count()
        );
        assert_eq!(cache.alive_nodes(), 1, "only root should remain");
    }

    #[test]
    fn freed_slots_are_reused() {
        let mut cache = cache();
        let deep = cache.insert(&tokens(&[&A])).unwrap();
        cache.evict_subtree(deep.node).unwrap();
        let again = cache.insert(&tokens(&[&B])).unwrap();
        // 槽位被复用，节点总数没有因为一次回收+一次插入而增长
        assert_eq!(cache.alive_nodes(), 2);
        assert_eq!(again.matched_tokens, 4);
    }
}
