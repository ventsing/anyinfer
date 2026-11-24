//! E-01：物理块分配器。
//!
//! # 为什么"O(1)"是硬要求而不是优化
//!
//! block 分配发生在**请求准入**和**前缀 cache 命中**时。准入一旦变成 O(n) 扫描，
//! 高并发下它就从"可以忽略"变成"排队时间"——而排队时间不随 batch 增大而摊薄。
//! 所以自由队列必须是真正的 O(1)：一个 `Vec` 当栈用，`pop`/`push`。
//!
//! LIFO 而不是 FIFO 还顺带一个好处：刚释放的块大概率还在 TLB / cache 里。
//!
//! # 引用计数与"能不能回收"是两件事
//!
//! | 概念 | 含义 |
//! | --- | --- |
//! | `in_use` | 这个块是否在自由队列之外（是否被某个节点持有） |
//! | `refcount` | 有多少个**活跃序列的 KV 路径**经过这个块 |
//!
//! 一个块可以"被持有但 refcount 为 0"——那正是一个可以驱逐的缓存节点。
//! 反过来，"refcount > 0 却被放回自由队列"就是 use-after-free，
//! 所以 [`Arena::release`] 拒绝这种调用，而不是默默让人读已释放的 HBM。
//!
//! 这类错误在设备上不会崩，只会让结果依赖历史 —— 也就是最难查的一类 bug。
//! 在这里把它变成 `Err` 是唯一划算的做法。

/// 物理块编号。用 newtype 是为了不和序列 id / 逻辑块号混用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block(pub u32);

impl Block {
    pub fn id(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaConfig {
    pub num_blocks: u32,
    /// **部署期参数，不是常量。** A2/V4.1 用 128，其余 32。
    /// 硬编 16/32 会让某些 SoC 上的 attention 掉档。
    pub block_size: u32,
    /// 永久保留、永不出现在自由队列里的块数。
    ///
    /// padding 行会读这些块（DESIGN §4.4）。它们必须一直有效：
    /// 一旦被回收，dummy 行就在读别人写过或已释放的 HBM，
    /// 于是同一个 graph 两次运行结果不同，且只在 batch 不满桶时出现。
    pub reserved_blocks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// 没有空闲块了。调用方应触发驱逐 / 换出 / 抢占（DESIGN §5.2）。
    Exhausted,
    /// 释放一个已经在自由队列里的块。自由队列出现重复项 = 两个请求会拿到同一块。
    DoubleFree(Block),
    /// 释放一个仍被活跃序列引用的块 = use-after-free。
    StillReferenced {
        block: Block,
        refcount: u32,
    },
    /// 保留块不许释放。
    Reserved(Block),
    /// 块号超出范围。
    UnknownBlock(Block),
    /// 引用计数减到负数（多减了一次）。
    RefcountUnderflow(Block),
    ZeroBlocks,
    ZeroBlockSize,
    ReservedExceedsTotal {
        reserved: u32,
        total: u32,
    },
}

pub struct Arena {
    config: ArenaConfig,
    /// 自由队列。`pop()` 取块，`push()` 还块，两端都在尾部 —— O(1)。
    free: Vec<u32>,
    in_use: Vec<bool>,
    refcount: Vec<u32>,
    /// 曾经同时被占用的峰值。用于报告真实内存压力，不做决策。
    high_water: u32,
}

impl Arena {
    pub fn new(config: ArenaConfig) -> Result<Self, ArenaError> {
        if config.num_blocks == 0 {
            return Err(ArenaError::ZeroBlocks);
        }
        if config.block_size == 0 {
            return Err(ArenaError::ZeroBlockSize);
        }
        if config.reserved_blocks > config.num_blocks {
            return Err(ArenaError::ReservedExceedsTotal {
                reserved: config.reserved_blocks,
                total: config.num_blocks,
            });
        }

        // 保留块在前，自由块在后。入栈顺序决定出栈顺序，所以先 push 大号块，
        // 让最先被分配出去的是较小的编号 —— 纯为了日志好读。
        let mut free = Vec::with_capacity((config.num_blocks - config.reserved_blocks) as usize);
        for id in (config.reserved_blocks..config.num_blocks).rev() {
            free.push(id);
        }

        Ok(Self {
            config,
            free,
            in_use: vec![false; config.num_blocks as usize],
            refcount: vec![0; config.num_blocks as usize],
            high_water: 0,
        })
    }

    pub fn config(&self) -> ArenaConfig {
        self.config
    }

    pub fn block_size(&self) -> u32 {
        self.config.block_size
    }

    pub fn free_count(&self) -> u32 {
        self.free.len() as u32
    }

    pub fn in_use_count(&self) -> u32 {
        self.config.num_blocks - self.config.reserved_blocks - self.free_count()
    }

    pub fn high_water(&self) -> u32 {
        self.high_water
    }

    /// O(1) 分配。
    pub fn allocate(&mut self) -> Result<Block, ArenaError> {
        let Some(id) = self.free.pop() else {
            return Err(ArenaError::Exhausted);
        };
        self.in_use[id as usize] = true;
        self.refcount[id as usize] = 0;
        let used = self.in_use_count();
        if used > self.high_water {
            self.high_water = used;
        }
        Ok(Block(id))
    }

    /// O(1) 归还。要求 refcount == 0 —— 否则就是 use-after-free。
    pub fn release(&mut self, block: Block) -> Result<(), ArenaError> {
        let id = block.0;
        if id >= self.config.num_blocks {
            return Err(ArenaError::UnknownBlock(block));
        }
        if id < self.config.reserved_blocks {
            return Err(ArenaError::Reserved(block));
        }
        if !self.in_use[id as usize] {
            return Err(ArenaError::DoubleFree(block));
        }
        let refcount = self.refcount[id as usize];
        if refcount != 0 {
            return Err(ArenaError::StillReferenced { block, refcount });
        }
        self.in_use[id as usize] = false;
        self.free.push(id);
        Ok(())
    }

    pub fn retain(&mut self, block: Block) -> Result<u32, ArenaError> {
        let id = block.0;
        if id >= self.config.num_blocks {
            return Err(ArenaError::UnknownBlock(block));
        }
        if !self.in_use[id as usize] {
            return Err(ArenaError::DoubleFree(block));
        }
        self.refcount[id as usize] += 1;
        Ok(self.refcount[id as usize])
    }

    pub fn unretain(&mut self, block: Block) -> Result<u32, ArenaError> {
        let id = block.0;
        if id >= self.config.num_blocks {
            return Err(ArenaError::UnknownBlock(block));
        }
        if !self.in_use[id as usize] {
            return Err(ArenaError::DoubleFree(block));
        }
        if self.refcount[id as usize] == 0 {
            return Err(ArenaError::RefcountUnderflow(block));
        }
        self.refcount[id as usize] -= 1;
        Ok(self.refcount[id as usize])
    }

    pub fn refcount(&self, block: Block) -> u32 {
        self.refcount[block.0 as usize]
    }

    pub fn is_in_use(&self, block: Block) -> bool {
        self.in_use[block.0 as usize]
    }

    /// 保留块（padding 行读的那些）。
    pub fn reserved_blocks(&self) -> impl Iterator<Item = Block> + '_ {
        (0..self.config.reserved_blocks).map(Block)
    }

    /// 所有引用计数之和。用于在测试里断言 attach/detach 严格配平。
    pub fn total_refcount(&self) -> u64 {
        self.refcount.iter().map(|&count| u64::from(count)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(num_blocks: u32, reserved: u32) -> Arena {
        Arena::new(ArenaConfig {
            num_blocks,
            block_size: 4,
            reserved_blocks: reserved,
        })
        .unwrap()
    }

    #[test]
    fn config_validation() {
        assert_eq!(
            Arena::new(ArenaConfig {
                num_blocks: 0,
                block_size: 4,
                reserved_blocks: 0
            })
            .err(),
            Some(ArenaError::ZeroBlocks)
        );
        assert_eq!(
            Arena::new(ArenaConfig {
                num_blocks: 4,
                block_size: 0,
                reserved_blocks: 0
            })
            .err(),
            Some(ArenaError::ZeroBlockSize)
        );
        assert_eq!(
            Arena::new(ArenaConfig {
                num_blocks: 4,
                block_size: 4,
                reserved_blocks: 5
            })
            .err(),
            Some(ArenaError::ReservedExceedsTotal {
                reserved: 5,
                total: 4
            })
        );
    }

    #[test]
    fn reserved_blocks_are_never_handed_out() {
        let mut arena = arena(8, 2);
        assert_eq!(arena.free_count(), 6);
        let mut taken = Vec::new();
        while let Ok(block) = arena.allocate() {
            taken.push(block);
        }
        assert_eq!(taken.len(), 6);
        assert!(
            taken.iter().all(|block| block.id() >= 2),
            "a reserved block leaked into the free pool: {taken:?}"
        );
        assert!(arena.reserved_blocks().all(|block| !arena.is_in_use(block)));
    }

    #[test]
    fn allocation_and_release_are_balanced() {
        let mut arena = arena(4, 1);
        let a = arena.allocate().unwrap();
        let b = arena.allocate().unwrap();
        assert_eq!(arena.free_count(), 1);
        arena.release(a).unwrap();
        arena.release(b).unwrap();
        assert_eq!(arena.free_count(), 3);
        assert_eq!(arena.in_use_count(), 0);
    }

    #[test]
    fn exhaustion_is_reported_not_panicked() {
        let mut arena = arena(2, 1);
        arena.allocate().unwrap();
        assert_eq!(arena.allocate().err(), Some(ArenaError::Exhausted));
    }

    /// 双释放会让自由队列出现重复项，于是两个请求拿到同一块 —— 必须报错。
    #[test]
    fn double_free_is_caught() {
        let mut arena = arena(4, 1);
        let block = arena.allocate().unwrap();
        arena.release(block).unwrap();
        assert_eq!(
            arena.release(block).err(),
            Some(ArenaError::DoubleFree(block))
        );
        // 而且自由队列没有因此变长
        assert_eq!(arena.free_count(), 3);
    }

    /// 释放一个仍被引用的块 = use-after-free。这是本模块存在的核心理由。
    #[test]
    fn releasing_a_referenced_block_is_refused() {
        let mut arena = arena(4, 1);
        let block = arena.allocate().unwrap();
        arena.retain(block).unwrap();
        assert_eq!(
            arena.release(block).err(),
            Some(ArenaError::StillReferenced { block, refcount: 1 })
        );
        // 拒绝之后状态没变
        assert!(arena.is_in_use(block));
        assert_eq!(arena.refcount(block), 1);

        arena.unretain(block).unwrap();
        arena.release(block).unwrap();
        assert!(!arena.is_in_use(block));
    }

    #[test]
    fn refcount_cannot_underflow() {
        let mut arena = arena(4, 1);
        let block = arena.allocate().unwrap();
        assert_eq!(
            arena.unretain(block).err(),
            Some(ArenaError::RefcountUnderflow(block))
        );
    }

    #[test]
    fn refcount_operations_on_free_blocks_are_refused() {
        let mut arena = arena(4, 1);
        let block = arena.allocate().unwrap();
        arena.release(block).unwrap();
        assert_eq!(
            arena.retain(block).err(),
            Some(ArenaError::DoubleFree(block))
        );
        assert_eq!(
            arena.unretain(block).err(),
            Some(ArenaError::DoubleFree(block))
        );
    }

    #[test]
    fn reserved_block_cannot_be_released() {
        let mut arena = arena(4, 1);
        assert_eq!(
            arena.release(Block(0)).err(),
            Some(ArenaError::Reserved(Block(0)))
        );
    }

    #[test]
    fn out_of_range_block_is_reported() {
        let mut arena = arena(4, 1);
        assert_eq!(
            arena.release(Block(99)).err(),
            Some(ArenaError::UnknownBlock(Block(99)))
        );
    }

    /// LIFO：刚释放的块下一个就拿到，热点块留在 TLB 里。
    #[test]
    fn free_queue_is_lifo() {
        let mut arena = arena(4, 1);
        let a = arena.allocate().unwrap();
        let b = arena.allocate().unwrap();
        assert_ne!(a, b);
        arena.release(a).unwrap();
        // 刚还回去的 a 必须立刻能拿回来
        assert_eq!(arena.allocate().unwrap(), a);
        // 此时 b 仍在使用中，所以下一个拿到的是别的块
        assert_ne!(arena.allocate().unwrap(), b);
    }

    #[test]
    fn high_water_tracks_peak_usage() {
        let mut arena = arena(6, 1);
        let blocks: Vec<_> = (0..3).map(|_| arena.allocate().unwrap()).collect();
        assert_eq!(arena.high_water(), 3);
        for block in &blocks {
            arena.release(*block).unwrap();
        }
        assert_eq!(arena.high_water(), 3, "high water must not decrease");
        assert_eq!(arena.in_use_count(), 0);
    }
}
