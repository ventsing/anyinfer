//! 步级 plan（D-02）：**所有分配都发生在 plan 期，稳态每步零分配**。
//!
//! # 为什么这条必须被断言而不是被假设
//!
//! V4.1 上一个 decode step 约 30 ms，但那是被 100+ 层的计算填满的。
//! host 侧一次 `malloc` 本身只要几百纳秒 —— 单看毫无意义。
//! 真正的问题是**它出现在热路径上的位置不可控**：
//!
//!   - 分配失败时，故障点落在"第 N 步的第 3 行已经写了一半"的地方；
//!   - 分配器抖动会把延迟毛刺（p99）推高，而 p99 是推理服务的核心指标；
//!   - 分配出来的新缓冲可能没被 pin，H2D 退化成同步拷贝，30 ms 的步被拉成 30 ms + 拷贝。
//!
//! 所以这里不去优化"分配快不快"，而是**把分配从热路径上彻底移除**：
//! 形状在 plan 期定死，缓冲在 plan 期一次要够，稳态只做 `copy_from_slice`。
//!
//! # 形状改变的处置：拒绝，而不是重新分配
//!
//! `AnyInferStep` 的形状字段（`batch_size` / `max_blocks` / `vocab_size` / `topk`）
//! 在同一个捕获图内必须不变（DESIGN §4.4）。所以 plan 遇到超界请求时返回错误，
//! 由上层换桶、换 plan —— **绝不在热路径上扩容**。
//! 扩容一次 = 重新分配 + 重新捕获图 = 这一帧的 30 ms 变成了秒级。

pub mod staging;

pub use staging::{
    DecodeRow, HostStaging, PAD_POSITION, PAD_SEQ_LEN, PAD_TOKEN_ID, SLOT_NO_WRITE, StepView,
};

/// 设备缓冲对齐。CANN 的 H2D 与算子都按 512 B 对齐组织。
pub const DEVICE_ALIGN: u32 = 512;

/// 全部区域合计的上限。用来把"配置写错一个零"变成错误而不是 OOM。
pub const MAX_TOTAL_BYTES: u64 = 1 << 40; // 1 TiB

/// plan 的形状。**创建后不可变**：改形状 = 换 plan = 换捕获图。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanConfig {
    /// batch 桶大小。捕获图的固定行数。
    pub max_batch: u32,
    /// 每序列 block 上限。block table 的固定列数（DESIGN §4.4 要求行数与列数都固定）。
    pub max_blocks: u32,
    /// 每 block 的 token 数。SoC 相关（A2/V4.1 为 128）。
    pub block_size: u32,
    pub vocab_size: u32,
    /// 采样候选数。同时决定 logits 之外的 top-k 缓冲宽度。
    pub topk: u32,
    /// padding 行读 KV 时指向的那个块。
    ///
    /// **分配器必须让这个块永久常驻**：padding 行会读到它，一旦被回收，
    /// dummy 行就在读已释放的 HBM —— 读本身不崩，但读到什么取决于别人写了什么，
    /// 于是"同一个 graph 两次运行结果不同"，且只在 batch 不满桶时出现。
    pub dummy_block: i32,
    /// KV cache 总字节数。由 arena（Track E）决定，plan 只照单预留。
    /// 这里刻意不去推演层数/头数/dtype —— 那是 E 的职责，两边各算一遍必然对不上。
    pub kv_cache_bytes: u64,
}

/// 步级缓冲的用途。每个用途在 plan 期**恰好申请一次**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    TokenIds,
    Positions,
    BlockTable,
    SeqLens,
    SlotMapping,
    KvCache,
    Logits,
    TopkValues,
    TopkIndices,
    LogSumexp,
    SamplingMask,
}

impl Purpose {
    pub const ALL: [Purpose; 11] = [
        Purpose::TokenIds,
        Purpose::Positions,
        Purpose::BlockTable,
        Purpose::SeqLens,
        Purpose::SlotMapping,
        Purpose::KvCache,
        Purpose::Logits,
        Purpose::TopkValues,
        Purpose::TopkIndices,
        Purpose::LogSumexp,
        Purpose::SamplingMask,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Purpose::TokenIds => "token_ids",
            Purpose::Positions => "positions",
            Purpose::BlockTable => "block_table",
            Purpose::SeqLens => "seq_lens",
            Purpose::SlotMapping => "slot_mapping",
            Purpose::KvCache => "kv_cache",
            Purpose::Logits => "logits",
            Purpose::TopkValues => "topk_values",
            Purpose::TopkIndices => "topk_indices",
            Purpose::LogSumexp => "logsumexp",
            Purpose::SamplingMask => "sampling_mask",
        }
    }
}

/// 一块已预留的区域。`id` 由 `RegionSource` 解释（设备指针、handle、池内偏移都行）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub purpose: Purpose,
    pub id: u64,
    pub bytes: u64,
    pub align: u32,
}

/// 区域来源。真实实现包 `aclrtMalloc`，测试实现只记账。
///
/// 这个 trait 存在的唯一理由是让"申请了几次、什么时候申请"变成**可断言的量**。
/// 没有它，"plan 期一次分配"就只是一句注释。
pub trait RegionSource {
    fn acquire(&mut self, purpose: Purpose, bytes: u64, align: u32) -> Result<Region, PlanError>;
}

/// 11 块区域，按用途命名。用结构体字段而不是 `Vec` 是为了让"一块都不缺"由类型保证。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Regions {
    pub token_ids: Region,
    pub positions: Region,
    pub block_table: Region,
    pub seq_lens: Region,
    pub slot_mapping: Region,
    pub kv_cache: Region,
    pub logits: Region,
    pub topk_values: Region,
    pub topk_indices: Region,
    pub logsumexp: Region,
    pub sampling_mask: Region,
}

impl Regions {
    pub fn total_bytes(&self) -> u64 {
        self.iter().map(|region| region.bytes).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Region> {
        [
            &self.token_ids,
            &self.positions,
            &self.block_table,
            &self.seq_lens,
            &self.slot_mapping,
            &self.kv_cache,
            &self.logits,
            &self.topk_values,
            &self.topk_indices,
            &self.logsumexp,
            &self.sampling_mask,
        ]
        .into_iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    ZeroBatch,
    ZeroBlocks,
    ZeroBlockSize,
    ZeroVocab,
    ZeroTopk,
    TopkExceedsVocab {
        topk: u32,
        vocab_size: u32,
    },
    NegativeDummyBlock(i32),
    /// 请求行数超出桶。上层必须换桶，不能在这里扩容。
    BatchExceedsBucket {
        requested: u32,
        bucket: u32,
    },
    TooManyBlocks {
        requested: u32,
        cap: u32,
    },
    /// block 数不足以覆盖 `seq_len`：KV 会读到没分配给自己的块。
    BlocksDoNotCoverSeqLen {
        blocks: u32,
        block_size: u32,
        seq_len: i32,
    },
    InvalidSeqLen(i32),
    InvalidPosition {
        position: i32,
        seq_len: i32,
    },
    NegativeTokenId(i32),
    /// token id 超出词表 = embedding 查表越界读。
    TokenIdOutOfVocab {
        token_id: i32,
        vocab_size: u32,
    },
    /// 负块号 = 越界设备指针。
    NegativeBlockId(i32),
    SizeOverflow,
    TotalTooLarge {
        bytes: u64,
        cap: u64,
    },
}

impl PlanError {
    pub fn describe(&self) -> &'static str {
        match self {
            PlanError::ZeroBatch => "batch 桶必须 >= 1",
            PlanError::ZeroBlocks => "max_blocks 必须 >= 1",
            PlanError::ZeroBlockSize => "block_size 必须 >= 1",
            PlanError::ZeroVocab => "vocab_size 必须 >= 1",
            PlanError::ZeroTopk => "topk 必须 >= 1",
            PlanError::TopkExceedsVocab { .. } => "topk 不能超过 vocab_size",
            PlanError::NegativeDummyBlock(_) => "dummy_block 必须 >= 0",
            PlanError::BatchExceedsBucket { .. } => "行数超出桶：上层应换桶换 plan",
            PlanError::TooManyBlocks { .. } => "序列占用的 block 数超出 max_blocks",
            PlanError::BlocksDoNotCoverSeqLen { .. } => "block 数不足以覆盖 seq_len：KV 会越界读",
            PlanError::InvalidSeqLen(_) => "seq_len 必须 >= 1",
            PlanError::InvalidPosition { .. } => {
                "decode 一步只追加一个 token，position 必须为 seq_len - 1"
            }
            PlanError::NegativeTokenId(_) => "token id 必须 >= 0",
            PlanError::TokenIdOutOfVocab { .. } => "token id 超出词表：embedding 会越界读",
            PlanError::NegativeBlockId(_) => "块号必须 >= 0：负块号等于越界设备指针",
            PlanError::SizeOverflow => "缓冲大小计算溢出",
            PlanError::TotalTooLarge { .. } => "预留总量超出上限",
        }
    }
}

fn mul(a: u64, b: u64) -> Result<u64, PlanError> {
    a.checked_mul(b).ok_or(PlanError::SizeOverflow)
}

/// 采样位图的 32 位字数。
pub fn mask_words(vocab_size: u32) -> Result<u64, PlanError> {
    Ok(u64::from(vocab_size).div_ceil(32))
}

/// 某用途需要多少字节。**只做乘法和边界检查，不做分配**。
pub fn region_bytes(config: &PlanConfig, purpose: Purpose) -> Result<u64, PlanError> {
    let batch = u64::from(config.max_batch);
    match purpose {
        Purpose::TokenIds | Purpose::Positions | Purpose::SeqLens | Purpose::SlotMapping => {
            mul(batch, 4)
        }
        Purpose::BlockTable => mul(mul(batch, u64::from(config.max_blocks))?, 4),
        Purpose::KvCache => Ok(config.kv_cache_bytes),
        Purpose::Logits => mul(mul(batch, u64::from(config.vocab_size))?, 4),
        Purpose::TopkValues | Purpose::TopkIndices => mul(mul(batch, u64::from(config.topk))?, 4),
        Purpose::LogSumexp => mul(batch, 4),
        Purpose::SamplingMask => mul(mul(batch, mask_words(config.vocab_size)?)?, 4),
    }
}

impl PlanConfig {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.max_batch == 0 {
            return Err(PlanError::ZeroBatch);
        }
        if self.max_blocks == 0 {
            return Err(PlanError::ZeroBlocks);
        }
        if self.block_size == 0 {
            return Err(PlanError::ZeroBlockSize);
        }
        if self.vocab_size == 0 {
            return Err(PlanError::ZeroVocab);
        }
        if self.topk == 0 {
            return Err(PlanError::ZeroTopk);
        }
        if self.topk > self.vocab_size {
            return Err(PlanError::TopkExceedsVocab {
                topk: self.topk,
                vocab_size: self.vocab_size,
            });
        }
        if self.dummy_block < 0 {
            return Err(PlanError::NegativeDummyBlock(self.dummy_block));
        }

        let mut total = 0_u64;
        for purpose in Purpose::ALL {
            total = total
                .checked_add(region_bytes(self, purpose)?)
                .ok_or(PlanError::SizeOverflow)?;
        }
        if total > MAX_TOTAL_BYTES {
            return Err(PlanError::TotalTooLarge {
                bytes: total,
                cap: MAX_TOTAL_BYTES,
            });
        }
        Ok(())
    }
}

/// 一个已定型、已分配完毕的步级 plan。
///
/// 泛型而不是 `Box<dyn RegionSource>`：来源在进程生命周期内唯一，
/// 没有理由为它留一层间接跳转，也没有理由让它占据热路径。
pub struct Plan<S: RegionSource> {
    config: PlanConfig,
    regions: Regions,
    staging: HostStaging,
    source: S,
}

impl<S: RegionSource> Plan<S> {
    /// **唯一发生分配的地方。**
    ///
    /// 11 块区域各申请一次，host staging 各分配一次。
    /// 失败时直接返回错误，已申请的区域由 `RegionSource` 的 `Drop` 负责回收
    /// （RAII 契约：`acquire` 成功即转移所有权）。
    pub fn new(config: PlanConfig, mut source: S) -> Result<Self, PlanError> {
        config.validate()?;

        let mut acquire = |purpose: Purpose| -> Result<Region, PlanError> {
            let bytes = region_bytes(&config, purpose)?;
            let region = source.acquire(purpose, bytes, DEVICE_ALIGN)?;
            // 来源必须照单交付。"悄悄多给一点"会掩盖调用方的计算错误。
            if region.bytes != bytes {
                return Err(PlanError::SizeOverflow);
            }
            Ok(region)
        };

        let regions = Regions {
            token_ids: acquire(Purpose::TokenIds)?,
            positions: acquire(Purpose::Positions)?,
            block_table: acquire(Purpose::BlockTable)?,
            seq_lens: acquire(Purpose::SeqLens)?,
            slot_mapping: acquire(Purpose::SlotMapping)?,
            kv_cache: acquire(Purpose::KvCache)?,
            logits: acquire(Purpose::Logits)?,
            topk_values: acquire(Purpose::TopkValues)?,
            topk_indices: acquire(Purpose::TopkIndices)?,
            logsumexp: acquire(Purpose::LogSumexp)?,
            sampling_mask: acquire(Purpose::SamplingMask)?,
        };

        let staging = HostStaging::new(&config)?;

        Ok(Self {
            config,
            regions,
            staging,
            source,
        })
    }

    pub fn config(&self) -> &PlanConfig {
        &self.config
    }

    pub fn regions(&self) -> &Regions {
        &self.regions
    }

    pub fn staging(&self) -> &HostStaging {
        &self.staging
    }

    /// 把来源还回去（例如需要在别的线程继续用它）。
    pub fn into_source(self) -> S {
        self.source
    }

    /// 预留总字节数。用于报告预算，不做决策。
    pub fn reserved_bytes(&self) -> u64 {
        self.regions.total_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSource {
        acquired: Vec<Purpose>,
    }

    impl RegionSource for FakeSource {
        fn acquire(
            &mut self,
            purpose: Purpose,
            bytes: u64,
            align: u32,
        ) -> Result<Region, PlanError> {
            self.acquired.push(purpose);
            Ok(Region {
                purpose,
                id: self.acquired.len() as u64,
                bytes,
                align,
            })
        }
    }

    fn config() -> PlanConfig {
        PlanConfig {
            max_batch: 4,
            max_blocks: 8,
            block_size: 4,
            vocab_size: 100,
            topk: 8,
            dummy_block: 0,
            kv_cache_bytes: 4096,
        }
    }

    #[test]
    fn every_purpose_is_acquired_exactly_once() {
        let plan = Plan::new(
            config(),
            FakeSource {
                acquired: Vec::new(),
            },
        )
        .unwrap();
        let acquired = plan.into_source().acquired;
        assert_eq!(acquired.len(), Purpose::ALL.len());
        for purpose in Purpose::ALL {
            assert_eq!(
                acquired.iter().filter(|&&p| p == purpose).count(),
                1,
                "{purpose:?} was not acquired exactly once"
            );
        }
    }

    #[test]
    fn region_sizes_follow_the_config() {
        let config = config();
        assert_eq!(region_bytes(&config, Purpose::TokenIds).unwrap(), 4 * 4);
        assert_eq!(
            region_bytes(&config, Purpose::BlockTable).unwrap(),
            4 * 8 * 4
        );
        assert_eq!(region_bytes(&config, Purpose::Logits).unwrap(), 4 * 100 * 4);
        assert_eq!(
            region_bytes(&config, Purpose::TopkValues).unwrap(),
            4 * 8 * 4
        );
        // 100 个 vocab -> ceil(100/32) = 4 个 32 位字
        assert_eq!(
            region_bytes(&config, Purpose::SamplingMask).unwrap(),
            4 * 4 * 4
        );
        assert_eq!(region_bytes(&config, Purpose::KvCache).unwrap(), 4096);
    }

    #[test]
    fn mask_words_rounds_up() {
        assert_eq!(mask_words(1).unwrap(), 1);
        assert_eq!(mask_words(32).unwrap(), 1);
        assert_eq!(mask_words(33).unwrap(), 2);
        assert_eq!(mask_words(128_256).unwrap(), 4008);
    }

    #[test]
    fn config_validation_covers_every_knob() {
        let bad = |mutate: fn(&mut PlanConfig)| {
            let mut c = config();
            mutate(&mut c);
            c.validate().unwrap_err()
        };
        assert_eq!(bad(|c| c.max_batch = 0), PlanError::ZeroBatch);
        assert_eq!(bad(|c| c.max_blocks = 0), PlanError::ZeroBlocks);
        assert_eq!(bad(|c| c.block_size = 0), PlanError::ZeroBlockSize);
        assert_eq!(bad(|c| c.vocab_size = 0), PlanError::ZeroVocab);
        assert_eq!(bad(|c| c.topk = 0), PlanError::ZeroTopk);
        assert_eq!(
            bad(|c| c.topk = 101),
            PlanError::TopkExceedsVocab {
                topk: 101,
                vocab_size: 100
            }
        );
        assert_eq!(
            bad(|c| c.dummy_block = -1),
            PlanError::NegativeDummyBlock(-1)
        );
    }

    /// 写错一个数量级必须变成错误，不能绕回成一个小数字然后 OOM。
    /// 两种失败模式要分开测 —— 它们指向不同的错误（算术溢出 vs 预算超限）。
    #[test]
    fn absurd_config_errors_instead_of_overflowing() {
        // (a) 中间乘积溢出 u64：u32::MAX × u32::MAX × 4 ≈ 7.4e19 > u64::MAX
        let mut overflow = config();
        overflow.max_batch = u32::MAX;
        overflow.max_blocks = u32::MAX;
        assert_eq!(overflow.validate(), Err(PlanError::SizeOverflow));

        // (b) 乘积没溢出，但总量超过 1 TiB 预算
        let mut too_large = config();
        too_large.max_batch = 4096;
        too_large.vocab_size = 100_000_000;
        too_large.topk = 4;
        // logits = 4096 × 1e8 × 4 B = 1.6e12 B > 1 TiB
        assert!(
            matches!(too_large.validate(), Err(PlanError::TotalTooLarge { .. })),
            "expected TotalTooLarge, got {:?}",
            too_large.validate()
        );
    }

    #[test]
    fn plan_rejects_a_source_that_returns_the_wrong_size() {
        struct Liar;
        impl RegionSource for Liar {
            fn acquire(
                &mut self,
                purpose: Purpose,
                bytes: u64,
                align: u32,
            ) -> Result<Region, PlanError> {
                Ok(Region {
                    purpose,
                    id: 0,
                    bytes: bytes + 1,
                    align,
                })
            }
        }
        // 用 .err() 而不是 unwrap_err()：不为了让测试能打印错误就给 Plan 加 Debug，
        // 那会诱导后来人把整个 plan 内容打进日志。
        assert_eq!(
            Plan::new(config(), Liar).err(),
            Some(PlanError::SizeOverflow)
        );
    }
}
