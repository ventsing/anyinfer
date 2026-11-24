//! host 侧 staging 缓冲与一步的填充逻辑。
//!
//! # 三条不可协商的规则
//!
//! 1. **所有缓冲在构造时一次分配，长度永远等于桶形状**。
//!    捕获图按固定形状录制，所以传给 ABI 的每一个指针后面都必须是满长度的缓冲，
//!    不允许"这次只有 3 行就传 3 行"。
//!
//! 2. **每一步都整块重写，不留残留**。
//!    只写"用到的那一段"是最隐蔽的 bug 源：上一步某个序列占了块 5、6、7，
//!    这一步它只占块 5，那么块表里 6、7 位上的旧值会被注意力核读到 ——
//!    读到的是别的序列已经释放的 HBM。这不会崩，只会让结果依赖历史。
//!
//! 3. **先全部校验，再全部写入**。
//!    "写了一半才发现第 2 行非法"会留下一个内部不自洽的 step，
//!    而调用方可能已经把它的一部分当合法数据用了。所以校验阶段只读不写。

use crate::{mul, PlanConfig, PlanError};

/// padding 行的 KV 长度。
///
/// 取 1 而不是 0：长度 0 在部分 attention 实现里会走到除零或空归约分支，
/// 而长度 1 是一个确定的、能正常算完的最小形状（DESIGN §4.4）。
pub const PAD_SEQ_LEN: i32 = 1;

/// padding 行的 `slot_mapping`。**KV 写入侧必须跳过 -1。**
///
/// 用 -1 而不是 0：0 是一个完全合法的 slot，落到 0 上就等于把 dummy 行的
/// hidden state 写进了别人的 KV —— 这是 DESIGN §4.4 说的"轨迹永久污染"。
pub const SLOT_NO_WRITE: i32 = -1;

pub const PAD_TOKEN_ID: i32 = 0;
pub const PAD_POSITION: i32 = 0;

/// host 侧 staging。长度全部由 `PlanConfig` 决定，构造后不变。
pub struct HostStaging {
    pub token_ids: Vec<i32>,
    pub positions: Vec<i32>,
    /// `max_batch × max_blocks`，行优先。
    pub block_table: Vec<i32>,
    pub seq_lens: Vec<i32>,
    pub slot_mapping: Vec<i32>,
    /// `max_batch × topk`。设备写入，plan 只留位置。
    pub topk_values: Vec<f32>,
    pub topk_indices: Vec<i32>,
    pub logsumexp: Vec<f32>,
    /// `max_batch × ceil(vocab/32)`。
    pub sampling_mask: Vec<u32>,
}

impl HostStaging {
    /// **plan 期唯一一次 host 分配。**
    pub fn new(config: &PlanConfig) -> Result<Self, PlanError> {
        config.validate()?;
        let batch = config.max_batch as usize;
        let blocks = config.max_blocks as usize;
        let topk = config.topk as usize;
        let words = crate::mask_words(config.vocab_size)? as usize;

        // 用 checked 乘后再转 usize：32 位目标上 usize 可能装不下 u64 的乘积
        let block_entries = mul(batch as u64, blocks as u64)?;
        let topk_entries = mul(batch as u64, topk as u64)?;
        let mask_entries = mul(batch as u64, words as u64)?;

        Ok(Self {
            token_ids: vec![PAD_TOKEN_ID; batch],
            positions: vec![PAD_POSITION; batch],
            block_table: vec![config.dummy_block; usize::try_from(block_entries).map_err(|_| PlanError::SizeOverflow)?],
            seq_lens: vec![PAD_SEQ_LEN; batch],
            slot_mapping: vec![SLOT_NO_WRITE; batch],
            topk_values: vec![0.0; usize::try_from(topk_entries).map_err(|_| PlanError::SizeOverflow)?],
            topk_indices: vec![-1; usize::try_from(topk_entries).map_err(|_| PlanError::SizeOverflow)?],
            logsumexp: vec![0.0; batch],
            sampling_mask: vec![0; usize::try_from(mask_entries).map_err(|_| PlanError::SizeOverflow)?],
        })
    }
}

/// decode 一步里的一行。**没有 position 字段**——decode 一步只追加一个 token，
/// 所以 `position ≡ seq_len - 1`。让它可以被单独传，就等于留了一个能和 `seq_len` 打架的入口。
#[derive(Debug, Clone, Copy)]
pub struct DecodeRow<'a> {
    pub token_id: i32,
    /// 加上本步 token 之后的 KV 长度。
    pub seq_len: i32,
    /// 该序列当前占用的物理块，按逻辑顺序。
    pub block_ids: &'a [i32],
}

/// 一步的只读视图。所有切片都是**桶的满长度**，可以直接喂给 `AnyInferStep`。
///
/// 输出缓冲（`topk_*` / `logsumexp` / `sampling_mask`）只有前 `rows` 行有意义。
#[derive(Debug, Clone, Copy)]
pub struct StepView<'a> {
    /// 实际行数。其余是 padding 行。
    pub rows: u32,
    pub token_ids: &'a [i32],
    pub positions: &'a [i32],
    pub block_table: &'a [i32],
    pub seq_lens: &'a [i32],
    pub slot_mapping: &'a [i32],
    pub topk_values: &'a [f32],
    pub topk_indices: &'a [i32],
    pub logsumexp: &'a [f32],
    pub sampling_mask: &'a [u32],
    pub max_blocks: u32,
    pub block_size: u32,
    pub topk: u32,
    pub mask_words: u32,
}

impl StepView<'_> {
    /// 第 `row` 行的 block 表切片。
    pub fn block_row(&self, row: u32) -> &[i32] {
        let start = row as usize * self.max_blocks as usize;
        &self.block_table[start..start + self.max_blocks as usize]
    }

    /// 第 `row` 行的采样位图切片。
    pub fn mask_row(&self, row: u32) -> &[u32] {
        let start = row as usize * self.mask_words as usize;
        &self.sampling_mask[start..start + self.mask_words as usize]
    }
}

/// 校验一行。**只读**，不碰 staging。
fn validate_row(config: &PlanConfig, row: &DecodeRow<'_>) -> Result<(), PlanError> {
    if row.token_id < 0 {
        return Err(PlanError::NegativeTokenId(row.token_id));
    }
    if row.token_id as u32 >= config.vocab_size {
        return Err(PlanError::TokenIdOutOfVocab {
            token_id: row.token_id,
            vocab_size: config.vocab_size,
        });
    }
    if row.seq_len < 1 {
        return Err(PlanError::InvalidSeqLen(row.seq_len));
    }
    if row.block_ids.len() > config.max_blocks as usize {
        return Err(PlanError::TooManyBlocks {
            requested: row.block_ids.len() as u32,
            cap: config.max_blocks,
        });
    }
    // block 数必须覆盖 seq_len，否则注意力会读到没分配给自己的块
    let capacity = row.block_ids.len() as u64 * u64::from(config.block_size);
    if capacity < row.seq_len as u64 {
        return Err(PlanError::BlocksDoNotCoverSeqLen {
            blocks: row.block_ids.len() as u32,
            block_size: config.block_size,
            seq_len: row.seq_len,
        });
    }
    // 负块号等于越界指针，且上面那个"是否覆盖"的检查会因为符号而失真
    if let Some(&bad) = row.block_ids.iter().find(|&&id| id < 0) {
        return Err(PlanError::NegativeBlockId(bad));
    }
    Ok(())
}

impl<S: crate::RegionSource> crate::Plan<S> {
    /// 填充一式 decode step。
    ///
    /// 顺序是：校验全部行 → 写实际行 → 写 padding 行。
    /// 校验失败时 staging **一个字节都没动**。
    pub fn prepare_decode_step<'a>(
        &'a mut self,
        rows: &[DecodeRow<'_>],
    ) -> Result<StepView<'a>, PlanError> {
        let config = *self.config();
        let batch = config.max_batch as usize;
        let blocks = config.max_blocks as usize;
        let block_size = config.block_size as usize;

        if rows.len() > batch {
            return Err(PlanError::BatchExceedsBucket {
                requested: rows.len() as u32,
                bucket: config.max_batch,
            });
        }
        for row in rows {
            validate_row(&config, row)?;
        }

        // ---- 写入。到这里已经没有任何失败路径，staging 要么全写要么没动。
        let staging = &mut self.staging;
        let dummy = config.dummy_block;

        for (index, row) in rows.iter().enumerate() {
            let position = row.seq_len - 1;
            staging.token_ids[index] = row.token_id;
            staging.positions[index] = position;
            staging.seq_lens[index] = row.seq_len;

            // 整行重写：先铺 dummy，再覆盖实际用到的前缀
            let base = index * blocks;
            staging.block_table[base..base + blocks].fill(dummy);
            staging.block_table[base..base + row.block_ids.len()].copy_from_slice(row.block_ids);

            // slot 由块表**推导**，不接受调用方另给一份：
            // 两份数据一旦不一致，KV 就会写进错位置，而且查起来极难。
            let logical_block = position as usize / block_size;
            let offset = position as usize % block_size;
            let physical_block = row.block_ids[logical_block];
            staging.slot_mapping[index] = physical_block * block_size as i32 + offset as i32;
        }

        for index in rows.len()..batch {
            staging.token_ids[index] = PAD_TOKEN_ID;
            staging.positions[index] = PAD_POSITION;
            staging.seq_lens[index] = PAD_SEQ_LEN;
            staging.slot_mapping[index] = SLOT_NO_WRITE;
            let base = index * blocks;
            staging.block_table[base..base + blocks].fill(dummy);
        }

        let words = crate::mask_words(config.vocab_size)? as u32;
        let rows_len = rows.len() as u32;
        let staging = &self.staging;
        Ok(StepView {
            rows: rows_len,
            token_ids: &staging.token_ids,
            positions: &staging.positions,
            block_table: &staging.block_table,
            seq_lens: &staging.seq_lens,
            slot_mapping: &staging.slot_mapping,
            topk_values: &staging.topk_values,
            topk_indices: &staging.topk_indices,
            logsumexp: &staging.logsumexp,
            sampling_mask: &staging.sampling_mask,
            max_blocks: config.max_blocks,
            block_size: config.block_size,
            topk: config.topk,
            mask_words: words,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Plan, PlanConfig, Purpose, Region, RegionSource};

    struct FakeSource;
    impl RegionSource for FakeSource {
        fn acquire(&mut self, purpose: Purpose, bytes: u64, align: u32) -> Result<Region, PlanError> {
            Ok(Region { purpose, id: 0, bytes, align })
        }
    }

    fn config() -> PlanConfig {
        PlanConfig {
            max_batch: 4,
            max_blocks: 4,
            block_size: 4,
            vocab_size: 64,
            topk: 4,
            dummy_block: 0,
            kv_cache_bytes: 1024,
        }
    }

    fn plan() -> Plan<FakeSource> {
        Plan::new(config(), FakeSource).unwrap()
    }

    #[test]
    fn slot_mapping_is_derived_from_the_block_table() {
        let mut plan = plan();
        let blocks = [10, 20, 30, 40];
        // seq_len = 5 -> position = 4 -> 逻辑块 1 -> 物理块 20 -> slot = 20*4 + 0 = 80
        let view = plan
            .prepare_decode_step(&[DecodeRow { token_id: 7, seq_len: 5, block_ids: &blocks[..2] }])
            .unwrap();
        assert_eq!(view.positions[0], 4);
        assert_eq!(view.slot_mapping[0], 80);

        // seq_len = 4 -> position = 3 -> 逻辑块 0 -> 物理块 10 -> slot = 10*4 + 3 = 43
        let view = plan
            .prepare_decode_step(&[DecodeRow { token_id: 7, seq_len: 4, block_ids: &blocks[..1] }])
            .unwrap();
        assert_eq!(view.positions[0], 3);
        assert_eq!(view.slot_mapping[0], 43);
    }

    #[test]
    fn padding_rows_are_isolated_on_both_ends() {
        let mut plan = plan();
        let blocks = [5, 6, 7, 8];
        let view = plan
            .prepare_decode_step(&[DecodeRow { token_id: 3, seq_len: 9, block_ids: &blocks }])
            .unwrap();
        assert_eq!(view.rows, 1);

        for row in 1..4u32 {
            assert_eq!(view.seq_lens[row as usize], PAD_SEQ_LEN, "读侧：KV 长度必须是 1");
            assert_eq!(view.positions[row as usize], PAD_POSITION);
            assert_eq!(
                view.slot_mapping[row as usize], SLOT_NO_WRITE,
                "写侧：dummy 行绝不落 KV"
            );
            assert_eq!(
                view.block_row(row),
                &[0, 0, 0, 0],
                "dummy 行只能指向 dummy_block"
            );
        }
        // 实际行不受影响
        assert_eq!(view.seq_lens[0], 9);
        assert_eq!(view.block_row(0), &[5, 6, 7, 8]);
    }

    /// 上一步用了 4 个块、这一步只用了 1 个：块表里剩下的位置必须被清掉。
    /// 否则注意力会读到上一个序列已经释放的 HBM。
    #[test]
    fn block_table_tail_never_keeps_stale_blocks() {
        let mut plan = plan();
        let wide = [11, 12, 13, 14];
        plan.prepare_decode_step(&[DecodeRow { token_id: 1, seq_len: 16, block_ids: &wide }])
            .unwrap();

        let narrow = [21];
        let view = plan
            .prepare_decode_step(&[DecodeRow { token_id: 1, seq_len: 1, block_ids: &narrow }])
            .unwrap();
        assert_eq!(view.block_row(0), &[21, 0, 0, 0], "stale blocks survived");
    }

    #[test]
    fn stale_rows_beyond_the_current_batch_are_reset() {
        let mut plan = plan();
        let blocks = [9];
        // 先填满 3 个实际行
        plan.prepare_decode_step(&[
            DecodeRow { token_id: 2, seq_len: 1, block_ids: &blocks },
            DecodeRow { token_id: 2, seq_len: 1, block_ids: &blocks },
            DecodeRow { token_id: 2, seq_len: 1, block_ids: &blocks },
        ])
        .unwrap();
        // 再来只有 1 个实际行
        let view = plan
            .prepare_decode_step(&[DecodeRow { token_id: 2, seq_len: 1, block_ids: &blocks }])
            .unwrap();
        for row in 1..4usize {
            assert_eq!(view.slot_mapping[row], SLOT_NO_WRITE, "row {row} not reset");
            assert_eq!(view.seq_lens[row], PAD_SEQ_LEN, "row {row} not reset");
        }
    }

    /// 校验失败时必须一字节都没写：否则调用方拿到的是一个半新半旧的 step。
    #[test]
    fn failed_validation_leaves_staging_untouched() {
        let mut plan = plan();
        let good = [3];
        plan.prepare_decode_step(&[DecodeRow { token_id: 5, seq_len: 1, block_ids: &good }])
            .unwrap();
        let before = snapshot(&plan);

        let bad_rows = [
            DecodeRow { token_id: 6, seq_len: 1, block_ids: &good },
            // 第 2 行非法：block 数不足
            DecodeRow { token_id: 6, seq_len: 64, block_ids: &good },
        ];
        let error = plan.prepare_decode_step(&bad_rows).unwrap_err();
        assert_eq!(
            error,
            PlanError::BlocksDoNotCoverSeqLen { blocks: 1, block_size: 4, seq_len: 64 }
        );
        assert_eq!(snapshot(&plan), before, "staging was partially overwritten");
    }

    fn snapshot(plan: &Plan<FakeSource>) -> Vec<i32> {
        let staging = plan.staging();
        let mut out = Vec::new();
        out.extend_from_slice(&staging.token_ids);
        out.extend_from_slice(&staging.positions);
        out.extend_from_slice(&staging.block_table);
        out.extend_from_slice(&staging.seq_lens);
        out.extend_from_slice(&staging.slot_mapping);
        out
    }

    #[test]
    fn row_validation_rejects_each_invariant() {
        let mut plan = plan();
        let fine = [1];
        assert_eq!(
            plan.prepare_decode_step(&[DecodeRow { token_id: -1, seq_len: 1, block_ids: &fine }])
                .unwrap_err(),
            PlanError::NegativeTokenId(-1)
        );
        assert_eq!(
            plan.prepare_decode_step(&[DecodeRow { token_id: 0, seq_len: 0, block_ids: &fine }])
                .unwrap_err(),
            PlanError::InvalidSeqLen(0)
        );
        assert_eq!(
            plan.prepare_decode_step(&[DecodeRow { token_id: 0, seq_len: 1, block_ids: &[1, 2, 3, 4, 5] }])
                .unwrap_err(),
            PlanError::TooManyBlocks { requested: 5, cap: 4 }
        );
        assert_eq!(
            plan.prepare_decode_step(&[DecodeRow { token_id: 0, seq_len: 1, block_ids: &[-3] }])
                .unwrap_err(),
            PlanError::NegativeBlockId(-3)
        );
    }

    #[test]
    fn too_many_rows_is_a_bucket_error_not_a_reallocation() {
        let mut plan = plan();
        let blocks = [1];
        let row = DecodeRow { token_id: 0, seq_len: 1, block_ids: &blocks };
        let error = plan.prepare_decode_step(&[row, row, row, row, row]).unwrap_err();
        assert_eq!(
            error,
            PlanError::BatchExceedsBucket { requested: 5, bucket: 4 }
        );
    }

    #[test]
    fn empty_batch_is_all_padding() {
        let mut plan = plan();
        let view = plan.prepare_decode_step(&[]).unwrap();
        assert_eq!(view.rows, 0);
        for row in 0..4usize {
            assert_eq!(view.slot_mapping[row], SLOT_NO_WRITE);
            assert_eq!(view.seq_lens[row], PAD_SEQ_LEN);
        }
    }

    /// staging 的长度与 `region_bytes` 是**两条独立算出来的公式**：
    /// 一条给 host 缓冲，一条给设备预留。两者一旦不一致，H2D 就会少拷或多拷一段 —— 而且不会报错。
    #[test]
    fn staging_lengths_match_reserved_region_sizes() {
        for config in [
            config(),
            // 非 2 的幂：ceil 类的取整错误只会在这里露出来
            PlanConfig { max_batch: 3, max_blocks: 5, block_size: 7, vocab_size: 100, topk: 3, dummy_block: 0, kv_cache_bytes: 0 },
            PlanConfig { max_batch: 1, max_blocks: 1, block_size: 1, vocab_size: 1, topk: 1, dummy_block: 0, kv_cache_bytes: 0 },
        ] {
            let staging = HostStaging::new(&config).unwrap();
            let expect = |len: usize, purpose: Purpose, name: &str| {
                assert_eq!(
                    len as u64 * 4,
                    crate::region_bytes(&config, purpose).unwrap(),
                    "{name} host length disagrees with reserved device size"
                );
            };
            expect(staging.token_ids.len(), Purpose::TokenIds, "token_ids");
            expect(staging.positions.len(), Purpose::Positions, "positions");
            expect(staging.block_table.len(), Purpose::BlockTable, "block_table");
            expect(staging.seq_lens.len(), Purpose::SeqLens, "seq_lens");
            expect(staging.slot_mapping.len(), Purpose::SlotMapping, "slot_mapping");
            expect(staging.topk_values.len(), Purpose::TopkValues, "topk_values");
            expect(staging.topk_indices.len(), Purpose::TopkIndices, "topk_indices");
            expect(staging.logsumexp.len(), Purpose::LogSumexp, "logsumexp");
            expect(staging.sampling_mask.len(), Purpose::SamplingMask, "sampling_mask");
        }
    }

    #[test]
    fn mask_region_covers_every_token_exactly_once() {
        // 位图必须能放下 vocab 个 bit，且不浪费整字：100 -> 4 字 = 128 bit
        let config = PlanConfig { vocab_size: 100, ..config() };
        let staging = HostStaging::new(&config).unwrap();
        let total_bits = staging.sampling_mask.len() / config.max_batch as usize * 32;
        assert!(total_bits >= config.vocab_size as usize);
        assert!(total_bits < config.vocab_size as usize + 32);
    }

    #[test]
    fn view_rows_index_block_and_mask_correctly() {
        let mut plan = plan();
        let blocks = [1, 2, 3, 4];
        let view = plan
            .prepare_decode_step(&[
                DecodeRow { token_id: 0, seq_len: 13, block_ids: &blocks },
                DecodeRow { token_id: 0, seq_len: 1, block_ids: &blocks[..1] },
            ])
            .unwrap();
        assert_eq!(view.block_row(0), &[1, 2, 3, 4]);
        assert_eq!(view.block_row(1), &[1, 0, 0, 0]);
        // vocab 64 -> 2 个 32 位字
        assert_eq!(view.mask_words, 2);
        assert_eq!(view.mask_row(3).len(), 2);
    }
}
