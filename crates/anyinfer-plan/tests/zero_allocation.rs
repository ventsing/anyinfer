//! D-02 的核心断言：**plan 期一次分配，稳态每步零分配。**
//!
//! 手法沿用原型 `tests/host_decode_allocations.rs`：装一个计数用 `#[global_allocator]`，
//! 计数器放在 thread-local 里 —— 于是只统计本测试线程的调用，
//! cargo 的测试框架线程、别的用例都不计入。
//!
//! 光有"零分配"断言是不够的，因为它对"钩子根本没接上"同样成立。
//! 所以第一个用例是**正控制**：窗口内主动分配一次，断言确实被数到。

use anyinfer_plan::{DecodeRow, Plan, PlanConfig, PlanError, Purpose, Region, RegionSource};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Calls {
    allocate: u64,
    reallocate: u64,
    free: u64,
}

thread_local! {
    /// `None` = 不在测量窗口内，不计数。
    static CALLS: Cell<Option<Calls>> = const { Cell::new(None) };
}

fn record(update: impl FnOnce(&mut Calls)) {
    let _ = CALLS.try_with(|counter| {
        if let Some(mut snapshot) = counter.get() {
            update(&mut snapshot);
            counter.set(Some(snapshot));
        }
    });
}

struct CountingAllocator;

// SAFETY: 每个操作都把参数原样转发给 System，计数用 const 初始化的 TLS，自身不分配。
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(|c| c.allocate += 1);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(|c| c.allocate += 1);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(|c| c.reallocate += 1);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(|c| c.free += 1);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(body: impl FnOnce()) -> Calls {
    CALLS.set(Some(Calls::default()));
    body();
    CALLS.replace(None).unwrap()
}

/// 记账用来源。**自己不做任何分配**（用定长数组），于是 plan 期的分配次数
/// 就精确等于 staging 缓冲的个数，这个数字才有意义。
struct CountingSource {
    acquired: [Option<Purpose>; 11],
    count: usize,
}

impl CountingSource {
    fn new() -> Self {
        Self {
            acquired: [None; 11],
            count: 0,
        }
    }
}

impl RegionSource for CountingSource {
    fn acquire(&mut self, purpose: Purpose, bytes: u64, align: u32) -> Result<Region, PlanError> {
        if self.count < 11 {
            self.acquired[self.count] = Some(purpose);
        }
        self.count += 1;
        Ok(Region {
            purpose,
            id: self.count as u64,
            bytes,
            align,
        })
    }
}

fn config() -> PlanConfig {
    PlanConfig {
        max_batch: 4,
        max_blocks: 4,
        block_size: 4,
        vocab_size: 32,
        topk: 4,
        dummy_block: 0,
        kv_cache_bytes: 8192,
    }
}

/// 正控制：钩子必须是活的。
///
/// 没有这一条，下面所有 `assert_eq!(observed, Calls::default())` 都可以因为
/// 钩子根本没生效而通过 —— 那就是一个永远绿的、什么都不检查的门。
#[test]
fn the_allocation_hook_actually_counts() {
    let observed = measure(|| {
        let v: Vec<u8> = Vec::with_capacity(64);
        assert_eq!(v.capacity(), 64);
    });
    assert_eq!(observed.allocate, 1, "hook did not see the allocation");
    assert_eq!(observed.reallocate, 0);

    // capacity 1 装得下 1 个元素，第二个才撑大 —— 所以这里必须是两次 push
    let observed = measure(|| {
        let mut v: Vec<u8> = Vec::with_capacity(1);
        v.push(1);
        v.push(2);
        assert_eq!(v.len(), 2);
    });
    assert_eq!(observed.allocate, 1);
    assert_eq!(observed.reallocate, 1, "hook did not see the reallocation");
}

/// plan 期：恰好 9 次分配（staging 的 9 个缓冲），一次不多。
///
/// 把这个数字钉死是有意的摩擦：以后有人往 `HostStaging` 里加缓冲却没想清楚
/// "它该不该在 plan 期分配"，这个测试会立刻红。
#[test]
fn plan_creation_allocates_each_staging_buffer_exactly_once() {
    let observed = measure(|| {
        let plan = Plan::new(config(), CountingSource::new()).unwrap();
        assert_eq!(plan.reserved_bytes(), 8192 + 16 + 16 + 64 + 16 + 16 + 512 + 64 + 64 + 16 + 16);
        drop(plan);
    });
    assert_eq!(observed.allocate, 9, "staging should be 9 one-shot allocations");
    assert_eq!(observed.reallocate, 0, "nothing should grow during planning");
}

/// 稳态：4096 步，零分配、零扩容。
#[test]
fn steady_state_decode_never_touches_the_allocator() {
    const STEPS: i32 = 4096;

    let plan_source = CountingSource::new();
    let mut plan = Plan::new(config(), plan_source).unwrap();

    // 每行一个固定的块数组，按需切片 —— 循环内不允许出现新的 Vec
    let blocks_a: [i32; 4] = [1, 2, 3, 4];
    let blocks_b: [i32; 4] = [5, 6, 7, 8];

    let mut rows = [
        DecodeRow { token_id: 3, seq_len: 1, block_ids: &blocks_a[..1] },
        DecodeRow { token_id: 9, seq_len: 1, block_ids: &blocks_b[..1] },
    ];

    // 预热：把任何一次性初始化（包括首次触碰 staging 的第一页）都做掉
    for step in 0..2 {
        rows[0].seq_len = 1 + step;
        rows[1].seq_len = 1 + step;
        let _ = plan.prepare_decode_step(&rows).unwrap();
    }

    let mut checksum = 0_i64;
    let observed = measure(|| {
        for step in 0..STEPS {
            // 让 seq_len 在 1..=16 之间循环，覆盖"块数随长度变化"的路径
            let seq_len = 1 + (step % 16);
            let needed = (seq_len as usize).div_ceil(4);
            rows[0].seq_len = seq_len;
            rows[0].block_ids = &blocks_a[..needed];
            rows[1].seq_len = seq_len;
            rows[1].block_ids = &blocks_b[..needed];

            let view = plan.prepare_decode_step(&rows).unwrap();
            // 真的读一遍产物，防止优化器把整段删掉
            checksum += i64::from(view.rows);
            checksum += i64::from(view.slot_mapping[0]);
            checksum += i64::from(view.slot_mapping[1]);
            checksum += i64::from(view.seq_lens[2]);
            checksum += i64::from(view.block_row(1)[0]);
        }
    });

    assert_eq!(
        observed,
        Calls::default(),
        "steady-state decode allocated: {observed:?}"
    );
    assert_ne!(checksum, 0, "loop body was optimized away");
}

/// 变化形状的输入也必须零分配 —— 只有"超界"才报错，不扩容。
#[test]
fn changing_row_count_stays_within_the_bucket() {
    let mut plan = Plan::new(config(), CountingSource::new()).unwrap();
    let blocks: [i32; 4] = [1, 2, 3, 4];
    let row = DecodeRow { token_id: 1, seq_len: 1, block_ids: &blocks[..1] };

    for count in 0..=4usize {
        let _ = plan.prepare_decode_step(&[row; 4][..count]).unwrap();
    }

    let observed = measure(|| {
        for _ in 0..1024 {
            for count in 0..=4usize {
                let view = plan.prepare_decode_step(&[row; 4][..count]).unwrap();
                assert_eq!(view.rows as usize, count);
            }
        }
    });
    assert_eq!(observed, Calls::default());
}

/// 失败路径也不许分配：错误必须在热路径上"免费"。
#[test]
fn error_path_does_not_allocate() {
    let mut plan = Plan::new(config(), CountingSource::new()).unwrap();
    let blocks: [i32; 4] = [1, 2, 3, 4];
    // 5 行超出 4 行桶
    let row = DecodeRow { token_id: 1, seq_len: 1, block_ids: &blocks[..1] };

    let observed = measure(|| {
        for _ in 0..1024 {
            let error = plan.prepare_decode_step(&[row; 5]).unwrap_err();
            assert_eq!(error, PlanError::BatchExceedsBucket { requested: 5, bucket: 4 });
        }
    });
    assert_eq!(observed, Calls::default(), "error path allocated");
}
