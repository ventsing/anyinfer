//! AnyInfer 的 C ABI 边界 —— 与 `native/abi/anyinfer.h` 一一对应（D-01）。
//!
//! 这个 crate **不做任何计算**，只定义跨界的数据形状。它存在的理由是：
//! 跨语言 ABI 的错误不会在编译期暴露，而是在运行期以"读到别的字段的值"
//! 这种最难查的方式出现。所以把布局本身做成可验证的产物。
//!
//! 三个不许妥协的点（与 C 侧注释一致）：
//!   1. 结构体里不允许隐式 padding —— padding 字节未定义，会破坏"两次构造逐位相等"。
//!   2. sizeof 硬钉住 —— 加字段必须同时改 C 常量与 `native/abi/abi_layout.txt`。
//!   3. 只用 `i32` + 具名常量表示状态码，不用 Rust enum / C enum ——
//!      enum 的底层类型是实现定义的，那层猜测没必要保留。
//!
//! 布局的唯一产物是 `native/abi/abi_layout.txt`，由 `layout_report()` 生成。
//! C 侧和 Rust 侧各自独立地对着这份产物校验，所以任何一侧漂移都会被抓到。

use core::ffi::c_void;

/// ABI 版本。形状不兼容的改动必须 +1，并且两侧同时改。
pub const ABI_VERSION: u32 = 1;

// ---------------------------------------------------------------- 状态码
pub type StatusCode = i32;

pub const OK: StatusCode = 0;
pub const ERR_INVALID_ARG: StatusCode = -1;
pub const ERR_NOT_PLANNED: StatusCode = -2;
pub const ERR_SHAPE: StatusCode = -3;
pub const ERR_DEVICE: StatusCode = -4;
pub const ERR_GRAPH: StatusCode = -5;
pub const ERR_UNSUPPORTED: StatusCode = -6;
pub const ERR_INTERNAL: StatusCode = -7;

/// `failed_op_id` 的哨兵：没有失败的算子。
pub const OP_NONE: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------- 步标志
pub const STEP_FLAG_DECODE: u32 = 1 << 0;
pub const STEP_FLAG_BUCKET_PADDED: u32 = 1 << 1;
pub const STEP_FLAG_TRACE: u32 = 1 << 2;

// ---------------------------------------------------------------- 结构体
/// 失败信息。`failed_op_id` 让误差能归因到具体算子，
/// 否则 RSI 闭环会退化成"整体重跑一遍试试"。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyInferStatus {
    pub code: StatusCode,
    pub failed_op_id: u32,
    pub detail: i32,
    pub reserved0: u32,
    pub step_index: u64,
}

impl AnyInferStatus {
    pub const fn ok() -> Self {
        Self {
            code: OK,
            failed_op_id: OP_NONE,
            detail: 0,
            reserved0: 0,
            step_index: 0,
        }
    }

    pub fn is_ok(&self) -> bool {
        self.code == OK
    }
}

impl Default for AnyInferStatus {
    fn default() -> Self {
        Self::ok()
    }
}

/// 一次 step 的全部输入与输出。所有指针都是设备地址，`stream` 是 aclrtStream。
///
/// 形状字段在同一个捕获图内必须保持不变 —— 变了就是另一个桶，必须换图。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AnyInferStep {
    pub abi_version: u32,
    pub flags: u32,
    pub batch_size: u32,
    pub num_tokens: u32,
    pub max_blocks: u32,
    pub vocab_size: u32,
    pub topk: u32,
    pub reserved0: u32,

    pub step_index: u64,
    pub weight_version: u64,

    pub stream: *mut c_void,

    pub token_ids: *const i32,
    pub positions: *const i32,
    pub block_table: *const i32,
    pub seq_lens: *const i32,
    pub slot_mapping: *const i32,

    pub kv_cache: *mut c_void,

    pub logits: *mut c_void,
    pub topk_values: *mut c_void,
    pub topk_indices: *mut i32,
    pub logsumexp: *mut c_void,
    pub sampling_mask: *mut c_void,
}

impl AnyInferStep {
    /// 全零构造（字段逐个显式写，避免 `mem::zeroed` 在含指针的结构体上引发争议）。
    pub const fn zeroed() -> Self {
        Self {
            abi_version: ABI_VERSION,
            flags: 0,
            batch_size: 0,
            num_tokens: 0,
            max_blocks: 0,
            vocab_size: 0,
            topk: 0,
            reserved0: 0,
            step_index: 0,
            weight_version: 0,
            stream: core::ptr::null_mut(),
            token_ids: core::ptr::null(),
            positions: core::ptr::null(),
            block_table: core::ptr::null(),
            seq_lens: core::ptr::null(),
            slot_mapping: core::ptr::null(),
            kv_cache: core::ptr::null_mut(),
            logits: core::ptr::null_mut(),
            topk_values: core::ptr::null_mut(),
            topk_indices: core::ptr::null_mut(),
            logsumexp: core::ptr::null_mut(),
            sampling_mask: core::ptr::null_mut(),
        }
    }

    /// 形状字段是否与另一份 step 一致（同一个捕获图的前提）。
    pub fn shape_matches(&self, other: &Self) -> bool {
        self.batch_size == other.batch_size
            && self.num_tokens == other.num_tokens
            && self.max_blocks == other.max_blocks
            && self.vocab_size == other.vocab_size
            && self.topk == other.topk
    }
}

// ---------------------------------------------------------------- 布局报告
/// 生成 ABI 布局报告。
///
/// 这份文本是 `native/abi/abi_layout.txt` 的唯一来源；C 侧测试解析同一个文件。
/// 用文本而不是二进制，是为了让"布局变了"这件事在 code review 里直接可见。
pub fn layout_report() -> String {
    use core::mem::{align_of, offset_of, size_of};

    macro_rules! dump_struct {
        ($out:expr, $ty:ty { $($field:ident : $fty:ty),* $(,)? }) => {{
            $out.push_str(&format!(
                "struct {} size {} align {}\n",
                stringify!($ty),
                size_of::<$ty>(),
                align_of::<$ty>()
            ));
            $(
                $out.push_str(&format!(
                    "field {} {} {}\n",
                    stringify!($field),
                    offset_of!($ty, $field),
                    size_of::<$fty>()
                ));
            )*
        }};
    }

    let mut out = String::new();
    out.push_str(&format!("anyinfer-abi-layout {}\n", ABI_VERSION));

    dump_struct!(
        out,
        AnyInferStatus {
            code: StatusCode,
            failed_op_id: u32,
            detail: i32,
            reserved0: u32,
            step_index: u64,
        }
    );

    dump_struct!(out, AnyInferStep {
        abi_version: u32,
        flags: u32,
        batch_size: u32,
        num_tokens: u32,
        max_blocks: u32,
        vocab_size: u32,
        topk: u32,
        reserved0: u32,
        step_index: u64,
        weight_version: u64,
        stream: *mut c_void,
        token_ids: *const i32,
        positions: *const i32,
        block_table: *const i32,
        seq_lens: *const i32,
        slot_mapping: *const i32,
        kv_cache: *mut c_void,
        logits: *mut c_void,
        topk_values: *mut c_void,
        topk_indices: *mut i32,
        logsumexp: *mut c_void,
        sampling_mask: *mut c_void,
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn sizes_are_pinned() {
        assert_eq!(size_of::<AnyInferStatus>(), 24, "AnyInferStatus size drift");
        assert_eq!(size_of::<AnyInferStep>(), 144, "AnyInferStep size drift");
        assert_eq!(align_of::<AnyInferStatus>(), 8);
        assert_eq!(align_of::<AnyInferStep>(), 8);
    }

    /// 与 C 侧 `abi_layout_test.cpp` 独立断言的同一个性质：
    /// 偏移首尾相接、无空洞，说明结构体里没有隐式 padding。
    #[test]
    fn no_implicit_padding() {
        use core::mem::offset_of;

        macro_rules! assert_contiguous {
            ($ty:ty, $($field:ident : $fty:ty),* $(,)?) => {{
                let mut cursor = 0usize;
                $(
                    let offset = offset_of!($ty, $field);
                    assert_eq!(
                        offset,
                        cursor,
                        concat!(
                            "gap before ", stringify!($field),
                            " in ", stringify!($ty), ": implicit padding"
                        )
                    );
                    cursor += size_of::<$fty>();
                )*
                assert_eq!(cursor, size_of::<$ty>(), "trailing padding");
            }};
        }

        assert_contiguous!(AnyInferStatus,
            code: StatusCode,
            failed_op_id: u32,
            detail: i32,
            reserved0: u32,
            step_index: u64,
        );

        assert_contiguous!(AnyInferStep,
            abi_version: u32,
            flags: u32,
            batch_size: u32,
            num_tokens: u32,
            max_blocks: u32,
            vocab_size: u32,
            topk: u32,
            reserved0: u32,
            step_index: u64,
            weight_version: u64,
            stream: *mut c_void,
            token_ids: *const i32,
            positions: *const i32,
            block_table: *const i32,
            seq_lens: *const i32,
            slot_mapping: *const i32,
            kv_cache: *mut c_void,
            logits: *mut c_void,
            topk_values: *mut c_void,
            topk_indices: *mut i32,
            logsumexp: *mut c_void,
            sampling_mask: *mut c_void,
        );
    }

    #[test]
    fn status_defaults_are_ok() {
        let status = AnyInferStatus::default();
        assert!(status.is_ok());
        assert_eq!(status.failed_op_id, OP_NONE);
    }

    #[test]
    fn step_zeroed_is_valid_shape() {
        let step = AnyInferStep::zeroed();
        assert_eq!(step.abi_version, ABI_VERSION);
        assert!(step.stream.is_null());
        assert!(step.shape_matches(&AnyInferStep::zeroed()));
    }

    #[test]
    fn shape_mismatch_is_detected() {
        let mut a = AnyInferStep::zeroed();
        let mut b = AnyInferStep::zeroed();
        a.batch_size = 8;
        b.batch_size = 8;
        assert!(a.shape_matches(&b));
        // step_index / weight_version 不属于形状，变化不应触发换图
        b.step_index = 42;
        b.weight_version = 3;
        assert!(a.shape_matches(&b));
        // 形状字段变化必须触发
        b.topk = 512;
        assert!(!a.shape_matches(&b));
    }
}
