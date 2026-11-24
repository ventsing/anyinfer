//! 把 ABI 布局报告打到 stdout。
//!
//! 用法（在仓库根目录）：
//!     cargo run -q --bin anyinfer-abi-dump > native/abi/abi_layout.txt
//!
//! 然后在 C 侧由 `tests/abi_layout_test.cpp` 独立解析同一份文件做校验。

fn main() {
    print!("{}", anyinfer_abi::layout_report());
}
