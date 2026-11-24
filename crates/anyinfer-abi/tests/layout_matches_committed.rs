//! 布局产物与 Rust 定义的一致性检查。
//!
//! 这是整条 ABI 验证链的一半：C 侧（`tests/abi_layout_test.cpp`）解析同一个文件
//! 并对着自己的 `offsetof` 校验。两侧独立校验同一份产物，
//! 任何一侧漂移都会被抓到 —— 而不是等到运行时读到别的字段的值。

use std::path::PathBuf;

fn artifact_path() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/anyinfer-abi
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("native")
        .join("abi")
        .join("abi_layout.txt")
}

#[test]
fn layout_matches_committed_artifact() {
    let path = artifact_path();
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "cannot read {}: {err}\n\
             generate it with: cargo run -q --bin anyinfer-abi-dump > native/abi/abi_layout.txt",
            path.display()
        )
    });

    let generated = anyinfer_abi::layout_report();

    if generated != committed {
        // 逐行指出差异，而不是丢一个巨大的 diff
        let committed_lines: Vec<&str> = committed.lines().collect();
        let generated_lines: Vec<&str> = generated.lines().collect();
        let mut report = String::from("ABI layout drift detected:\n");
        let max = committed_lines.len().max(generated_lines.len());
        for index in 0..max {
            let want = committed_lines.get(index).copied().unwrap_or("<missing>");
            let got = generated_lines.get(index).copied().unwrap_or("<missing>");
            if want != got {
                report.push_str(&format!("  line {}: committed {want:?}\n", index + 1));
                report.push_str(&format!("           rust      {got:?}\n"));
            }
        }
        report.push_str(
            "\n如果这是有意的 ABI 变更：\n\
             1. 同时修改 native/abi/anyinfer.h 与其 sizeof 断言\n\
             2. 重新生成：cargo run -q --bin anyinfer-abi-dump > native/abi/abi_layout.txt\n\
             3. 若形状不兼容，把 ABI_VERSION 加一\n",
        );
        panic!("{report}");
    }
}
