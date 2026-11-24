#!/usr/bin/env python3
"""不可自修改面的哈希锁（F-05）。

# 它防的是什么

RSI 允许 agent 改引擎自身。但如果 agent 也能改**测量**，那"变好了"就不再是一个
有意义的说法 —— 它可以把判据改松、把参考 oracle 改到跟自己的输出一致、
或者把 golden 语料换掉。**一个能修改裁判的选手没有输的可能。**

所以这些文件被哈希锁住，改动必须显式解封并在 git 历史里留下理由。

# 它防不了什么（先说实话）

锁文件本身在仓库里。一个足够强的 agent 完全可以解封、改文件、再封回去，
全程绿灯。所以这个工具**不是沙箱，是留痕机制**：

- 一切改动都会在 `docs/IMMUTABLE_LOG.md` 追加一行（带时间、路径、旧/新哈希、理由）。
- 追加是只增不改的，所以任何解封都出现在 git diff 里，人能看到。

真正的强制只能来自仓库外（CI 持有私钥签名、或文件系统挂载为只读）。
这一层留在一个仓库里能做到的极限：**让越界不可能不留痕迹**。

# 用法

    python3 tools/immutable.py check                  # 漂移则 exit 1
    python3 tools/immutable.py seal --reason "…"      # 重新封印，必须给理由
    python3 tools/immutable.py list                   # 当前受保护的文件
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_STAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 退出码。argparse 自己用 2 表示用法错误，所以这里把几种失败分开，
# 免得"理由没给"和"命令行打错了"在 CI 里看起来一样。
EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_NO_LOCK = 4
EXIT_NO_REASON = 3
LOCK = ROOT / "tests" / "golden" / "immutable.lock.json"
LOG = ROOT / "docs" / "IMMUTABLE_LOG.md"

# 受保护面：按**目录整棵**锁，而不是逐个文件。
# 逐个文件锁不住"新增一个影子文件"——agent 可以加一个 `bench_v2.h` 然后让构建去用它。
PROTECTED_DIRS = [
    # 判据与测量协议。改这里等于改尺子。
    "native/harness",
    # 参考 oracle 与对拍器。改这里等于改答案。
    "tools/toy",
    # golden 语料与哈希锁。
    "tests/golden",
    # 裁决权威与构建逻辑。
    "tools/accept",
    "ci",
]

PROTECTED_FILES = [
    # 布局产物：Rust 与 C++ 两侧各自独立校验的**同一份**基准。
    "native/abi/abi_layout.txt",
    # 构建入口。
    "CMakeLists.txt",
    # ADR 只追加，不可改写。
    "docs/DECISIONS.md",
    # 执行者自己。能被改的检查器等于没有检查器 —— 改它会出现在漂移报告里。
    "tools/immutable.py",
    # 上机判决的解读层。这里存的是判据：kernel_broken 不是架构结论、
    # content_folded 禁止改代码去绕。改这里等于改"事故该往哪修"的答案。
    "tools/device_a02.sh",
    "tools/device_probe.sh",
    # 验证上面那些东西的测试本身。能被改的测试等于没测试。
    "tests/abi_layout_test.cpp",
    "tests/graph_probe_logic_test.cpp",
    "tests/harness_accept_test.cpp",
    "tests/harness_bench_test.cpp",
    "tests/harness_compare_test.cpp",
    "tests/harness_exact_test.cpp",
    "tests/toy_reference_test.py",
    "tests/device_a02_test.py",
]

# 显式排除：`__pycache__` 是构建副产物，内容随 Python 版本变。
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_DIRS = {"__pycache__"}

# 本工具自己的产物不能进自己的集合 —— 否则每次 seal 都会把自己写进去，
# 下一次 check 立刻报"新增了一个文件"。锁文件与日志因此永远不在被锁集合里。
SELF_ARTIFACTS = {"tests/golden/immutable.lock.json", "docs/IMMUTABLE_LOG.md"}


def collect() -> list[str]:
    """受保护文件集合，相对路径，排序。"""
    paths: list[str] = []
    for entry in PROTECTED_DIRS:
        base = ROOT / entry
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            parts = set(path.parts)
            if parts & EXCLUDE_DIRS:
                continue
            if path.suffix in EXCLUDE_SUFFIXES:
                continue
            relative = str(path.relative_to(ROOT))
            if relative in SELF_ARTIFACTS:
                continue
            paths.append(relative)
    for entry in PROTECTED_FILES:
        if entry in SELF_ARTIFACTS:
            continue
        if (ROOT / entry).is_file():
            paths.append(entry)
    return sorted(set(paths))


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    return {relative: digest(relative) for relative in collect()}


def load_lock() -> dict[str, str] | None:
    if not LOCK.is_file():
        return None
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    return payload.get("files", {})


def check() -> int:
    locked = load_lock()
    if locked is None:
        print(f"缺少锁文件 {LOCK.relative_to(ROOT)}；先跑 seal", file=sys.stderr)
        return EXIT_NO_LOCK

    current = snapshot()
    added = sorted(set(current) - set(locked))
    removed = sorted(set(locked) - set(current))
    changed = sorted(
        relative
        for relative in set(current) & set(locked)
        if current[relative] != locked[relative]
    )

    if not (added or removed or changed):
        print(f"不可自修改面：{len(current)} 个文件全部未变")
        return EXIT_OK

    print("不可自修改面发生漂移：", file=sys.stderr)
    for relative in changed:
        print(f"  修改  {relative}", file=sys.stderr)
        print(f"        旧 {locked[relative][:16]}", file=sys.stderr)
        print(f"        新 {current[relative][:16]}", file=sys.stderr)
    for relative in added:
        print(f"  新增  {relative}", file=sys.stderr)
    for relative in removed:
        print(f"  删除  {relative}", file=sys.stderr)
    print(
        "\n若这是有意的改动，用 `python3 tools/immutable.py seal --reason \"…\"` 重新封印；"
        "理由会写进 docs/IMMUTABLE_LOG.md 并留在 git 历史里。",
        file=sys.stderr,
    )
    return EXIT_DRIFT


def seal(reason: str) -> int:
    if not reason.strip():
        print("必须给 --reason：解封的理由是这套机制唯一的价值来源", file=sys.stderr)
        return EXIT_NO_REASON

    previous = load_lock() or {}
    current = snapshot()

    lines = ["", f"## {LOG_STAMP}", "", f"理由：{reason.strip()}", ""]
    changed = sorted(
        relative
        for relative in set(current) & set(previous)
        if current[relative] != previous[relative]
    )
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    if not (changed or added or removed):
        lines.append("（无变化，仅重新确认）")
    for relative in changed:
        lines.append(
            f"- 修改 `{relative}`：{previous[relative][:16]} → {current[relative][:16]}"
        )
    for relative in added:
        lines.append(f"- 新增 `{relative}`：{current[relative][:16]}")
    for relative in removed:
        lines.append(f"- 删除 `{relative}`：原 {previous[relative][:16]}")

    if not LOG.is_file():
        LOG.write_text(
            "# 不可自修改面变更日志\n\n"
            "只增不改。每次 `tools/immutable.py seal` 都会在这里留一条记录，\n"
            "所以任何对判据 / 参考实现 / 语料的改动都能在 git diff 里被看见。\n",
            encoding="utf-8",
        )
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(
        json.dumps(
            {
                "_comment": "由 tools/immutable.py seal 生成；不要手改",
                "directories": PROTECTED_DIRS,
                "single_files": PROTECTED_FILES,
                "files": current,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已封印 {len(current)} 个文件；记录写入 {LOG.relative_to(ROOT)}")
    return EXIT_OK


def list_surface() -> int:
    for relative in collect():
        print(relative)
    return EXIT_OK


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="不可自修改面的哈希锁")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="漂移则 exit 1")
    seal_parser = sub.add_parser("seal", help="重新封印（必须给理由）")
    seal_parser.add_argument("--reason", required=True, help="为什么必须改")
    sub.add_parser("list", help="列出受保护文件")

    args = parser.parse_args(argv)
    if args.command == "check":
        return check()
    if args.command == "seal":
        return seal(args.reason)
    return list_surface()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
