#!/usr/bin/env python3
"""工具链可见性检查（F-02）。

# 为什么不是 `rust-toolchain.toml`

原本 F-02 写的是"加 `rust-toolchain.toml` 锁定版本"。实测发现两个问题：

1. rustup 把 `channel = "1.100.0"` 当作**另一个槽位**，即使本机 stable 恰好就是
   1.100.0，也要重新下载一整套工具链（约 200 MB）。本机网络实测 ~6 KB/s，
   拉完要几小时；而一旦这个文件存在，**每一次 cargo 调用都会先同步**，
   在拉完之前整个仓库不可构建。
2. 它给出的是**虚假的确信**：`rust-toolchain.toml` 只保证"用了哪个编译器"，
   不保证"结果可复算"。真正需要的是**可归因**——一份测量结果必须能说明
   它是哪个编译器产的。这两件事不一样，锁版本只做到了前者的弱化版。

所以这里换一个更弱但更诚实的目标：**让版本不匹配可见**，并把"必须匹配"
留给测量路径（`--strict`）。这个仓库里每份测量 JSON 都该带编译器和版本戳，
那样 n 个月后回看一个数字，能知道它是谁算的。

# 用法

    python3 tools/check_toolchain.py            # 只报告，永远 exit 0
    python3 tools/check_toolchain.py --strict   # 不匹配 exit 1（测量前跑）
    python3 tools/check_toolchain.py --json     # 机读，供测量报告吸收
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

# 产出本仓库现有测量结果时使用的版本。
#
# 为什么是"实测值"而不是"某个理想值"：钉一个我们从没跑过的版本没有意义 ——
# 那只是把一个未验证的假设写在文件里。等真正需要在两台机器间比位级结果时，
# 再把它改成那次测量的版本，并同时改这一行和 EXPECTED 的下限。
RECORDED_RUSTC = "1.100.0"

# Cargo.toml 里 `rust-version` 声明的最低可编译版本（edition 2024 的要求）。
MINIMUM_RUSTC = (1, 85)

TARGET = "x86_64-unknown-linux-gnu"


def rustc_version() -> str | None:
    binary = shutil.which("rustc")
    if binary is None:
        return None
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"rustc (\d+\.\d+\.\d+)", out.stdout)
    return match.group(1) if match else None


def parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:2])


def collect() -> dict[str, object]:
    version = rustc_version()
    notes: list[str] = []
    if version is None:
        notes.append("rustc 不在 PATH 上；静态检查（C++/Python）不受影响")
        ok = False
    else:
        ok = parse(version) >= MINIMUM_RUSTC
        if not ok:
            notes.append(
                f"rustc {version} 低于 rust-version 要求 "
                f"{'.'.join(map(str, MINIMUM_RUSTC))}"
            )
        if version != RECORDED_RUSTC:
            notes.append(
                f"rustc {version} 与产出现有测量结果的 {RECORDED_RUSTC} 不同；"
                "跨版本比较位级结果前需要重新建立基线"
            )
    return {
        "rustc": version,
        "recorded_with": RECORDED_RUSTC,
        "minimum": ".".join(map(str, MINIMUM_RUSTC)),
        "target": TARGET,
        "meets_minimum": ok,
        "matches_recorded": version == RECORDED_RUSTC,
        "notes": notes,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="工具链可见性检查")
    parser.add_argument("--strict", action="store_true", help="不匹配则 exit 1")
    parser.add_argument("--json", action="store_true", help="机读输出")
    args = parser.parse_args(argv)

    report = collect()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"rustc        {report['rustc'] or '（缺）'}")
        print(f"recorded     {report['recorded_with']}")
        print(f"minimum      {report['minimum']}")
        print(f"target       {report['target']}")
        if report["matches_recorded"]:
            print("结论         与产出现有测量结果的版本一致")
        for note in report["notes"]:
            print(f"注           {note}")

    if args.strict and not (report["meets_minimum"] and report["matches_recorded"]):
        print(
            "\n--strict：版本与测量基线不一致，拒绝在该环境下产出可比较的测量结果。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
