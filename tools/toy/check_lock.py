#!/usr/bin/env python3
"""校验玩具模型产物的哈希锁。

为什么只锁权重、不锁参考轨迹：
  - **权重**由纯整数哈希运算生成（k * 2^-7），跨平台、跨 Python 版本逐位一致，
    因此可以对它做硬哈希锁。锁住了它，就等于锁住了"引擎读到的权重"。
  - **参考轨迹**含 exp / cos / sin，依赖 libm 实现。不同 glibc / 不同 CPU 上
    最后一位 ulp 可能不同，硬锁会造成假失败。所以它只在 manifest 里记录，不做强制。

这条区分本身就是设计的一部分：**能逐位复现的才加锁，不能的只记录。**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="verify toy weights hash against the lock")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--lock", default=str(ROOT / "tests" / "golden" / "toy_manifest.lock.json")
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))

    weights_sha = manifest["files"]["weights.safetensors"]["sha256"]
    expected_sha = lock["weights_sha256"]
    spec = manifest["spec"]

    problems: list[str] = []
    if weights_sha != expected_sha:
        problems.append(f"weights sha256 {weights_sha} != locked {expected_sha}")
    for key, want in lock["spec"].items():
        got = spec.get(key)
        if got != want:
            problems.append(f"spec.{key} = {got!r}, locked {want!r}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        print(
            "\n如果这是有意的模型变更，请重跑 export 并更新 "
            "tests/golden/toy_manifest.lock.json。\n"
            "如果不是有意的，说明权重生成器或参考实现被改坏了。"
        )
        return 1

    print(f"lock ok: weights sha256={weights_sha[:16]}… spec={lock['spec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
