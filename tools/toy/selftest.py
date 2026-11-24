#!/usr/bin/env python3
"""对拍器的自证：确认它真的会红。

单测覆盖了单元级语义，但"整套 CLI 流程 + 实际产物"这条链也要自证一次：
    1. 参考 vs 参考            -> 必须 pass
    2. 参考 vs 扰动 0.5%        -> 必须 tolerance_exceeded，且只报被扰动的那个张量
    3. 参考 vs 注入 NaN         -> 必须 non_finite，且只报被注入的那个张量

只允许"永远绿"的验收门等于没有验收门。这个脚本就是防止那一天的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare as compare_mod  # noqa: E402
from safetensors_io import Tensor, load, save  # noqa: E402


def _write_variant(source: Path, target: Path, tensor: str, index: int,
                   factor: float | None) -> None:
    tensors, metadata = load(source)
    entry = tensors[tensor]
    values = list(entry.values)
    values[index] = float("nan") if factor is None else values[index] * factor
    tensors[tensor] = Tensor(entry.dtype, entry.shape, tuple(values))
    save(target, tensors, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="prove the comparator actually fails on bad input")
    parser.add_argument("--dir", required=True, help="directory produced by export.py")
    parser.add_argument("--prompt", default="simple")
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parents[2] / "tests" / "golden" / "toy_corpus.json"),
    )
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    tolerances = corpus["tolerances"]

    directory = Path(args.dir)
    reference = directory / f"reference_{args.prompt}.safetensors"
    if not reference.exists():
        print(f"FAIL missing {reference}; run export.py first", file=sys.stderr)
        return 2

    failures = 0

    def check(name: str, expected_gate: str, expected_failed: list[str],
              candidate: Path) -> None:
        nonlocal failures
        result = compare_mod.run(reference, candidate, tolerances)
        ok = result["gate"] == expected_gate and result["failed"] == expected_failed
        status = "ok  " if ok else "FAIL"
        print(f"  {status} {name}: gate={result['gate']} failed={result['failed']}")
        if not ok:
            print(
                f"        expected gate={expected_gate} failed={expected_failed}",
                file=sys.stderr,
            )
            failures += 1

    check("reference vs itself", compare_mod.GATE_PASS, [], reference)

    scaled = directory / "selftest_scaled.safetensors"
    _write_variant(reference, scaled, "logits", 0, 1.005)
    check("logits +0.5%", compare_mod.GATE_TOLERANCE, ["logits"], scaled)

    injected = directory / "selftest_nan.safetensors"
    _write_variant(reference, injected, "layer0.attn", 3, None)
    check("layer0.attn NaN", compare_mod.GATE_NON_FINITE, ["layer0.attn"], injected)

    if failures:
        print(f"\n{failures} self-check(s) failed", file=sys.stderr)
        return 1
    print("comparator self-check: all expectations met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
