#!/usr/bin/env python3
"""C-04：对拍执行器。

把一个"被测输出"（引擎导出的 safetensors）与 FP64 参考轨迹比对，产出与 C++ harness
**同名同语义**的 Gate 结论。这样 RSI 的自动裁决脚本只解析一种词汇表。

判据（逐元素）：|actual - ref| <= atol + rtol * |ref|

与 `native/harness/fp64_reference.h` 的实现刻意保持一致，包括归因顺序：
    参考非法 > 被测件非有限 > 超容差
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors_io import Tensor, load  # noqa: E402

# 与 native/harness/report.h 的 Gate 一一对应
GATE_PASS = "pass"
GATE_LENGTH = "length_mismatch"
GATE_NON_FINITE = "non_finite"
GATE_TOLERANCE = "tolerance_exceeded"
GATE_BAD_REFERENCE = "invalid_reference"

# 归因优先级：数值越大越靠前（越"上游"）
_SEVERITY = {
    GATE_PASS: 0,
    GATE_TOLERANCE: 1,
    GATE_NON_FINITE: 2,
    GATE_BAD_REFERENCE: 3,
    GATE_LENGTH: 4,
}


def tolerance_for(name: str, tolerances: dict) -> tuple[float, float]:
    """按前缀规则取容差。第一条匹配的规则生效，否则用 default。"""
    for rule in tolerances.get("rules", []):
        if name.startswith(rule["prefix"]):
            return float(rule["atol"]), float(rule["rtol"])
    default = tolerances["default"]
    return float(default["atol"]), float(default["rtol"])


def compare_tensor(
    name: str, reference: Tensor, candidate: Tensor, atol: float, rtol: float
) -> dict:
    report: dict = {
        "name": name,
        "atol": atol,
        "rtol": rtol,
        "count": reference.numel,
    }

    if reference.shape != candidate.shape:
        report.update(
            gate=GATE_LENGTH,
            detail=f"shape {tuple(candidate.shape)} != reference {tuple(reference.shape)}",
        )
        return report

    violations = 0
    first_violation = None
    non_finite = 0
    bad_reference = 0
    max_abs = 0.0
    max_abs_index = 0
    max_rel = 0.0
    max_rel_index = 0
    worst = None

    for index, (ref_value, act_value) in enumerate(zip(reference.values, candidate.values)):
        if not math.isfinite(ref_value):
            bad_reference += 1
            continue
        if not math.isfinite(act_value):
            non_finite += 1
            if first_violation is None:
                first_violation = index
                worst = (ref_value, act_value)
            continue

        diff = abs(act_value - ref_value)
        allowed = atol + rtol * abs(ref_value)
        if diff > max_abs:
            max_abs = diff
            max_abs_index = index
        if ref_value != 0.0:
            rel = diff / abs(ref_value)
            if rel > max_rel:
                max_rel = rel
                max_rel_index = index
        if diff > allowed:
            violations += 1
            if first_violation is None:
                first_violation = index
                worst = (ref_value, act_value)

    if bad_reference:
        gate = GATE_BAD_REFERENCE
        detail = f"{bad_reference} non-finite reference element(s)"
    elif non_finite:
        gate = GATE_NON_FINITE
        detail = f"{non_finite} non-finite candidate element(s)"
    elif violations:
        gate = GATE_TOLERANCE
        detail = f"{violations} element(s) exceed atol + rtol*abs(ref)"
    else:
        gate = GATE_PASS
        detail = ""

    report.update(
        gate=gate,
        detail=detail,
        violations=violations,
        first_violation=first_violation,
        max_abs=max_abs,
        max_abs_index=max_abs_index,
        max_rel=max_rel,
        max_rel_index=max_rel_index,
        worst_ref=worst[0] if worst else 0.0,
        worst_actual=worst[1] if worst else 0.0,
    )
    return report


def overall_gate(reports: list[dict]) -> str:
    worst = GATE_PASS
    for report in reports:
        if _SEVERITY[report["gate"]] > _SEVERITY[worst]:
            worst = report["gate"]
    return worst


def run(
    reference_path: str | Path,
    candidate_path: str | Path,
    tolerances: dict,
) -> dict:
    reference, _ = load(reference_path)
    candidate, _ = load(candidate_path)

    reports: list[dict] = []
    for name in sorted(reference):
        if name not in candidate:
            reports.append(
                {
                    "name": name,
                    "gate": GATE_LENGTH,
                    "detail": "missing in candidate",
                    "count": reference[name].numel,
                    "atol": 0.0,
                    "rtol": 0.0,
                }
            )
            continue
        atol, rtol = tolerance_for(name, tolerances)
        reports.append(compare_tensor(name, reference[name], candidate[name], atol, rtol))

    for name in sorted(set(candidate) - set(reference)):
        reports.append(
            {
                "name": name,
                "gate": GATE_LENGTH,
                "detail": "unexpected tensor in candidate",
                "count": candidate[name].numel,
                "atol": 0.0,
                "rtol": 0.0,
            }
        )

    gate = overall_gate(reports)
    return {
        "format": "anyinfer-toy-compare-v1",
        "gate": gate,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "tensors": len(reports),
        "failed": sorted(r["name"] for r in reports if r["gate"] != GATE_PASS),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="compare engine output against FP64 reference")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parents[2] / "tests" / "golden" / "toy_corpus.json"),
    )
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    result = run(args.reference, args.candidate, corpus["tolerances"])

    if args.json:
        path = Path(args.json)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    if not args.quiet:
        summary = {k: result[k] for k in ("gate", "tensors", "failed")}
        print(json.dumps(summary, sort_keys=True))

    return 0 if result["gate"] == GATE_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
