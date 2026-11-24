#!/usr/bin/env python3
"""B-04b：算子验收的**单命令入口**。

只做两件事：把三份产物报告里的结论**提取**出来，然后交给 C++ 裁决器 `anyinfer-accept`。
它刻意**不**实现任何聚合逻辑 —— 裁决权威只有 `native/harness/accept.h` 一处。
两份实现迟早会分叉，然后就会有人花一天去查"为什么本地过、流水线不过"。

三段门：
  --correctness-report   L1/L2 正确性报告（含 "gate" 字段）
  --bench-report         微基准报告（含 "verdict"/"speedup"/"sign_test_p"）
  --l4-report            端到端报告（含 "gate" 字段，通常来自 tools/toy/compare.py）

没给报告 = 那道门没跑 = not_run = **阻断晋升**。缺证据不等于通过。

退出码：0 可晋升 / 1 失败 / 2 证据不全 / 64 用法错误
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# report.h 的 Gate 词汇（与 native/harness/report.h 的 gate_name 一一对应）
REPORT_GATES = {
    "pass",
    "length_mismatch",
    "non_finite",
    "bit_mismatch",
    "tolerance_exceeded",
    "invalid_reference",
}

BENCH_VERDICTS = {"b_faster", "a_faster", "indistinguishable", "not_run"}


class UsageError(RuntimeError):
    pass


def find_accept_binary(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise UsageError(f"--accept-bin {path} does not exist")
        return path
    for candidate in (
        ROOT / "build" / "ci" / "cmake" / "anyinfer-accept",
        ROOT / "build" / "cmake" / "anyinfer-accept",
        ROOT / "build" / "anyinfer-accept",
    ):
        if candidate.is_file():
            return candidate
    found = shutil.which("anyinfer-accept")
    if found:
        return Path(found)
    raise UsageError(
        "anyinfer-accept not found. Build it first:\n"
        "  cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake --target anyinfer-accept\n"
        "or pass --accept-bin PATH"
    )


def load_report(path: str | None, what: str) -> dict | None:
    if path is None:
        return None
    file = Path(path)
    if not file.is_file():
        raise UsageError(f"{what} report {file} does not exist")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UsageError(f"{what} report {file} is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise UsageError(f"{what} report {file} is not a JSON object")
    return payload


def gate_from_report(payload: dict | None, what: str, path: str | None) -> list[str]:
    """把一个含 gate 字段的报告转成裁决器的参数。"""
    state_option = "--correctness" if what == "correctness" else "--l4"
    gate_option = "--correctness-gate" if what == "correctness" else "--l4-gate"
    reason_option = "--correctness-reason" if what == "correctness" else "--l4-reason"

    if payload is None:
        # 没给报告就是没跑过。用 not_run 而不是 pass —— 缺证据不等于通过
        return [state_option, "not_run", reason_option,
                f"no {what} report supplied (gate never ran)"]

    gate = payload.get("gate")
    if gate is None:
        raise UsageError(f"{what} report {path} has no 'gate' field")
    if gate not in REPORT_GATES:
        raise UsageError(f"{what} report {path} has unknown gate {gate!r}")
    return [gate_option, gate]


def bench_args(payload: dict | None, path: str | None) -> list[str]:
    if payload is None:
        return ["--microbench", "not_run"]
    verdict = payload.get("verdict")
    if verdict is None:
        raise UsageError(f"bench report {path} has no 'verdict' field")
    if verdict not in BENCH_VERDICTS:
        raise UsageError(f"bench report {path} has unknown verdict {verdict!r}")
    args = ["--microbench", verdict]
    if verdict != "not_run":
        speedup = payload.get("speedup")
        p_value = payload.get("sign_test_p")
        if speedup is None or p_value is None:
            raise UsageError(
                f"bench report {path} must carry 'speedup' and 'sign_test_p' "
                "(without them the regression guard cannot be applied)"
            )
        args += ["--bench-speedup", repr(float(speedup)),
                 "--bench-sign-test-p", repr(float(p_value))]
    return args


class Parser(argparse.ArgumentParser):
    """用法错误一律退 64。

    argparse 默认退 2，而裁决器用 2 表示"证据不全"。两者撞车会让流水线把
    "参数打错"读成"门没跑完"，进而去补证据而不是去看命令。
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(64)


def main() -> int:
    parser = Parser(
        description="single-command operator acceptance (delegates the verdict to anyinfer-accept)"
    )
    parser.add_argument("--op", required=True, help="operator name, e.g. hc_post")
    parser.add_argument("--correctness-report", help="JSON with a 'gate' field")
    parser.add_argument("--bench-report", help="JSON with verdict/speedup/sign_test_p")
    parser.add_argument("--l4-report", help="JSON with a 'gate' field")
    parser.add_argument("--max-regression", type=float, default=0.10)
    parser.add_argument("--accept-bin", help="path to anyinfer-accept")
    parser.add_argument("--out", help="write the acceptance JSON here")
    parser.add_argument("--quiet", action="store_true", help="suppress the JSON on stdout")
    args = parser.parse_args()

    try:
        binary = find_accept_binary(args.accept_bin)
        correctness = load_report(args.correctness_report, "correctness")
        bench = load_report(args.bench_report, "bench")
        l4 = load_report(args.l4_report, "l4")

        command = [str(binary), "--op", args.op]
        command += gate_from_report(correctness, "correctness", args.correctness_report)
        command += bench_args(bench, args.bench_report)
        command += gate_from_report(l4, "l4", args.l4_report)
        command += ["--max-regression", repr(args.max_regression)]
        if args.out:
            command += ["--json", args.out]
        if args.quiet:
            command += ["--quiet"]
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 64

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
