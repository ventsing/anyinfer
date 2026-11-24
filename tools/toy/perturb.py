#!/usr/bin/env python3
"""把一个 safetensors 里的单个元素按比例放大，用于制造"必须被抓到"的负向样本。

存在的理由：没有负向样本的对拍器无法自证有效。RSI 的自动裁决必须定期被"喂错"，
确认它真的会红 —— 否则某天它永远绿你也不知道为什么。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors_io import Tensor, load, save  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="scale one element of a safetensors file")
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--factor", type=float, default=1.005, help="multiplier (1.005 = +0.5%%)")
    parser.add_argument("--nan", action="store_true", help="write NaN instead of scaling")
    args = parser.parse_args()

    tensors, metadata = load(args.source)
    if args.tensor not in tensors:
        print(f"FAIL tensor {args.tensor!r} not found; have {sorted(tensors)}", file=sys.stderr)
        return 2

    tensor = tensors[args.tensor]
    if not 0 <= args.index < tensor.numel:
        print(
            f"FAIL index {args.index} outside [0, {tensor.numel}) for {args.tensor!r}",
            file=sys.stderr,
        )
        return 2

    values = list(tensor.values)
    if args.nan:
        values[args.index] = float("nan")
        description = "NaN"
    else:
        values[args.index] *= args.factor
        description = f"x{args.factor}"
    tensors[args.tensor] = Tensor(tensor.dtype, tensor.shape, tuple(values))

    save(args.target, tensors, metadata=metadata)
    print(
        f"perturbed {args.tensor}[{args.index}]: "
        f"{tensor.values[args.index]:.9g} -> {values[args.index]:.9g} ({description})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
