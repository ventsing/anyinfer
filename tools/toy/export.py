#!/usr/bin/env python3
"""C-01 + C-02 的产出器：生成玩具模型权重与 FP64 参考轨迹。

产物（默认写进 --out 目录，不进 git）：
    weights.safetensors            全部权重，F32，dyadic（引擎可逐位复现）
    reference_<prompt>.safetensors 逐层激活 + logits + logprobs，F64
    manifest.json                  spec、语料、每个文件的 sha256、每个张量的 sha256

为什么不把产物提交进 git：它们是**由 seed 完全确定的**，提交字节只会让仓库变重
且容易与代码漂移。提交的是 manifest 里的哈希 —— 只要有人改了参考实现或权重生成器，
哈希就会变，CI 立刻发现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Spec, build_weights, reference_forward, spec_from_dict  # noqa: E402
from safetensors_io import Tensor, save, sha256_file, sha256_tensor  # noqa: E402


def build(spec: Spec, corpus: dict) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    weights = build_weights(spec)
    trajectories: dict[str, dict[str, Tensor]] = {}
    for prompt in corpus["prompts"]:
        name = prompt["name"]
        tokens = list(prompt["tokens"])
        captures, _ = reference_forward(spec, weights, tokens)
        trajectories[name] = captures
    return weights, trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description="export toy model + FP64 reference")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parents[2] / "tests" / "golden" / "toy_corpus.json"),
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--inter", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--seed", type=int, default=Spec.seed)
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    spec = spec_from_dict(
        {
            "layers": args.layers,
            "hidden": args.hidden,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "inter": args.inter,
            "vocab": args.vocab,
            "seed": args.seed,
        }
    )
    # heads*head_dim 必须等于 hidden；不满足就让 Spec.validate() 报错，不做静默纠正
    spec.validate()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights, trajectories = build(spec, corpus)

    weights_path = out / "weights.safetensors"
    save(
        weights_path,
        weights,
        metadata={"anyinfer": "toy-weights-v1", "seed": str(spec.seed), "dtype": "F32"},
    )

    files: dict[str, dict[str, object]] = {}
    tensor_hashes: dict[str, dict[str, str]] = {}

    files[weights_path.name] = {
        "sha256": sha256_file(weights_path),
        "tensors": len(weights),
        "dtype": "F32",
    }
    for name, captures in trajectories.items():
        path = out / f"reference_{name}.safetensors"
        save(
            path,
            captures,
            metadata={"anyinfer": "toy-reference-v1", "prompt": name, "dtype": "F64"},
        )
        files[path.name] = {
            "sha256": sha256_file(path),
            "tensors": len(captures),
            "dtype": "F64",
        }
        tensor_hashes[name] = {key: sha256_tensor(value) for key, value in captures.items()}

    manifest = {
        "format": "anyinfer-toy-manifest-v1",
        "spec": {
            "layers": spec.layers,
            "hidden": spec.hidden,
            "heads": spec.heads,
            "head_dim": spec.head_dim,
            "inter": spec.inter,
            "vocab": spec.vocab,
            "seed": spec.seed,
            "rope_base": spec.rope_base,
            "eps": spec.eps,
        },
        "corpus": corpus["prompts"],
        "tolerances": corpus["tolerances"],
        "files": files,
        "tensor_sha256": tensor_hashes,
    }
    manifest_path = out / "manifest.json"
    tmp = manifest_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)

    print(f"spec      : {spec.layers}L hidden={spec.hidden} heads={spec.heads} "
          f"inter={spec.inter} vocab={spec.vocab}")
    print(f"weights   : {len(weights)} tensors -> {weights_path.name}")
    print(f"reference : {len(trajectories)} prompts -> "
          f"{', '.join(sorted(t.name for t in out.glob('reference_*.safetensors')))}")
    print(f"manifest  : {manifest_path}")
    print(f"weights sha256 = {files[weights_path.name]['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
