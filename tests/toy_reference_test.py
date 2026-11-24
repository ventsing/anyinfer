"""C 轨道的验证：参考实现、容器格式、对拍器。

原则与 C++ 侧一致 —— **每个"通过"都必须配一个"必须失败"**：
只有证明对拍器能抓住扰动，参考轨迹的绿灯才有意义。
"""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "toy"))

import compare as compare_mod  # noqa: E402
from model import Spec, build_weights, reference_forward  # noqa: E402
from safetensors_io import Tensor, load, save, sha256_tensor  # noqa: E402

SPEC = Spec()
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TOLERANCES = {
    "default": {"atol": 1e-6, "rtol": 1e-5},
    "rules": [{"prefix": "logits", "atol": 1e-5, "rtol": 1e-4}],
}


def fp32_roundtrip(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class WeightTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        first = build_weights(SPEC)
        second = build_weights(SPEC)
        self.assertEqual(sorted(first), sorted(second))
        for name in first:
            self.assertEqual(sha256_tensor(first[name]), sha256_tensor(second[name]), name)

    def test_seed_changes_weights(self) -> None:
        other = Spec(seed=SPEC.seed ^ 1)
        first = build_weights(SPEC)
        second = build_weights(other)
        differing = [n for n in first if sha256_tensor(first[n]) != sha256_tensor(second[n])]
        self.assertTrue(differing, "different seeds must produce different weights")

    def test_dyadic_and_fp32_exact(self) -> None:
        """权重必须是 k*2^-7 且能被 fp32 精确表示 —— 否则引擎与 oracle 读到的权重就不同。"""
        weights = build_weights(SPEC)
        for name, tensor in weights.items():
            self.assertEqual(tensor.dtype, "F32", name)
            for value in tensor.values:
                scaled = value * 128.0
                self.assertEqual(scaled, round(scaled), f"{name}: {value} is not k*2^-7")
                self.assertEqual(
                    fp32_roundtrip(value), value, f"{name}: {value} is not fp32-exact"
                )

    def test_shapes(self) -> None:
        weights = build_weights(SPEC)
        self.assertEqual(weights["embed.weight"].shape, (SPEC.vocab, SPEC.hidden))
        self.assertEqual(weights["final_norm.weight"].shape, (1, SPEC.hidden))
        for layer in range(SPEC.layers):
            prefix = f"layer{layer}"
            self.assertEqual(weights[f"{prefix}.wq.weight"].shape, (SPEC.hidden, SPEC.hidden))
            self.assertEqual(
                weights[f"{prefix}.wgate.weight"].shape, (SPEC.inter, SPEC.hidden)
            )
            self.assertEqual(
                weights[f"{prefix}.wdown.weight"].shape, (SPEC.hidden, SPEC.inter)
            )


class ContainerTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        tensors = {
            # F32 用可精确表示的值（0.001953125 = 2^-9），否则往返本就会有舍入
            "f32": Tensor("F32", (2, 3), (1.0, -2.5, 3.25, 0.0, 0.001953125, -1.0)),
            "f64": Tensor("F64", (3,), (math.pi, math.e, 0.0)),
            "i32": Tensor("I32", (2, 2), (0, 1, -1, 2)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.safetensors"
            save(path, tensors, metadata={"k": "v"})
            loaded, metadata = load(path)
        self.assertEqual(metadata, {"k": "v"})
        self.assertEqual(sorted(loaded), sorted(tensors))
        for name, tensor in tensors.items():
            self.assertEqual(loaded[name].dtype, tensor.dtype)
            self.assertEqual(loaded[name].shape, tensor.shape)
            self.assertEqual(loaded[name].values, tensor.values)
            self.assertEqual(sha256_tensor(loaded[name]), sha256_tensor(tensor))

    def test_f32_is_lossy_and_container_does_not_pretend(self) -> None:
        """容器不会凭空造精度：非 fp32 可表示的值往返后就是 fp32 的值。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lossy.safetensors"
            save(path, {"t": Tensor("F32", (2,), (0.001, 1.0))})
            loaded, _ = load(path)
        values = loaded["t"].values
        self.assertNotEqual(values[0], 0.001)
        self.assertEqual(values[0], fp32_roundtrip(0.001))
        self.assertEqual(values[1], 1.0)
        # 推论：需要精确比较的场景必须用 F64，不能指望 F32 存住原始十进制值
        self.assertEqual(Tensor("F64", (1,), (0.001,)).values[0], 0.001)

    def test_reproducible_bytes(self) -> None:
        tensors = {"a": Tensor("F32", (4,), (1.0, 2.0, 3.0, 4.0))}
        with tempfile.TemporaryDirectory() as tmp:
            one = Path(tmp) / "one.safetensors"
            two = Path(tmp) / "two.safetensors"
            save(one, tensors)
            save(two, tensors)
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save(Path(tmp) / "bad.safetensors", {"a": Tensor("F32", (2, 2), (1.0, 2.0))})

    def test_rejects_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.safetensors"
            save(path, {"a": Tensor("F32", (2,), (1.0, 2.0))})
            raw = path.read_bytes()
            path.write_bytes(raw[:-2])
            with self.assertRaises(ValueError):
                load(path)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = build_weights(SPEC)
        cls.captures, cls.logits = reference_forward(SPEC, cls.weights, TOKENS)

    def test_deterministic(self) -> None:
        again, _ = reference_forward(SPEC, self.weights, TOKENS)
        self.assertEqual(sorted(again), sorted(self.captures))
        for name in self.captures:
            self.assertEqual(
                sha256_tensor(again[name]), sha256_tensor(self.captures[name]), name
            )

    def test_captures_present(self) -> None:
        expected = {"embed.out", "final_norm", "logits", "logprobs"}
        for layer in range(SPEC.layers):
            for suffix in ("norm1", "q", "k", "v", "q_rope", "k_rope", "attn",
                           "attn_out", "resid1", "norm2", "ffn_act", "resid2"):
                expected.add(f"layer{layer}.{suffix}")
        self.assertTrue(expected.issubset(self.captures),
                        f"missing captures: {sorted(expected - set(self.captures))}")

    def test_all_finite(self) -> None:
        for name, tensor in self.captures.items():
            for value in tensor.values:
                self.assertTrue(math.isfinite(value), f"{name} contains non-finite value")

    def test_logprobs_normalized(self) -> None:
        logprobs = self.captures["logprobs"].values
        total = math.fsum(math.exp(v) for v in logprobs)
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_logits_match_logprobs(self) -> None:
        logits = self.captures["logits"].values
        logprobs = self.captures["logprobs"].values
        peak = max(logits)
        log_denom = math.log(math.fsum(math.exp(v - peak) for v in logits))
        for index in (0, 1, SPEC.vocab // 2, SPEC.vocab - 1):
            self.assertAlmostEqual(
                logprobs[index], logits[index] - peak - log_denom, places=12
            )

    def test_causality(self) -> None:
        """改动最后一个 token 之后，前面所有位置的行必须**逐位不变**。

        这是参考实现里最值得测的一条：因果掩码写错时，数值看起来依然"正常"，
        只有这种位置级断言能抓住它。
        """
        mutated = list(TOKENS)
        mutated[-1] = (TOKENS[-1] + 7) % SPEC.vocab
        other, _ = reference_forward(SPEC, self.weights, mutated)

        seq = len(TOKENS)
        checked = 0
        for name, tensor in self.captures.items():
            if len(tensor.shape) != 2 or tensor.shape[0] != seq:
                continue  # logits/logprobs 依赖最后位置，本就应当变化
            cols = tensor.shape[1]
            for row in range(seq - 1):  # 前 seq-1 行不依赖被改的 token
                begin = row * cols
                end = begin + cols
                self.assertEqual(
                    tensor.values[begin:end],
                    other[name].values[begin:end],
                    f"{name} row {row} changed when only the last token changed",
                )
                checked += 1
        self.assertGreater(checked, 0, "causality test checked nothing")

    def test_last_position_does_change(self) -> None:
        """反向确认：上一条测试不是因为"什么都没变"而侥幸通过。"""
        mutated = list(TOKENS)
        mutated[-1] = (TOKENS[-1] + 7) % SPEC.vocab
        other, other_logits = reference_forward(SPEC, self.weights, mutated)
        self.assertNotEqual(
            self.captures["logits"].values, other["logits"].values,
            "changing the last token must change the logits",
        )

    def test_rope_preserves_norm(self) -> None:
        """RoPE 是旋转：每个头的向量长度必须不变（fp64 下几乎精确）。"""
        for layer in range(SPEC.layers):
            before = self.captures[f"layer{layer}.q"]
            after = self.captures[f"layer{layer}.q_rope"]
            rows = before.shape[0]
            for row in range(rows):
                for head in range(SPEC.heads):
                    begin = head * SPEC.head_dim
                    end = begin + SPEC.head_dim
                    norm_before = math.sqrt(
                        math.fsum(v * v for v in before.values[row * SPEC.hidden + begin:
                                                              row * SPEC.hidden + end])
                    )
                    norm_after = math.sqrt(
                        math.fsum(v * v for v in after.values[row * SPEC.hidden + begin:
                                                             row * SPEC.hidden + end])
                    )
                    self.assertAlmostEqual(
                        norm_before, norm_after, places=12,
                        msg=f"layer{layer} row{row} head{head} norm changed under RoPE",
                    )

    def test_rope_is_not_dyadic(self) -> None:
        """RoPE 之后的值是超越数 —— 这是"只能走 L2、不许声称 L1"的具体证据。"""
        rope = self.captures["layer0.q_rope"]
        non_dyadic = sum(
            1 for value in rope.values if (value * 2**20) != round(value * 2**20)
        )
        self.assertGreater(
            non_dyadic, 0, "expected RoPE output to contain non-dyadic values"
        )


class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        weights = build_weights(SPEC)
        cls.captures, _ = reference_forward(SPEC, weights, TOKENS)
        cls.reference = cls.dir / "reference.safetensors"
        save(cls.reference, cls.captures)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _write_candidate(self, name: str, tensors: dict) -> Path:
        path = self.dir / f"{name}.safetensors"
        save(path, tensors)
        return path

    def test_identical_passes(self) -> None:
        result = compare_mod.run(self.reference, self.reference, TOLERANCES)
        self.assertEqual(result["gate"], compare_mod.GATE_PASS)
        self.assertEqual(result["failed"], [])

    def test_relative_perturbation_caught(self) -> None:
        tensors = dict(self.captures)
        logits = list(tensors["logits"].values)
        logits[0] *= 1.0 + 1e-3  # 相对偏差 1e-3，远大于 rtol=1e-4
        tensors["logits"] = Tensor("F64", tensors["logits"].shape, tuple(logits))
        path = self._write_candidate("relu_perturbed", tensors)
        result = compare_mod.run(self.reference, path, TOLERANCES)
        self.assertEqual(result["gate"], compare_mod.GATE_TOLERANCE)
        self.assertEqual(result["failed"], ["logits"])
        report = next(r for r in result["reports"] if r["name"] == "logits")
        self.assertEqual(report["violations"], 1)
        self.assertEqual(report["first_violation"], 0)

    def test_one_ulp_caught_when_bitwise(self) -> None:
        """零容差模式下：1 个 ulp 的差异必须被抓（这是确定性自检的形态）。"""
        tensors = dict(self.captures)
        values = list(tensors["final_norm"].values)
        values[0] = math.nextafter(values[0], math.inf)
        tensors["final_norm"] = Tensor("F64", tensors["final_norm"].shape, tuple(values))
        path = self._write_candidate("ulp", tensors)
        strict = {"default": {"atol": 0.0, "rtol": 0.0}, "rules": []}
        result = compare_mod.run(self.reference, path, strict)
        self.assertEqual(result["gate"], compare_mod.GATE_TOLERANCE)
        self.assertEqual(result["failed"], ["final_norm"])
        # 同一份数据在默认容差下应当通过 —— 证明容差确实在起作用
        self.assertEqual(
            compare_mod.run(self.reference, path, TOLERANCES)["gate"],
            compare_mod.GATE_PASS,
        )

    def test_non_finite_caught(self) -> None:
        tensors = dict(self.captures)
        values = list(tensors["layer0.attn"].values)
        values[3] = float("nan")
        tensors["layer0.attn"] = Tensor("F64", tensors["layer0.attn"].shape, tuple(values))
        path = self._write_candidate("nan", tensors)
        result = compare_mod.run(self.reference, path, TOLERANCES)
        self.assertEqual(result["gate"], compare_mod.GATE_NON_FINITE)
        self.assertEqual(result["failed"], ["layer0.attn"])

    def test_shape_mismatch_caught(self) -> None:
        # 保持 numel 不变、只改形状，容器能写出去，比对器必须判 shape 不符
        tensors = dict(self.captures)
        tensor = tensors["layer0.norm1"]
        tensors["layer0.norm1"] = Tensor("F64", (tensor.numel,), tensor.values)
        path = self._write_candidate("shape", tensors)
        result = compare_mod.run(self.reference, path, TOLERANCES)
        self.assertEqual(result["gate"], compare_mod.GATE_LENGTH)
        self.assertEqual(result["failed"], ["layer0.norm1"])

    def test_missing_and_extra_tensors_caught(self) -> None:
        missing = dict(self.captures)
        del missing["layer1.resid2"]
        result = compare_mod.run(
            self.reference, self._write_candidate("missing", missing), TOLERANCES
        )
        self.assertEqual(result["gate"], compare_mod.GATE_LENGTH)
        self.assertIn("layer1.resid2", result["failed"])

        extra = dict(self.captures)
        extra["bogus"] = Tensor("F64", (1,), (0.0,))
        result = compare_mod.run(
            self.reference, self._write_candidate("extra", extra), TOLERANCES
        )
        self.assertEqual(result["gate"], compare_mod.GATE_LENGTH)
        self.assertIn("bogus", result["failed"])

    def test_cross_dtype_f32_candidate(self) -> None:
        """引擎产出的是 F32；对拍器必须能跨 dtype 比较，而不是要求候选也是 F64。"""
        tensors = {}
        for name, tensor in self.captures.items():
            values = tuple(fp32_roundtrip(v) for v in tensor.values)
            tensors[name] = Tensor("F32", tensor.shape, values)
        path = self._write_candidate("f32", tensors)
        result = compare_mod.run(self.reference, path, TOLERANCES)
        self.assertEqual(result["gate"], compare_mod.GATE_PASS,
                         f"failed: {result['failed']}")


class CorpusTests(unittest.TestCase):
    def test_corpus_is_valid(self) -> None:
        corpus = json.loads((ROOT / "tests" / "golden" / "toy_corpus.json").read_text())
        self.assertIn("prompts", corpus)
        self.assertIn("tolerances", corpus)
        names = [p["name"] for p in corpus["prompts"]]
        self.assertEqual(len(names), len(set(names)), "prompt names must be unique")
        for prompt in corpus["prompts"]:
            tokens = prompt["tokens"]
            self.assertTrue(tokens, f"{prompt['name']} has no tokens")
            for token in tokens:
                self.assertGreaterEqual(token, 0)
                self.assertLess(token, SPEC.vocab)


if __name__ == "__main__":
    unittest.main(verbosity=2)
