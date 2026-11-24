"""玩具模型的规格与 FP64 参考实现（C-01 / C-02）。

这一段代码的定位是 **oracle**：引擎实现得对不对，以它为准。因此：

1. **零第三方依赖**。参考实现只用标准库，Python 的 float 就是 IEEE-754 binary64，
   所有中间结果都在 fp64 上算，不存在"参考实现自己也不准"的问题。
2. **权重是 dyadic**：每个权重 = k * 2^-7，|k| <= 128。这种值在 fp32 中**精确可表示**，
   所以引擎读到的权重与 oracle 用的权重逐位相同 —— 误差只可能来自计算，不可能来自读权重。
   生成器是纯哈希函数，C++ 侧可以脱离文件复现同一组权重。
3. **架构与真实 decoder 同构**：rmsnorm / qkv / RoPE / causal attention / o_proj / 残差 /
   swiglu FFN / final norm / 词嵌入绑定（tied lm_head）。不是为了跑得快，是为了让
   "这一层对不上"能直接映射到真实实现的某一层。

一个刻意保留的诚实之处：**RoPE 之后的激活是超越数**（cos/sin 无理），
它不可能满足 `exact.h` 的位级判据。所以含 RoPE 的层只能走 L2（FP64 参考 + 容差），
永远不许声称 L1 —— 这与 `native/harness/exact.h` 的规则一致。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from safetensors_io import Tensor

# 权重网格：k * 2^-7，|k| <= 128 -> 值域 [-1, 1)，8 位小数位，fp32 精确
WEIGHT_SHIFT = 7
MATMUL_K_LOW = -128
MATMUL_K_HIGH = 128
NORM_K_LOW = 96  # 归一化权重取 [0.75, 1.25)，避免接近 0 把信号压没
NORM_K_HIGH = 160


@dataclass(frozen=True)
class Spec:
    """玩具模型的形状。改这里就要重跑 export 并更新 manifest 的哈希。"""

    layers: int = 2
    hidden: int = 32
    heads: int = 4
    head_dim: int = 8
    inter: int = 64
    vocab: int = 257  # 刻意取素数：任何"按 2 的幂对齐"的越界/padding bug 都会暴露
    seed: int = 0x5EED_0001
    rope_base: float = 10000.0
    eps: float = 1e-6

    def validate(self) -> None:
        if self.heads * self.head_dim != self.hidden:
            raise ValueError(
                f"heads*head_dim must equal hidden: {self.heads}*{self.head_dim} "
                f"!= {self.hidden}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.vocab < 2:
            raise ValueError("vocab must be at least 2")


# ---------------------------------------------------------------- 确定性权重生成
def mix(seed: int, index: int) -> int:
    """与 native/harness/exact.h 的 mix() 同构，保证跨语言可复现同一组权重。"""
    mask = 0xFFFFFFFF
    x = ((seed + 0x9E3779B1) & mask) ^ ((index + 0x85EBCA6B) & mask)
    x = (x ^ (x >> 16)) & mask
    x = (x * 0x7FEB352D) & mask
    x = (x ^ (x >> 15)) & mask
    x = (x * 0x846CA68B) & mask
    x = (x ^ (x >> 16)) & mask
    return x


def _salt(name: str) -> int:
    """由张量名派生盐值，使不同张量之间不相关，同时保持"名字定值"的确定性。"""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def dyadic_matrix(
    name: str, rows: int, cols: int, *, seed: int, k_low: int, k_high: int
) -> Tensor:
    span = k_high - k_low
    salt = _salt(name)
    scale = math.ldexp(1.0, -WEIGHT_SHIFT)
    values = [
        ((mix(seed ^ salt, row * cols + col) % span) + k_low) * scale
        for row in range(rows)
        for col in range(cols)
    ]
    return Tensor(dtype="F32", shape=(rows, cols), values=tuple(values))


def build_weights(spec: Spec) -> dict[str, Tensor]:
    """生成全部权重（F32，逐位可被引擎复现）。"""
    spec.validate()
    seed = spec.seed
    weights: dict[str, Tensor] = {}

    weights["embed.weight"] = dyadic_matrix(
        "embed.weight", spec.vocab, spec.hidden, seed=seed,
        k_low=MATMUL_K_LOW, k_high=MATMUL_K_HIGH,
    )
    weights["final_norm.weight"] = dyadic_matrix(
        "final_norm.weight", 1, spec.hidden, seed=seed,
        k_low=NORM_K_LOW, k_high=NORM_K_HIGH,
    )

    h = spec.hidden
    for layer in range(spec.layers):
        prefix = f"layer{layer}"
        for name, rows, cols, low, high in (
            (f"{prefix}.norm1.weight", 1, h, NORM_K_LOW, NORM_K_HIGH),
            (f"{prefix}.wq.weight", h, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.wk.weight", h, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.wv.weight", h, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.wo.weight", h, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.norm2.weight", 1, h, NORM_K_LOW, NORM_K_HIGH),
            (f"{prefix}.wgate.weight", spec.inter, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.wup.weight", spec.inter, h, MATMUL_K_LOW, MATMUL_K_HIGH),
            (f"{prefix}.wdown.weight", h, spec.inter, MATMUL_K_LOW, MATMUL_K_HIGH),
        ):
            weights[name] = dyadic_matrix(
                name, rows, cols, seed=seed, k_low=low, k_high=high
            )
    return weights


# ---------------------------------------------------------------- FP64 参考前向
def _rows(tensor: Tensor) -> list[list[float]]:
    cols = tensor.shape[1] if len(tensor.shape) == 2 else 1
    values = tensor.values
    return [list(values[r * cols : (r + 1) * cols]) for r in range(tensor.shape[0])]


def _matmul(x: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
    """y[t][o] = sum_i x[t][i] * weight[o][i]（weight 按 [out, in] 存）。"""
    out: list[list[float]] = []
    for row in x:
        acc_row = []
        for w_row in weight:
            total = 0.0
            for a, b in zip(row, w_row):
                total += a * b
            acc_row.append(total)
        out.append(acc_row)
    return out


def _rmsnorm(x: list[list[float]], weight: list[float], eps: float) -> list[list[float]]:
    out = []
    for row in x:
        mean_square = sum(v * v for v in row) / len(row)
        denom = math.sqrt(mean_square + eps)
        out.append([row[i] / denom * weight[i] for i in range(len(row))])
    return out


def _rope(
    x: list[list[float]], spec: Spec
) -> list[list[float]]:
    """对 [seq, hidden] 按头切分后旋转。位置即行号。"""
    half = spec.head_dim // 2
    inv_freq = [
        spec.rope_base ** (-2.0 * d / spec.head_dim) for d in range(half)
    ]
    out = []
    for pos, row in enumerate(x):
        rotated = list(row)
        for head in range(spec.heads):
            base = head * spec.head_dim
            for d in range(half):
                angle = pos * inv_freq[d]
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                a = row[base + 2 * d]
                b = row[base + 2 * d + 1]
                rotated[base + 2 * d] = a * cos_a - b * sin_a
                rotated[base + 2 * d + 1] = a * sin_a + b * cos_a
        out.append(rotated)
    return out


def _causal_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]], spec: Spec
) -> list[list[float]]:
    seq = len(q)
    scale = 1.0 / math.sqrt(spec.head_dim)
    out = [[0.0] * spec.hidden for _ in range(seq)]
    for head in range(spec.heads):
        base = head * spec.head_dim
        for t in range(seq):
            q_t = q[t][base : base + spec.head_dim]
            scores = []
            for s in range(t + 1):  # 因果：只看 <= t
                k_s = k[s][base : base + spec.head_dim]
                scores.append(sum(a * b for a, b in zip(q_t, k_s)) * scale)
            peak = max(scores)
            exps = [math.exp(sc - peak) for sc in scores]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            for s in range(t + 1):
                v_s = v[s][base : base + spec.head_dim]
                weight = probs[s]
                for d in range(spec.head_dim):
                    out[t][base + d] += weight * v_s[d]
    return out


def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def reference_forward(
    spec: Spec, weights: dict[str, Tensor], tokens: list[int]
) -> tuple[dict[str, Tensor], list[float]]:
    """跑 FP64 参考前向，返回逐层激活与最后位置的 logits。

    返回的 captures 是"对拍用"的：引擎在同样位置导出同名张量即可逐层定位误差。
    """
    spec.validate()
    for token in tokens:
        if not 0 <= token < spec.vocab:
            raise ValueError(f"token {token} outside vocab [0, {spec.vocab})")

    embed = _rows(weights["embed.weight"])
    final_norm_w = weights["final_norm.weight"].values

    captures: dict[str, Tensor] = {}
    seq = len(tokens)

    x = [list(embed[token]) for token in tokens]
    captures["embed.out"] = _capture(x)

    for layer in range(spec.layers):
        prefix = f"layer{layer}"
        norm1_w = weights[f"{prefix}.norm1.weight"].values
        norm2_w = weights[f"{prefix}.norm2.weight"].values

        h = _rmsnorm(x, norm1_w, spec.eps)
        captures[f"{prefix}.norm1"] = _capture(h)

        q = _matmul(h, _rows(weights[f"{prefix}.wq.weight"]))
        k = _matmul(h, _rows(weights[f"{prefix}.wk.weight"]))
        v = _matmul(h, _rows(weights[f"{prefix}.wv.weight"]))
        # RoPE 之前导出，方便把"旋转实现"与"注意力实现"的错误分开定位
        captures[f"{prefix}.q"] = _capture(q)
        captures[f"{prefix}.k"] = _capture(k)
        captures[f"{prefix}.v"] = _capture(v)

        q = _rope(q, spec)
        k = _rope(k, spec)
        # RoPE 之后单独导出：此处的值已是超越数，只能走 L2
        captures[f"{prefix}.q_rope"] = _capture(q)
        captures[f"{prefix}.k_rope"] = _capture(k)

        attn = _causal_attention(q, k, v, spec)
        captures[f"{prefix}.attn"] = _capture(attn)

        projected = _matmul(attn, _rows(weights[f"{prefix}.wo.weight"]))
        captures[f"{prefix}.attn_out"] = _capture(projected)

        x = [[a + b for a, b in zip(xr, pr)] for xr, pr in zip(x, projected)]
        captures[f"{prefix}.resid1"] = _capture(x)

        h = _rmsnorm(x, norm2_w, spec.eps)
        captures[f"{prefix}.norm2"] = _capture(h)

        gate = _matmul(h, _rows(weights[f"{prefix}.wgate.weight"]))
        up = _matmul(h, _rows(weights[f"{prefix}.wup.weight"]))
        activated = [
            [_silu(g) * u for g, u in zip(grow, urow)] for grow, urow in zip(gate, up)
        ]
        captures[f"{prefix}.ffn_act"] = _capture(activated)

        down = _matmul(activated, _rows(weights[f"{prefix}.wdown.weight"]))
        x = [[a + b for a, b in zip(xr, dr)] for xr, dr in zip(x, down)]
        captures[f"{prefix}.resid2"] = _capture(x)

    h = _rmsnorm(x, final_norm_w, spec.eps)
    captures["final_norm"] = _capture(h)

    # 词嵌入绑定：lm_head 就是 embed.weight
    last = h[seq - 1]
    logits = [sum(a * b for a, b in zip(last, row)) for row in embed]
    captures["logits"] = Tensor(
        dtype="F64", shape=(spec.vocab,), values=tuple(logits)
    )

    logprobs = _log_softmax(logits)
    captures["logprobs"] = Tensor(
        dtype="F64", shape=(spec.vocab,), values=tuple(logprobs)
    )

    return captures, logits


def _capture(x: list[list[float]]) -> Tensor:
    rows = len(x)
    cols = len(x[0]) if rows else 0
    flat: list[float] = []
    for row in x:
        flat.extend(row)
    return Tensor(dtype="F64", shape=(rows, cols), values=tuple(flat))


def _log_softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    denom = sum(exps)
    log_denom = math.log(denom)
    return [v - peak - log_denom for v in values]


def spec_from_dict(payload: dict) -> Spec:
    """从容差/语料 JSON 里读 spec，未知字段直接报错（避免静默忽略拼写错误）。"""
    allowed = {f for f in Spec.__dataclass_fields__}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown spec field(s): {sorted(unknown)}")
    return Spec(**payload)
