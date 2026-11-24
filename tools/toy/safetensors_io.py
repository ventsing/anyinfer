"""极简 safetensors 读写，只用标准库。

为什么自己写而不用 `safetensors` 包：
  1. 本机没有 pip 环境，且不想给"参考 oracle"引入任何第三方依赖 ——
     oracle 应当是最可信的那一个东西，它的依赖越少越可信。
  2. 引擎侧最终要用 C++/Rust 读同一份文件，所以**容器格式必须是引擎要实现的常态**，
     而不是测试专用的临时格式。
  3. 格式本身极小：8 字节头长 + JSON 头 + 连续数据区。

格式（与 HF safetensors 一致）：
    [0:8)      little-endian u64  头部长度 N
    [8:8+N)    UTF-8 JSON 头部
    [8+N:end)  数据缓冲区；每个 tensor 的 data_offsets 相对该缓冲区起点，
               且各 tensor 必须连续、无空洞、无重叠。

JSON 头部：
    {"tensor_name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]}, ...}
    可选 "__metadata__": {str: str}
"""

from __future__ import annotations

import array
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# dtype -> (array typecode, 每元素字节数)
_DTYPES: dict[str, tuple[str, int]] = {
    "F32": ("f", 4),
    "F64": ("d", 8),
    "I32": ("i", 4),
    "I64": ("q", 8),
}

# 本机与目标机都是小端；大端时统一 byteswap，避免"文件格式跟着机器变"。
_NATIVE_LITTLE = sys.byteorder == "little"


class SafetensorsError(ValueError):
    """格式或内容不合法。刻意用异常而非静默兜底：静默兜底会让哈希校验失去意义。"""


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    values: tuple  # 扁平、行主序

    @property
    def numel(self) -> int:
        return len(self.values)

    def element_size(self) -> int:
        try:
            return _DTYPES[self.dtype][1]
        except KeyError:
            raise SafetensorsError(f"unsupported dtype {self.dtype!r}") from None

    def nbytes(self) -> int:
        return self.numel * self.element_size()


def _element_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if dim < 0:
            raise SafetensorsError(f"negative dimension in shape {shape}")
        count *= dim
    return count


def save(path: str | Path, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    """写入 safetensors。先写 .part 再原子改名，避免中断留下半个文件。"""
    path = Path(path)
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)

    offset = 0
    for name in sorted(tensors):  # 排序保证同一输入产生同一字节流（可复现）
        tensor = tensors[name]
        if len(tensor.shape) == 0:
            raise SafetensorsError("scalar tensors are not supported; use shape [1]")
        if _element_count(tensor.shape) != tensor.numel:
            raise SafetensorsError(
                f"{name}: shape {tensor.shape} implies {_element_count(tensor.shape)} "
                f"elements but {tensor.numel} were supplied"
            )
        end = offset + tensor.nbytes()
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end

    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    # 头部按 8 字节对齐（用空格填充，JSON 允许尾随空白）
    padding = (-len(blob)) % 8
    blob += b" " * padding

    payload = bytearray()
    for name in sorted(tensors):
        tensor = tensors[name]
        typecode, _ = _DTYPES[tensor.dtype]
        buf = array.array(typecode, tensor.values)
        if not _NATIVE_LITTLE:
            buf.byteswap()
        payload += buf.tobytes()

    if len(payload) != offset:
        raise SafetensorsError("internal: payload size disagrees with header offsets")

    target = path.with_suffix(path.suffix + ".part")
    with target.open("wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)
        handle.write(bytes(payload))
    target.replace(path)


def load(path: str | Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise SafetensorsError(f"{path}: truncated file")

    header_len = int.from_bytes(raw[0:8], "little")
    if 8 + header_len > len(raw):
        raise SafetensorsError(f"{path}: header length {header_len} exceeds file size")
    try:
        header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{path}: bad JSON header: {exc}") from None

    metadata: dict[str, str] = {}
    if "__metadata__" in header:
        metadata = dict(header.pop("__metadata__"))

    buffer = memoryview(raw)[8 + header_len :]
    tensors: dict[str, Tensor] = {}
    seen_offsets: list[tuple[int, int, str]] = []

    for name, entry in header.items():
        try:
            dtype = entry["dtype"]
            shape = tuple(int(d) for d in entry["shape"])
            begin, end = (int(v) for v in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(f"{path}: malformed header entry for {name!r}: {exc}") from None

        if dtype not in _DTYPES:
            raise SafetensorsError(f"{path}: unsupported dtype {dtype!r} for {name!r}")
        typecode, element_size = _DTYPES[dtype]
        expected = _element_count(shape) * element_size
        if end - begin != expected:
            raise SafetensorsError(
                f"{path}: {name!r} claims {end - begin} bytes but shape {shape}/{dtype} "
                f"needs {expected}"
            )
        if begin < 0 or end > len(buffer):
            raise SafetensorsError(f"{path}: {name!r} offsets [{begin},{end}) outside buffer")
        seen_offsets.append((begin, end, name))

        buf = array.array(typecode)
        buf.frombytes(bytes(buffer[begin:end]))
        if not _NATIVE_LITTLE:
            buf.byteswap()
        tensors[name] = Tensor(dtype=dtype, shape=shape, values=tuple(buf))

    # 连续无空洞：参考实现如此，也让我们能对整份文件做字节级比对
    seen_offsets.sort()
    cursor = 0
    for begin, end, name in seen_offsets:
        if begin != cursor:
            raise SafetensorsError(
                f"{path}: {name!r} starts at {begin} but buffer continues at {cursor} "
                "(holes or overlaps are not allowed)"
            )
        cursor = end
    if cursor != len(buffer):
        raise SafetensorsError(f"{path}: trailing bytes after last tensor")

    return tensors, metadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: Tensor) -> str:
    """单个 tensor 的内容哈希。

    只哈希**数值语义**（dtype + shape + IEEE 位模式），不含名字与偏移，
    这样即使容器布局变了，"这份数据有没有变"仍然可判定。
    """
    digest = hashlib.sha256()
    digest.update(tensor.dtype.encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    typecode, _ = _DTYPES[tensor.dtype]
    buf = array.array(typecode, tensor.values)
    if not _NATIVE_LITTLE:
        buf.byteswap()
    digest.update(buf.tobytes())
    return digest.hexdigest()
