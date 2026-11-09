"""Minimal GGUF reader focused on tensor metadata + raw byte slices."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

# Value type tags in GGUF metadata
_GGUF_STRING = 8
_GGUF_ARRAY = 9


@dataclass
class GgufTensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int  # relative to data section start

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GgufFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: list[GgufTensorInfo]
    data_offset: int
    alignment: int

    def tensor_map(self) -> dict[str, GgufTensorInfo]:
        return {t.name: t for t in self.tensors}


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == 0:  # UINT8
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:  # INT32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:  # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:  # BOOL
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _GGUF_STRING:
        return _read_string(f)
    if vtype == _GGUF_ARRAY:
        (atype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, atype) for _ in range(n)]
    if vtype == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF value type {vtype}")


def open_gguf(path: str | Path) -> GgufFile:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, vtype)
        tensors: list[GgufTensorInfo] = []
        for _ in range(tensor_count):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append(GgufTensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))
        alignment = int(metadata.get("general.alignment", 32))
        pos = f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        data_offset = pos + pad
    return GgufFile(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tensors,
        data_offset=data_offset,
        alignment=alignment,
    )


def read_tensor_bytes(gguf: GgufFile, tensor: GgufTensorInfo, nbytes: int) -> bytes:
    with gguf.path.open("rb") as f:
        f.seek(gguf.data_offset + tensor.offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Short read for {tensor.name}: got {len(data)}, want {nbytes}")
    return data
