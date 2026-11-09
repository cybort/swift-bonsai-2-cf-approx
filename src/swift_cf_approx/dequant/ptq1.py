"""PTQ1_0 (ggml type 143, PrismML dialect) dequant / quant kernels.

Ported from rlx-gguf `ptq1_dequant` / Prism `dequantize_row_ptq1_0`.
Block = 28 bytes / 128 trits: qs(24) + qh(2) + f16 scale d.

Fast path uses the CUDA element-map (verified equal to the stage walk).
"""

from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

QK_PTQ1_0 = 128
PTQ1_0_QS_BYTES = 24
PTQ1_0_QH_BYTES = 2
PTQ1_0_BLOCK_BYTES = 28
PTQ1_0_STAGES = (32, 16, 8)
POW3 = (1, 3, 9, 27, 81)

GGML_TYPE_PTQ1_0 = 143


def _build_element_map() -> tuple[np.ndarray, np.ndarray]:
    """For each of 128 outputs: (byte_index_in_block, digit_n)."""
    byte_idx = np.empty(QK_PTQ1_0, dtype=np.int32)
    digit = np.empty(QK_PTQ1_0, dtype=np.int32)
    for e in range(QK_PTQ1_0):
        if e < 80:
            byte_idx[e] = e & 15
            digit[e] = e >> 4
        elif e < 120:
            t = e - 80
            byte_idx[e] = 16 + (t & 7)
            digit[e] = t >> 3
        else:
            t = e - 120
            byte_idx[e] = 24 + (t & 1)
            digit[e] = t >> 1
    return byte_idx, digit


_BYTE_IDX, _DIGIT = _build_element_map()
_POW3_U16 = np.array(POW3, dtype=np.uint16)
_DIGIT_MUL = _POW3_U16[_DIGIT]


def ptq1_0_bytes(n: int) -> int | None:
    if n % QK_PTQ1_0 != 0:
        return None
    return (n // QK_PTQ1_0) * PTQ1_0_BLOCK_BYTES


def _trit(byte: int, n: int) -> int:
    q = (byte * POW3[n]) & 0xFF
    return ((q * 3) >> 8) - 1


def dequant_ptq1_0_block(block: bytes | memoryview, out: np.ndarray) -> None:
    """Dequantize one 28-byte block into out[:128] float32."""
    if len(block) < PTQ1_0_BLOCK_BYTES:
        raise ValueError(f"PTQ1_0 block needs {PTQ1_0_BLOCK_BYTES} bytes")
    # Use vector path for a single block too.
    arr = dequant_ptq1_0(block, QK_PTQ1_0)
    out[:QK_PTQ1_0] = arr


def dequant_ptq1_0(
    data: bytes | memoryview | np.ndarray,
    n: int,
    *,
    dtype: np.dtype | str = np.float32,
    block_chunk: int = 65536,
) -> np.ndarray:
    """Dequant PTQ1_0 to float32/float16, processing `block_chunk` blocks at a time."""
    if n % QK_PTQ1_0 != 0:
        raise ValueError(f"PTQ1_0: n={n} not divisible by {QK_PTQ1_0}")
    nb = n // QK_PTQ1_0
    expected = nb * PTQ1_0_BLOCK_BYTES
    if isinstance(data, np.ndarray):
        raw = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    else:
        raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size != expected:
        raise ValueError(f"PTQ1_0: expected {expected} bytes, got {raw.size}")

    out_dtype = np.dtype(dtype)
    out = np.empty(n, dtype=out_dtype)
    for start in range(0, nb, block_chunk):
        end = min(nb, start + block_chunk)
        blocks = raw[start * PTQ1_0_BLOCK_BYTES : end * PTQ1_0_BLOCK_BYTES].reshape(
            end - start, PTQ1_0_BLOCK_BYTES
        )
        scales = blocks[:, 26:28].copy().view(dtype="<f2").reshape(end - start).astype(np.float32)
        gathered = blocks[:, _BYTE_IDX]
        q = (gathered.astype(np.uint16) * _DIGIT_MUL) & 0xFF
        trits = ((q.astype(np.uint16) * 3) >> 8).astype(np.int8) - 1
        chunk = (trits.astype(np.float32) * scales[:, None]).reshape(-1)
        out[start * QK_PTQ1_0 : end * QK_PTQ1_0] = chunk.astype(out_dtype, copy=False)
    return out


def _code(w: float, id_: float) -> int:
    return int(round(max(-1.0, min(1.0, w * id_)))) + 1


def _rescale(q: int) -> int:
    return (q * 256 + 242) // 243


def quantize_ptq1_0(src: np.ndarray | Iterable[float]) -> bytes:
    """Lossless for already-ternary-at-128 input (published checkpoint case)."""
    arr = np.asarray(src, dtype=np.float32).reshape(-1)
    if arr.size % QK_PTQ1_0 != 0:
        raise ValueError(f"PTQ1_0: len={arr.size} not divisible by {QK_PTQ1_0}")
    out = bytearray()
    for bi in range(arr.size // QK_PTQ1_0):
        blk = arr[bi * QK_PTQ1_0 : (bi + 1) * QK_PTQ1_0]
        d = float(np.max(np.abs(blk))) if blk.size else 0.0
        id_ = (1.0 / d) if d > 0.0 else 0.0
        qs = bytearray(PTQ1_0_QS_BYTES)
        qh = bytearray(PTQ1_0_QH_BYTES)
        base = 0
        j = 0
        for c in PTQ1_0_STAGES:
            while j + c <= PTQ1_0_QS_BYTES:
                for m in range(c):
                    q = 0
                    for n_ in range(5):
                        q = q * 3 + _code(float(blk[base + m + n_ * c]), id_)
                    qs[j + m] = _rescale(q)
                base += 5 * c
                j += c
        for h in range(PTQ1_0_QH_BYTES):
            q = 0
            for m in range(4):
                q = q * 3 + _code(float(blk[base + h + m * PTQ1_0_QH_BYTES]), id_)
            q *= 3
            qh[h] = _rescale(q)
        out.extend(qs)
        out.extend(qh)
        out.extend(struct.pack("<e", d))
    return bytes(out)
