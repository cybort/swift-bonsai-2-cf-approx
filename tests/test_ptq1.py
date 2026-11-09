"""PTQ1_0 kernel tests aligned with rlx-gguf / Prism element map."""

from __future__ import annotations

import numpy as np

from swift_cf_approx.dequant.ptq1 import (
    PTQ1_0_BLOCK_BYTES,
    QK_PTQ1_0,
    dequant_ptq1_0,
    dequant_ptq1_0_block,
    ptq1_0_bytes,
    quantize_ptq1_0,
)


def _ref_trit(qs: bytes, qh: bytes, e: int) -> int:
    if e < 80:
        b, n = qs[e & 15], e >> 4
    elif e < 120:
        t = e - 80
        b, n = qs[16 + (t & 7)], t >> 3
    else:
        t = e - 120
        b, n = qh[t & 1], t >> 1
    v = b
    for i in range(4):
        if i < n:
            v = (v * 3) & 0xFF
    return ((v * 3) >> 8) - 1


def test_ptq1_bytes():
    assert ptq1_0_bytes(QK_PTQ1_0 - 1) is None
    assert ptq1_0_bytes(QK_PTQ1_0) == PTQ1_0_BLOCK_BYTES


def test_traversal_matches_cuda_element_map():
    seed = 12345

    def nxt() -> int:
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
        return (seed >> 16) & 0xFF

    for _ in range(200):
        qs = bytes(nxt() for _ in range(24))
        qh = bytes(nxt() for _ in range(2))
        import struct

        block = qs + qh + struct.pack("<e", 1.0)
        out = np.zeros(QK_PTQ1_0, dtype=np.float32)
        dequant_ptq1_0_block(block, out)
        for e in range(QK_PTQ1_0):
            assert int(out[e]) == _ref_trit(qs, qh, e)


def test_roundtrip_ternary():
    n = 3 * QK_PTQ1_0
    scales = [0.125, 0.5, 2.0]
    src = np.array([((i % 3) - 1) * scales[i // QK_PTQ1_0] for i in range(n)], dtype=np.float32)
    packed = quantize_ptq1_0(src)
    assert len(packed) == ptq1_0_bytes(n)
    back = dequant_ptq1_0(packed, n)
    np.testing.assert_allclose(back, src, rtol=0, atol=0)


def test_rejects_bad_sizes():
    try:
        dequant_ptq1_0(b"\x00" * 10, QK_PTQ1_0)
        assert False
    except ValueError:
        pass
