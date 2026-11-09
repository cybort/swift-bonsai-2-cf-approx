from __future__ import annotations

from pathlib import Path

from swift_cf_approx.dequant import unpack_swift_ptq1_to_hf


def test_unpack_fixture(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    gguf = root / "tests" / "fixtures" / "mini_ptq1.gguf"
    assert gguf.exists(), "run fixture generator or commit tests/fixtures/mini_ptq1.gguf"
    out = tmp_path / "out"
    report = unpack_swift_ptq1_to_hf(gguf, out)
    assert report["n_tensors"] == 1
    assert not report["skipped"]
    assert (out / "config.json").exists()
    assert any(out.glob("model-*.safetensors"))
