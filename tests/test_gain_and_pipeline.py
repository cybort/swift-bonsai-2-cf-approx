from __future__ import annotations

import json
from pathlib import Path

from swift_cf_approx.distill import generate_distill_dataset, load_prompts
from swift_cf_approx.eval import run_eval_compare
from swift_cf_approx.train import GainProxyConfig, compute_sample_weight, run_train


def test_gain_proxy_weights():
    cfg = GainProxyConfig()
    assert compute_sample_weight(10, cfg) > 1.0
    assert compute_sample_weight(800, cfg) < 1.0
    cfg.enabled = False
    assert compute_sample_weight(10, cfg) == 1.0


def test_distill_synthetic(tmp_path: Path):
    prompts = [{"id": "1", "prompt": "Say hello in one word."}]
    out = tmp_path / "d.jsonl"
    examples = generate_distill_dataset(prompts, out, base_url=None)
    assert len(examples) == 1
    assert out.exists()
    row = json.loads(out.read_text().splitlines()[0])
    assert "messages" in row


def test_train_dry_run(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "train.jsonl"
    prompts = load_prompts(root / "data" / "prompts" / "distill_seed.jsonl")[:3]
    generate_distill_dataset(prompts, data, base_url=None)
    out = tmp_path / "artifacts"
    summary = run_train(
        config_path=root / "configs" / "train_qlora.yaml",
        gain_config_path=root / "configs" / "gain_proxy.yaml",
        data_path=data,
        output_dir=out,
        backend="cpu",
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert (out / "lora_adapter" / "adapter_config.json").exists()


def test_eval_offline(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "reports"
    result = run_eval_compare(
        config_path=root / "configs" / "eval.yaml",
        output_dir=out,
        adapter_path=tmp_path / "lora",
        use_live_endpoints=False,
    )
    assert "deltas" in result
    assert (out / "scores.json").exists()
    assert (out / "comparison.md").exists()
    assert result["deltas"]["thinking_token_reduction_pct"] > 0
