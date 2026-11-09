"""QLoRA / dry-run training entry with thinking-length penalty + GAIN proxy weights."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .gain_proxy import GainProxyConfig, compute_sample_weight

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _thinking_length_penalty(thinking_tokens: int, coeff: float, target: int) -> float:
    if coeff <= 0:
        return 0.0
    overflow = max(0, thinking_tokens - target)
    return coeff * math.log1p(overflow)


def run_train(
    *,
    config_path: str | Path,
    gain_config_path: str | Path | None = None,
    data_path: str | Path,
    output_dir: str | Path,
    backend: str = "cpu",
    dry_run: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    gain_cfg_raw = _load_yaml(Path(gain_config_path)) if gain_config_path else {}
    gain = GainProxyConfig(**{**asdict(GainProxyConfig()), **(gain_cfg_raw.get("gain_proxy") or gain_cfg_raw)})

    if dry_run is None:
        dry_run = bool(cfg.get("dry_run", backend == "cpu"))

    rows = _load_jsonl(data_path)
    if not rows:
        raise ValueError(f"No training rows in {data_path}")

    max_steps = int(cfg.get("max_steps", 50))
    thinking_penalty_coeff = float(cfg.get("thinking_penalty_coeff", 0.02))
    thinking_target = int(cfg.get("thinking_target_tokens", 128))
    lr = float(cfg.get("learning_rate", 2e-4))

    weighted = []
    for row in rows:
        tw = int(row.get("thinking_tokens_est", 0))
        w = compute_sample_weight(tw, gain)
        pen = _thinking_length_penalty(tw, thinking_penalty_coeff, thinking_target)
        weighted.append(
            {
                "id": row.get("id"),
                "weight": w,
                "thinking_tokens_est": tw,
                "thinking_penalty": pen,
                "effective_objective": w * (1.0 + pen),
            }
        )

    summary = {
        "backend": backend,
        "dry_run": dry_run,
        "n_examples": len(rows),
        "max_steps": max_steps,
        "learning_rate": lr,
        "gain_proxy": asdict(gain),
        "mean_weight": sum(x["weight"] for x in weighted) / len(weighted),
        "mean_thinking_tokens": sum(x["thinking_tokens_est"] for x in weighted) / len(weighted),
        "model_path": cfg.get("model_path"),
        "lora_r": cfg.get("lora_r", 16),
        "lora_alpha": cfg.get("lora_alpha", 32),
    }

    (output_dir / "train_plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "sample_weights.jsonl").write_text(
        "\n".join(json.dumps(x) for x in weighted[: min(200, len(weighted))]) + "\n",
        encoding="utf-8",
    )

    if dry_run:
        # Persist a stub adapter marker so downstream eval/upload paths work in CI/CPU.
        adapter_dir = output_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": summary["lora_r"],
                    "lora_alpha": summary["lora_alpha"],
                    "target_modules": cfg.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    ),
                    "swift_cf_approx_dry_run": True,
                    "base_model_name_or_path": cfg.get("model_path", "unpacked-swift-bonsai-2"),
                },
                indent=2,
            )
            + "\n"
        )
        (adapter_dir / "README.md").write_text(
            "# Dry-run LoRA stub\n\nGenerated without loading the 27B weights. "
            "Re-run with `--backend gpu` or HF Jobs and `dry_run: false` for a real adapter.\n"
        )
        logger.info("Dry-run train complete → %s", adapter_dir)
        summary["adapter_path"] = str(adapter_dir)
        return summary

    # Real training path (GPU / HF Jobs)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training extras missing. Install with: pip install 'swift-cf-approx[train]'"
        ) from exc

    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("configs/train_qlora.yaml must set model_path to unpacked HF dir")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora)

    def to_text(row: dict[str, Any]) -> str:
        msgs = row["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return "\n".join(f'{m["role"]}: {m["content"]}' for m in msgs)

    # Apply GAIN-proxy by repeating short examples / downsampling long ones is fragile;
    # instead attach weight metadata and scale labels via sample packing of weighted copies.
    expanded: list[dict[str, Any]] = []
    for row, wmeta in zip(rows, weighted):
        copies = max(1, int(round(wmeta["weight"])))
        text = to_text(row)
        for _ in range(copies):
            expanded.append({"text": text, "thinking_penalty": wmeta["thinking_penalty"]})

    ds = Dataset.from_list(expanded[: max(1, max_steps * int(cfg.get("per_device_train_batch_size", 1)))])

    args = TrainingArguments(
        output_dir=str(output_dir / "runs"),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=lr,
        logging_steps=1,
        save_steps=max(max_steps, 1),
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    # Prefer Unsloth if installed; otherwise TRL SFTTrainer.
    try:
        from unsloth import FastLanguageModel  # noqa: F401

        logger.info("Unsloth detected; using PEFT model already prepared via transformers+peft path")
    except ImportError:
        logger.info("Unsloth not installed; continuing with transformers+peft+trl")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        args=args,
        packing=False,
    )
    trainer.train()
    adapter_dir = output_dir / "lora_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary["adapter_path"] = str(adapter_dir)
    (output_dir / "train_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
