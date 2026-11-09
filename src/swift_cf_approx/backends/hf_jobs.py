"""Submit training / eval work to Hugging Face Jobs when available."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_job_command(cfg: dict[str, Any]) -> list[str]:
    """Build an `hf jobs` command. Actual submission requires hf CLI + billing."""
    flavor = cfg.get("flavor", "a10g-large")
    timeout = cfg.get("timeout", "6h")
    script = cfg.get(
        "script",
        "pip install -e '.[train]' && python -m swift_cf_approx.cli train --backend gpu --config configs/train_qlora.yaml",
    )
    image = cfg.get("image", "huggingface/transformers-pytorch-gpu")
    # Prefer the modern Jobs API surface; users can paste/adapt.
    cmd = [
        "hf",
        "jobs",
        "run",
        "--flavor",
        str(flavor),
        "--timeout",
        str(timeout),
        image,
        "bash",
        "-lc",
        script,
    ]
    return cmd


def write_job_spec(config_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(Path(config_path))
    cmd = build_job_command(cfg)
    spec = {
        "tool": "huggingface_hub jobs",
        "command": cmd,
        "command_shell": " ".join(shlex.quote(c) for c in cmd),
        "notes": [
            "Requires HF account with Jobs enabled and a write token.",
            "Mount or download unpacked Swift weights + distill JSONL inside the job script.",
            "Upload resulting lora_adapter/ with huggingface-cli upload.",
        ],
        "config": cfg,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def try_submit(config_path: str | Path, *, dry_run: bool = True) -> dict[str, Any]:
    spec = write_job_spec(config_path, Path("reports/hf_job_spec.json"))
    if dry_run:
        logger.info("HF Jobs dry-run only; wrote reports/hf_job_spec.json")
        return {**spec, "submitted": False}
    try:
        proc = subprocess.run(spec["command"], check=False, capture_output=True, text=True)
        return {
            **spec,
            "submitted": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except FileNotFoundError:
        logger.warning("`hf` CLI not found; kept job spec only")
        return {**spec, "submitted": False, "error": "hf_cli_missing"}
