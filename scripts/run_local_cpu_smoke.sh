#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m pip install -e '.[dev]' -q
python3 -m pytest -q
python3 -m swift_cf_approx.cli pipeline --backend cpu --root "$ROOT" --limit 8
python3 -m swift_cf_approx.cli hf-jobs --config configs/hf_jobs.yaml
echo "CPU smoke OK. See reports/comparison.md and artifacts/train/lora_adapter"
