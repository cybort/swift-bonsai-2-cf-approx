#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m pip install -e '.[train]'
GGUF="${GGUF:-$ROOT/artifacts/gguf/Swift-Bonsai-2-PTQ1_0.gguf}"
if [[ ! -f "$GGUF" ]]; then
  bash "$ROOT/scripts/download.sh" "$ROOT/artifacts/gguf"
fi
python3 -m swift_cf_approx.cli dequant --gguf "$GGUF" --out "$ROOT/artifacts/unpacked-swift-ptq1"
TEACHER_URL="${TEACHER_URL:-}"
python3 -m swift_cf_approx.cli distill \
  --prompts data/prompts/distill_seed.jsonl \
  --out data/distill/train.jsonl \
  ${TEACHER_URL:+--base-url "$TEACHER_URL"}
python3 -m swift_cf_approx.cli train \
  --backend gpu \
  --config configs/train_qlora.yaml \
  --gain-config configs/gain_proxy.yaml \
  --data data/distill/train.jsonl \
  --out artifacts/train \
  --no-dry-run
python3 -m swift_cf_approx.cli eval \
  --config configs/eval.yaml \
  --out reports \
  --adapter artifacts/train/lora_adapter \
  ${LIVE:+--live}
