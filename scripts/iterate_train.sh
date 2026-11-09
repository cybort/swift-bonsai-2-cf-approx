#!/usr/bin/env bash
# FINAL step only: auto-iterate QLoRA on prepared unpacked base + prepared_train.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

UNPACKED="${UNPACKED:-artifacts/unpacked-swift-ptq1}"
DATA="${DATA:-data/distill/prepared_train.jsonl}"
OUT="${OUT:-artifacts/train}"
BACKEND="${BACKEND:-gpu}"  # gpu | hf-jobs

if [[ ! -f "$DATA" ]]; then
  echo "Missing $DATA — run scripts/prepare_all.sh first" >&2
  exit 1
fi

echo "Waiting for unpacked base at $UNPACKED/unpack_report.json ..."
for i in $(seq 1 720); do
  if [[ -f "$UNPACKED/unpack_report.json" ]]; then
    break
  fi
  sleep 10
done
if [[ ! -f "$UNPACKED/unpack_report.json" ]]; then
  echo "Unpacked base not ready. Start: bash scripts/run_dequant.sh" >&2
  exit 1
fi

# Point train config at unpacked dir
python3 - <<PY
from pathlib import Path
import yaml
p = Path('configs/train_qlora.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['model_path'] = '$UNPACKED'
cfg['dry_run'] = False
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('updated', p, 'model_path=', cfg['model_path'])
PY

python3 -m pip install -e '.[train]' -q

if [[ "$BACKEND" == "hf-jobs" ]]; then
  python3 -m swift_cf_approx.cli hf-jobs --config configs/hf_jobs.yaml --submit
  exit 0
fi

python3 -m swift_cf_approx.cli train \
  --backend gpu \
  --config configs/train_qlora.yaml \
  --gain-config configs/gain_proxy.yaml \
  --data "$DATA" \
  --out "$OUT" \
  --no-dry-run

python3 -m swift_cf_approx.cli eval \
  --config configs/eval.yaml \
  --out reports \
  --adapter "$OUT/lora_adapter"

echo "ITERATE done → $OUT/lora_adapter and reports/"
