#!/usr/bin/env bash
# After dequant finishes: attach Qwen tokenizer/config and mark base ready.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UNPACKED="${1:-artifacts/unpacked-swift-ptq1}"
TOK="${2:-artifacts/tokenizer-qwen}"

if [[ ! -f "$UNPACKED/unpack_report.json" ]]; then
  echo "missing $UNPACKED/unpack_report.json" >&2
  exit 1
fi

cp -a "$TOK"/. "$UNPACKED"/
# Prefer our unpack config.json (already written); keep Qwen tokenizer files.
# If Qwen config overwrote, restore unpack config from report side-car if needed.
python3 - <<PY
import json
from pathlib import Path
u = Path('$UNPACKED')
cfg = json.loads((u/'config.json').read_text())
# Ensure our provenance tag remains
cfg.setdefault('swift_cf_approx', {})
cfg['swift_cf_approx']['tokenizer_source'] = 'Qwen/Qwen3.8-27B'
cfg['swift_cf_approx']['ready_for_iterate'] = True
(u/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
print('finalized', u, 'files', len(list(u.iterdir())))
PY

python3 - <<'PY'
from pathlib import Path
import yaml, json
p = Path('configs/train_qlora.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['model_path'] = 'artifacts/unpacked-swift-ptq1'
cfg['dry_run'] = False
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
ready = {
  'status': 'READY',
  'unpacked': 'artifacts/unpacked-swift-ptq1',
  'data': 'data/distill/prepared_train.jsonl',
  'next': 'bash scripts/iterate_train.sh',
  'alt_hf_jobs': 'BACKEND=hf-jobs bash scripts/iterate_train.sh',
}
Path('artifacts/READY_FOR_ITERATE.json').write_text(json.dumps(ready, indent=2)+'\n')
print(ready)
PY
