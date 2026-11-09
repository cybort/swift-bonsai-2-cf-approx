#!/usr/bin/env bash
# Distill SFT from DavidAU loaded in local LM Studio.
# Run on the host where LM Studio local server is up.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LMSTUDIO_BASE_URL="${LMSTUDIO_BASE_URL:-http://127.0.0.1:1234/v1}"
export LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-}"
OUT_DIR="${OUT_DIR:-artifacts/lmstudio}"
PROMPTS="${PROMPTS:-data/prompts/reason_mode_distill.jsonl}"
LIMIT="${LIMIT:-40}"

mkdir -p "$OUT_DIR" data/distill

python3 scripts/lmstudio_probe_modes.py --base-url "$LMSTUDIO_BASE_URL" --model "$LMSTUDIO_MODEL" --out-dir "$OUT_DIR"

# Resolve model id if unset
if [[ -z "$LMSTUDIO_MODEL" ]]; then
  LMSTUDIO_MODEL="$(python3 - <<'PY'
import json, os, urllib.request
url=os.environ['LMSTUDIO_BASE_URL'].rstrip('/')+'/models'
with urllib.request.urlopen(url, timeout=10) as r:
    data=json.load(r)
print((data.get('data') or [{}])[0].get('id') or 'local-model')
PY
)"
fi

python3 -m swift_cf_approx.cli distill \
  --prompts "$PROMPTS" \
  --out data/distill/lmstudio_davidau.jsonl \
  --base-url "$LMSTUDIO_BASE_URL" \
  --model "$LMSTUDIO_MODEL" \
  --limit "$LIMIT" \
  --require-teacher

python3 scripts/merge_lmstudio_distill.py
echo "LM Studio distill ready → data/distill/prepared_train.jsonl"
