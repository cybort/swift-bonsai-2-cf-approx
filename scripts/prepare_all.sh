#!/usr/bin/env bash
# Run everything except the final GPU/HF auto-iterate train step.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

python3 -m pip install -e '.[dev]' -q
python3 -m pip install 'datasets>=3.0' -q

echo "== unit tests =="
python3 -m pytest -q

echo "== ensure Swift GGUF =="
if [[ ! -f artifacts/gguf/Swift-Bonsai-2-PTQ1_0.gguf ]]; then
  python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('ukisai/Swift-Bonsai-2-GGUF','Swift-Bonsai-2-PTQ1_0.gguf', local_dir='artifacts/gguf')
PY
fi
ls -lh artifacts/gguf/Swift-Bonsai-2-PTQ1_0.gguf

echo "== proxy + mode-primer datasets =="
python3 scripts/prepare_proxy_datasets.py --out data/distill/prepared_train.jsonl --per-source 120
wc -l data/distill/prepared_train.jsonl

echo "== expand seed prompts =="
python3 - <<'PY'
import json
from pathlib import Path
seed = Path('data/prompts/distill_seed.jsonl')
extra = Path('data/prompts/distill_expanded.jsonl')
rows = [json.loads(l) for l in seed.read_text().splitlines() if l.strip()]
# Add Cold-Fusion-style tasks
more = [
    "Compress this reasoning: restating the question three times wastes tokens. Rewrite a short think+answer about why 2+2=4.",
    "Solve: A train travels 120 km in 1.5 hours. What is its average speed? Brief think.",
    "Write a minimal reproducible example of a Python KeyError and how to fix it.",
    "Compare PQ2_0 vs PTQ1_0 packing for ternary GGUF in three bullets.",
    "Given a flaky CI test, list a debugging checklist (instruct style, no long CoT).",
    "{REASON:logic} Why does reducing thinking tokens not always reduce answer quality?",
    "{REASON:coder} Implement FizzBuzz in Rust for 1..=30.",
    "Explain Hadamard-rotated ternary weights at a high level for practitioners.",
]
base = len(rows)
for i, p in enumerate(more):
    rows.append({"id": f"x{base+i}", "prompt": p})
extra.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
print('wrote', extra, 'n=', len(rows))
PY

echo "== synthetic distill from expanded prompts (no teacher endpoint) =="
python3 -m swift_cf_approx.cli distill \
  --prompts data/prompts/distill_expanded.jsonl \
  --out data/distill/synthetic_expanded.jsonl
# Append into prepared set
cat data/distill/synthetic_expanded.jsonl >> data/distill/prepared_train.jsonl
wc -l data/distill/prepared_train.jsonl

echo "== write ready marker + iterate config =="
python3 - <<'PY'
import json
from pathlib import Path
meta = {
  "base_gguf": "artifacts/gguf/Swift-Bonsai-2-PTQ1_0.gguf",
  "unpacked_dir": "artifacts/unpacked-swift-ptq1",
  "train_data": "data/distill/prepared_train.jsonl",
  "train_config": "configs/train_qlora.yaml",
  "gain_config": "configs/gain_proxy.yaml",
  "next_step": "bash scripts/iterate_train.sh",
  "notes": "Dequant may still be running; iterate_train waits for unpack_report.json",
}
Path('artifacts/READY_FOR_ITERATE.json').write_text(json.dumps(meta, indent=2)+'\n')
print(meta)
PY

echo "PREPARE_ALL done (dequant launched separately if needed)"
