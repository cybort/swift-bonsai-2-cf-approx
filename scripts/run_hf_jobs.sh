#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m pip install -e . -q
if [[ "${SUBMIT:-0}" == "1" ]]; then
  python3 -m swift_cf_approx.cli hf-jobs --config configs/hf_jobs.yaml --submit
else
  python3 -m swift_cf_approx.cli hf-jobs --config configs/hf_jobs.yaml
  echo "Wrote reports/hf_job_spec.json (set SUBMIT=1 to attempt submission)"
fi
