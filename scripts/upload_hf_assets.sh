#!/usr/bin/env bash
# Upload unpacked Swift weights + prepared distill data to Hugging Face Hub.
# Requires HF_TOKEN with write scope (or `hf auth login`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USER_NS="${HF_NAMESPACE:-cybort}"
MODEL_REPO="${HF_MODEL_REPO:-$USER_NS/swift-bonsai-2-ptq1-unpacked-hf}"
DATA_REPO="${HF_DATA_REPO:-$USER_NS/swift-bonsai-2-cf-approx-distill}"
WEIGHTS="${WEIGHTS_DIR:-$ROOT/artifacts/unpacked-swift-ptq1}"
DATA_DIR="${DATA_DIR:-$ROOT/data/distill}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: HF_TOKEN not set and hf CLI not logged in."
    echo "Add a Write token, then: export HF_TOKEN=... && bash scripts/upload_hf_assets.sh"
    exit 1
  fi
fi

python3 - <<'PY'
from huggingface_hub import HfApi, login
import os
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
api = HfApi(token=token)
who = api.whoami()
print("logged_in_as", who.get("name") or who)
PY

echo "==> Creating repos (idempotent)"
hf repos create "${MODEL_REPO#*/}" --type model --private false 2>/dev/null || true
hf repos create "${DATA_REPO#*/}" --type dataset --private false 2>/dev/null || true
# namespace-qualified create
python3 - <<PY
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get("HF_TOKEN"))
for repo_id, repo_type in [("$MODEL_REPO", "model"), ("$DATA_REPO", "dataset")]:
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
        print("ok", repo_type, repo_id)
    except Exception as e:
        print("warn", repo_id, e)
PY

echo "==> Upload distill dataset (small)"
# Stage a clean dataset folder
STAGE="$ROOT/artifacts/hf-upload/dataset"
mkdir -p "$STAGE"
cp -f "$DATA_DIR"/prepared_train.jsonl "$STAGE"/
cp -f "$ROOT/configs/reason_modes.json" "$STAGE"/ 2>/dev/null || true
cp -f "$ROOT/docs/reason-mode-injection.md" "$STAGE"/ 2>/dev/null || true
cat > "$STAGE/README.md" <<EOF
---
license: apache-2.0
pretty_name: Swift-Bonsai-2 Cold Fusion Approx Distill
tags:
  - distillation
  - qwen
  - lora
  - turbo-brilliance-approx
---

# Swift-Bonsai-2 CF Approx Distill

Prepared SFT JSONL approximating public Cold Fusion / Turbo Brilliance **goals** (short thinking, mode-tagged primers).

- Base target: dequantized ukisai/Swift-Bonsai-2 (PTQ1_0 → HF)
- Teacher path: local LM Studio DavidAU Ultimate Heretic Turbo Brilliance when available
- Mode catalog: \`reason_modes.json\` (public help-menu names/descriptions only; no proprietary construct dumps)

**Not** Polar/F451 private datasets; not original GAIN/Heretic IP.
EOF

hf upload "$DATA_REPO" "$STAGE" . --repo-type dataset
echo "Dataset: https://huggingface.co/datasets/$DATA_REPO"

echo "==> Upload unpacked weights (~54GB; resumable)"
# Model card
cat > "$WEIGHTS/README.md" <<EOF
---
license: apache-2.0
base_model: ukisai/Swift-Bonsai-2-GGUF
tags:
  - qwen
  - safetensors
  - dequantized
  - ptq1_0
pipeline_tag: text-generation
---

# Swift-Bonsai-2 PTQ1_0 → HF (unpacked)

FP16/BF16-compatible safetensors produced by streaming PTQ1_0 dequant of \`ukisai/Swift-Bonsai-2-GGUF\`.

Intended as a QLoRA training base for the open Cold Fusion approx scaffold.
Tokenizer/config attached from Qwen3.8-27B family helpers where applicable.

Attribution: ternary Bonsai / PrismML lineage + ukisai GGUF packaging. Dequant tool: this repo's \`swift_cf_approx.dequant\`.
EOF

hf upload "$MODEL_REPO" "$WEIGHTS" . --repo-type model
echo "Model: https://huggingface.co/$MODEL_REPO"
echo "DONE"
