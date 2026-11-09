#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${HF_REPO:-cybort/swift-bonsai-2-cf-approx-lora}"
SRC="${1:-$ROOT/release/swift-bonsai-2-cf-approx-lora}"
if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "Login required: hf auth login   OR   export HF_TOKEN=..."
  echo "Then: hf repos create ${REPO##*/} --type model || true"
  echo "      hf upload $REPO $SRC . --repo-type model"
  exit 1
fi
hf repos create "${REPO#*/}" --type model 2>/dev/null || true
hf upload "$REPO" "$SRC" . --repo-type model
echo "Uploaded to https://huggingface.co/$REPO"
