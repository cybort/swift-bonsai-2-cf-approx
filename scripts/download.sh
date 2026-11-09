#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/artifacts/gguf}"
mkdir -p "$OUT"
echo "Downloading Swift-Bonsai-2-PTQ1_0.gguf to $OUT"
huggingface-cli download ukisai/Swift-Bonsai-2-GGUF \
  Swift-Bonsai-2-PTQ1_0.gguf \
  --local-dir "$OUT"
echo "Optional teacher GGUF (large): set DOWNLOAD_TEACHER=1"
if [[ "${DOWNLOAD_TEACHER:-0}" == "1" ]]; then
  huggingface-cli download \
    DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF \
    --local-dir "$ROOT/artifacts/teacher-gguf" \
    --include "*.gguf"
fi
