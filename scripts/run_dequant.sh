#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
GGUF="${GGUF:-artifacts/gguf/Swift-Bonsai-2-PTQ1_0.gguf}"
OUT="${OUT:-artifacts/unpacked-swift-ptq1}"
python3 -m pip install -e . -q
mkdir -p "$OUT" logs
echo "Dequant $GGUF → $OUT (streaming; ~54GB disk)"
python3 -m swift_cf_approx.cli dequant --gguf "$GGUF" --out "$OUT" 2>&1 | tee logs/dequant.log
ls -lh "$OUT" | head
