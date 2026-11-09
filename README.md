# Swift-Bonsai Cold Fusion Approx

Open scaffold to approximate DavidAU Cold Fusion / Turbo Brilliance **public goals** (shorter thinking, stronger instruction following) on dequantized [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF).

> Created using Bonsai by Prism ML. Swift Bonsai 2 by UkisAI. Teacher models by DavidAU.

## Layout

| Where | What |
|---|---|
| **This GitHub repo** | Scripts, docs, configs, Python package (code) |
| [HF dataset](https://huggingface.co/datasets/cybort/swift-bonsai-2-cf-approx-distill) | Prepared distill JSONL + mode catalog |
| [HF model](https://huggingface.co/cybort/swift-bonsai-2-ptq1-unpacked-hf) | Unpacked PTQ1_0 → safetensors (~54GB) |
| HF dataset `source/*.tar.gz` | Full source snapshot (if a file is still catching up on GitHub) |

```bash
# Data (no large JSONL in git)
huggingface-cli download cybort/swift-bonsai-2-cf-approx-distill --repo-type dataset --local-dir data/distill

# Optional: full source tarball
huggingface-cli download cybort/swift-bonsai-2-cf-approx-distill source/swift-bonsai-2-cf-approx-source.tar.gz --repo-type dataset
```

## Install & LM Studio

```bash
pip install -e ".[dev]"

# Linux/macOS — LM Studio local server on :1234
bash scripts/lmstudio_distill.sh

# Windows
.\scripts\lmstudio_distill.ps1
```

Docs: `docs/reason-mode-injection.md`, `docs/method-analysis.md`, `docs/gain-fable-turbo-research.md`, `docs/budget-and-hybrid-plan.md`.

## Train

```bash
bash scripts/iterate_train.sh
# or: BACKEND=hf-jobs bash scripts/iterate_train.sh
```
