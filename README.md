# Swift-Bonsai Cold Fusion Approx

Open scaffold to **approximate** the public goals of DavidAU Cold Fusion / Ultimate Heretic Turbo Brilliance (shorter thinking, stronger instruction following) on a **dequantized** [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) base.

This is **not** a 1:1 replica: proprietary GAIN, Polar/F451 datasets, Heretic, and Turbo Brilliance constructs are unavailable or out of scope. See [`docs/method-analysis.md`](docs/method-analysis.md).

> Created using Bonsai by Prism ML. Swift Bonsai 2 by UkisAI. Teacher models by DavidAU.

## Data & weights (Hugging Face)

GitHub keeps **scripts / docs / code** only. Training JSONL and unpacked weights stay on Hub:

| Asset | URL |
|---|---|
| Distill dataset | https://huggingface.co/datasets/cybort/swift-bonsai-2-cf-approx-distill |
| Unpacked PTQ1_0 → HF base | https://huggingface.co/cybort/swift-bonsai-2-ptq1-unpacked-hf |

```bash
huggingface-cli download cybort/swift-bonsai-2-cf-approx-distill --repo-type dataset --local-dir data/distill
```

## Install

```bash
python -m pip install -e ".[dev]"
# GPU training: python -m pip install -e ".[train]"
```

## LM Studio teacher

```bash
# Linux / macOS
bash scripts/lmstudio_distill.sh

# Windows PowerShell
.\scripts\lmstudio_distill.ps1
# or: scripts\lmstudio_distill.cmd
```

Docs: [`docs/reason-mode-injection.md`](docs/reason-mode-injection.md), [`docs/method-analysis.md`](docs/method-analysis.md), [`docs/gain-fable-turbo-research.md`](docs/gain-fable-turbo-research.md), [`docs/budget-and-hybrid-plan.md`](docs/budget-and-hybrid-plan.md).

## Iterate train

```bash
bash scripts/iterate_train.sh
# or: BACKEND=hf-jobs bash scripts/iterate_train.sh
```
