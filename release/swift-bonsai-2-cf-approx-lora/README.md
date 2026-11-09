---
license: apache-2.0
base_model: ukisai/Swift-Bonsai-2-GGUF
tags:
  - lora
  - qlora
  - gguf
  - swift-bonsai-2
  - cold-fusion-approx
  - unsloth
library_name: peft
---

# Swift-Bonsai-2 Cold Fusion Approx LoRA

Open **approximation** of DavidAU Cold Fusion public goals (shorter thinking traces, stronger instruction following) targeting dequantized [ukisai/Swift-Bonsai-2-GGUF](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) (PTQ1_0).

## Status

This package currently contains a **CPU dry-run adapter stub** plus offline evaluation harness scores from the scaffold repo. Re-run on GPU / HF Jobs with `dry_run: false` after unpacking the real Swift PTQ1_0 GGUF to produce trainable LoRA weights.

## Offline harness scores

See `scores.json` / `comparison.md` in this folder. Offline stub mode shows large thinking-token reduction by construction; use `--live` endpoints for real numbers.

## Attribution

Created using Bonsai by Prism ML. Swift Bonsai 2 by UkisAI. Teacher / method references: DavidAU Cold Fusion / Ultimate Heretic cards (not a 1:1 replica; GAIN and Polar/F451 datasets are not public).

## Reproduce

Clone the accompanying git repository and run `bash scripts/run_local_cpu_smoke.sh` or `bash scripts/run_local_gpu.sh`.
