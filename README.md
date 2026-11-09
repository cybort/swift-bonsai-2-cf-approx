# Swift-Bonsai Cold Fusion Approx

Open scaffold to **approximate** the public goals of DavidAU Cold Fusion / Ultimate Heretic Turbo Brilliance (shorter thinking, stronger instruction following) on a **dequantized** [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) base.

This is **not** a 1:1 replica: proprietary GAIN, Polar/F451 datasets, Heretic, and Turbo Brilliance constructs are unavailable or out of scope. See [`docs/method-analysis.md`](docs/method-analysis.md).

> Created using Bonsai by Prism ML. Swift Bonsai 2 by UkisAI. Teacher models by DavidAU.

## What you get

| Piece | Path |
|---|---|
| Method analysis | `docs/method-analysis.md` |
| PTQ1_0 dequant → HF shards | `python -m swift_cf_approx.cli dequant` |
| DavidAU / synthetic distill | `python -m swift_cf_approx.cli distill` |
| QLoRA + GAIN-proxy + thinking penalty | `python -m swift_cf_approx.cli train` |
| Scores + comparison | `reports/scores.json`, `reports/comparison.md` |
| Optional Rust PTQ1_0 CLI | `crates/ptq1_dequant_cli` |

## Disk / VRAM (27B)

| Stage | Approx |
|---|---|
| Download PTQ1_0 GGUF | ~6 GB |
| Unpacked FP16 safetensors | ~54 GB+ |
| QLoRA train (4-bit + adapters) | high-end GPU (e.g. 24–48 GB+) or HF Jobs |
| CPU smoke | no full model download required |

## Install

```bash
python -m pip install -e '.[dev]'
# GPU training extras:
python -m pip install -e '.[train]'
# Optional: Unsloth on CUDA hosts for faster LoRA
```

## Recommended flow (prep everything, leave only iterate)

```bash
# 1) Research + proxy data + synthetic distill + ensure GGUF present
bash scripts/prepare_all.sh

# 2) Dequant Swift PTQ1_0 → HF shards (~54GB disk; runs in foreground or tmux)
bash scripts/run_dequant.sh
bash scripts/finalize_unpacked_base.sh

# 3) ONLY remaining step — needs GPU or HF Jobs
bash scripts/iterate_train.sh
# or: BACKEND=hf-jobs bash scripts/iterate_train.sh
```

Deep dive: [`docs/gain-fable-turbo-research.md`](docs/gain-fable-turbo-research.md).  
REASON mode injection (public catalog + LM Studio path): [`docs/reason-mode-injection.md`](docs/reason-mode-injection.md).

### Data & weights (Hugging Face)

GitHub keeps scripts/docs/code only. Pull data/weights from Hub:

| Asset | URL |
|---|---|
| Prepared distill dataset | https://huggingface.co/datasets/cybort/swift-bonsai-2-cf-approx-distill |
| Unpacked PTQ1_0 → HF base | https://huggingface.co/cybort/swift-bonsai-2-ptq1-unpacked-hf |

```bash
huggingface-cli download cybort/swift-bonsai-2-cf-approx-distill --repo-type dataset --local-dir data/distill
# optional ~54GB weights:
# huggingface-cli download cybort/swift-bonsai-2-ptq1-unpacked-hf --local-dir artifacts/unpacked-swift-ptq1
```

### LM Studio teacher (your machine)

Load DavidAU `…Turbo-Brilliance-50x-AGI-NEO` in LM Studio, enable the local server, then:

```bash
# Linux / macOS
bash scripts/lmstudio_distill.sh

# Windows (PowerShell)
.\scripts\lmstudio_distill.ps1
# or: scripts\lmstudio_distill.cmd
```

Probes `{REASON:help}` / modes → `data/distill/lmstudio_davidau.jsonl` → merges into `prepared_train.jsonl` (after you downloaded the Hub dataset or created a local one).

### Local CPU smoke (no 27B weights)

```bash
bash scripts/run_local_cpu_smoke.sh
```

### Hugging Face Jobs

```bash
bash scripts/run_hf_jobs.sh          # writes reports/hf_job_spec.json
SUBMIT=1 bash scripts/run_hf_jobs.sh  # attempts `hf jobs run` if CLI/billing allow
```

## CLI

```bash
python -m swift_cf_approx.cli dequant --gguf path/to/Swift-Bonsai-2-PTQ1_0.gguf --out artifacts/unpacked-swift-ptq1
python -m swift_cf_approx.cli distill --prompts data/prompts/distill_seed.jsonl --out data/distill/train.jsonl --base-url http://127.0.0.1:8080/v1
python -m swift_cf_approx.cli train --backend gpu --data data/distill/train.jsonl --out artifacts/train --no-dry-run
python -m swift_cf_approx.cli eval --adapter artifacts/train/lora_adapter --out reports
python -m swift_cf_approx.cli pipeline --backend cpu --limit 8
```

## Upload artifacts

Unpacked weights (~54GB) + prepared distill data:

```bash
export HF_TOKEN=...   # Write scope
bash scripts/upload_hf_assets.sh
```

Published:
- Model (PTQ1_0 → HF shards): https://huggingface.co/cybort/swift-bonsai-2-ptq1-unpacked-hf
- Dataset (prepared distill + mode catalog): https://huggingface.co/datasets/cybort/swift-bonsai-2-cf-approx-distill

Dry-run LoRA stub + scores ship in [`release/swift-bonsai-2-cf-approx-lora/`](release/swift-bonsai-2-cf-approx-lora/). Publish to the Hub after `hf auth login` (or `HF_TOKEN`):

```bash
HF_REPO=cybort/swift-bonsai-2-cf-approx-lora bash scripts/upload_artifacts.sh
```

This Cloud Agent session has Hugging Face **MCP read/jobs** OAuth only; shell upload needs a user access token with write scope.

## Important limits

- Prefer **original Swift GGUF** (Prism llama.cpp) for baseline inference scores; unpacked HF may be non-faithful due to Hadamard / hybrid attention.
- Default unpack uses **PTQ1_0** (open `rlx-gguf`-aligned kernel). `PQ2_0` has newer Swift merges but needs a separate unpack path.
- Offline eval stubs exercise the harness; point `configs/eval.yaml` endpoints and pass `--live` for real numbers.
