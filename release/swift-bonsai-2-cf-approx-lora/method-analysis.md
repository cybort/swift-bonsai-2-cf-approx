# Method analysis: DavidAU Ultimate vs Swift-Bonsai approx

## DavidAU `Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF`

Public card tags and related Cold Fusion cards describe this pipeline:

1. **Base**: `Qwen/Qwen3.8-27B`
2. **Heretic**: de-censor / abliteration pass before or between tunes
3. **Cold Fusion = proprietary GAIN + Unsloth**: per-sample dynamic training adjustments while Unsloth trains; claimed to keep ~99% of BF16 quality at 4/8-bit and cut thinking tokens (~1/10–1/2)
4. **Multi-stage / Fable Fusion 711 lineage**: heavier multi-day cooks for stronger variants
5. **Turbo Brilliance 50x AGI**: prompt-time reasoning/instruct/AI/AGI constructs baked for on-the-fly mode selection
6. **Export**: BF16 source (`DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored`) → NEO Imatrix / MTP GGUF

### Datasets listed on the card

| Dataset ID | Hub status |
|---|---|
| `DavidAU/Polar-STRICT-Datasets` | Not found as a public dataset/model |
| `DavidAU/F451-STRICT-Datasets` | Not found as a public dataset/model |

Training data for a faithful replica is therefore unavailable. Cold Fusion cards describe goals (raise general IQ, shorten thinking) without releasing recipes or GAIN code.

### Reported public goals (what we approximate)

- Nuanced instruction following with strong reasoning modes
- Reduce thinking / reasoning token usage while keeping answer quality
- Stable low-bit behavior after Unsloth-style QLoRA-style training

## Target base: `ukisai/Swift-Bonsai-2-GGUF`

- Reasoning-efficiency fine-tune of PrismML Ternary Bonsai 2 27B (from Qwen3.8-27B)
- Distributed as **ternary** GGUF only: `PTQ1_0` (~5.95 GB) and `PQ2_0` (~7.21 GB)
- No official BF16 / safetensors HF pack for Swift Bonsai 2
- Community Katana MTP GGUF (`ayshinko/...`) adds MTP head; still not trainable with Unsloth as-is

Precedent for unpack: `prism-ml/Ternary-Bonsai-27B-unpacked` (Bonsai **1** FP16 HF). Bonsai **2** requires open-source dequant of Prism types 142/143.

### Dequant tool choice

| Tool | PTQ1_0 (143) | PQ2_0 (142) |
|---|---|---|
| anamnesis | No | No |
| rlx-gguf `ptq1_dequant` | Yes | No (use Prism kernels / later port) |
| This repo Python PTQ1_0 kernel | Yes (ported from rlx-gguf / C map) | Not in v1 |

**Default unpack source**: `Swift-Bonsai-2-PTQ1_0.gguf` (earlier Swift merge, but open dequant path). `PQ2_0` is newer Swift weights — optional later.

Unpacked size ≈ **54 GB+**. Hadamard / hybrid-attention means Transformers inference may not be bit-faithful; **public baseline remains the original GGUF**.

## This project's approximation boundary

**In scope**

- Stream-dequant Swift PTQ1_0 → HF sharded safetensors
- Distill SFT data from DavidAU teacher outputs (short thinking + strong answers)
- Unsloth-compatible QLoRA + open **GAIN proxy** (per-sample loss scaling) + thinking-length penalty
- Eval: thinking-token stats + small task accuracy; comparison report
- Runners: local GPU, local CPU smoke, HF Jobs

**Out of scope**

- Claiming original GAIN / Fable Fusion / Turbo Brilliance constructs
- Heretic / abliteration
- Faithful Polar/F451 dataset replay
- Guaranteed re-pack to Prism PTQ1_0 after training (ship LoRA + HF artifacts)

## Attribution

- Swift / Bonsai: “Created using Bonsai by Prism ML.” UkisAI Swift Bonsai 2; Apache-2.0 / Prism community terms as applicable
- Teacher: DavidAU models under their stated Apache-2.0 cards; distill data is derived synthetic text for research training only
