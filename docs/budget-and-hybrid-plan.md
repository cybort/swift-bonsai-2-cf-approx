# Budget: HF 1× L4 vs other clouds vs local 4080 16GB hybrid

Assumptions match the current repo state (2026-11-08):

- Unpacked Swift PTQ1_0 HF base already local: **~54 GB** (`artifacts/unpacked-swift-ptq1/`)
- Train set: **393** rows (`data/distill/prepared_train.jsonl`)
- Config: QLoRA r=16, `max_steps=100`, batch=1, grad_accum=8, `max_seq_length=2048`
- **Not** included: Heretic, full Fable Fusion multi-day cooks, live DavidAU teacher GPU

Prices below are **on-demand list rates** (HF Jobs billed per minute while Starting/Running; no charge while building). Sources: [HF Jobs pricing](https://huggingface.co/docs/hub/en/jobs-pricing), public GPU price aggregators (Nov 2026).

## 1. Hugging Face Jobs — 1× Nvidia L4 @ $0.80/hr

| Spec | Value |
|---|---|
| GPU | 1× L4 **24 GB** |
| CPU / RAM / Disk | 8 vCPU / 30 GB / **400 GB** |
| Hourly | **$0.80** |

### Work breakdown (remaining iterate only)

| Phase | Wall time (est.) | Bills on L4? | Notes |
|---|---:|:---:|---|
| Upload/sync 54 GB weights into job | 15–45 min | Yes if done *inside* the job | Prefer Hub upload from local/CPU first |
| `pip install` + import | 5–15 min | Yes | Cache images if possible |
| QLoRA 100 steps (seq 1024–2048) | **1.0–3.0 h** | Yes | L4 ~similar class to A10 for throughput |
| Eval offline harness | &lt;5 min | Yes | Cheap |
| Retry / OOM restart buffer | 0.5–1.0 h | Yes | First 27B run often needs 1 retry |

### Dollar estimate (L4 only)

| Scenario | GPU-hours | Cost @ $0.80 |
|---|---:|---:|
| **Lean** (weights pre-staged on Hub, 100 steps, seq≈1024, no retry) | **1.5–2.5 h** | **$1.20–$2.00** |
| **Base** (download in-job, 100 steps, seq≈2048, one retry) | **3–5 h** | **$2.40–$4.00** |
| **Comfort** (2–3× data pass / ~150–250 steps + eval live) | **6–10 h** | **$4.80–$8.00** |
| **Multi-stage “FF711-like”** | days | **$50–$200+** | Out of scope; not recommended on L4 hourly |

**Recommended HF budget to finish this project’s iterate step: ≈ $3–$5** (Base), with a **$8–$10** soft cap if you allow retries and longer seq.

### L4 feasibility for 27B QLoRA

- 4-bit 27B weights ≈ **13–15 GB**; LoRA + AdamW (adapter-only) + activations → typically fits **24 GB** at batch 1.
- System RAM **30 GB** is tight if anything accidentally loads BF16; keep `BitsAndBytes` 4-bit + no full merge on GPU.
- Ephemeral **400 GB** is enough for 54 GB weights + env + checkpoints.

### Cost-control tips on HF Jobs

1. Upload `artifacts/unpacked-swift-ptq1` to a Hub model repo **before** starting the GPU job (local bandwidth, $0 GPU).
2. Job script: download Hub pack → train → upload LoRA only (MBs, not 54 GB).
3. Start with `max_seq_length: 1024`, `max_steps: 50` smoke ($0.40–$0.80), then full 100 steps.
4. Prefer **L4 $0.80** over A10G-large $1.50 for this shape; escalate to **L40S $1.80** only if 24 GB OOMs.

## 2. Other online services (same class of work)

Indicative **on-demand $/GPU-hr** (Nov 2026 aggregators; spot/community fluctuate):

| Option | Typical $/hr | VRAM | Fit for this iterate? | Notes |
|---|---:|---:|---|---|
| **HF Jobs 1× L4** | **0.80** | 24 GB | Yes | Best UX with Hub; already scripted |
| HF Jobs T4 medium | 0.60 | 16 GB | Risky | Same VRAM class as 4080; may OOM |
| HF Jobs A10G large | 1.50 | 24 GB | Yes | More CPU/RAM; ~2× L4 price |
| HF Jobs L40S | 1.80 | 48 GB | Comfortable | Overkill for 100-step LoRA |
| HF Jobs A100 | 2.50 | 80 GB | Overkill | Faster steps; higher $/job unless much shorter |
| **RunPod L4** | ~0.40–0.45 | 24 GB | Yes | Often cheaper than HF; DIY Docker |
| **RunPod RTX 4080** | ~0.27 | 16 GB | Borderline | Same VRAM constraint as local |
| **RunPod 4090** | ~0.34 | 24 GB | Yes | Good price/perf for QLoRA |
| Vast.ai L4 / 4090 | ~0.10–0.60 | varies | Yes | Cheapest floor; reliability varies |
| Lambda / CoreWeave | higher / enterprise | — | Optional | Better for multi-day FF711-scale |

**Rough complete-iterate cost if using RunPod 4090 (~$0.34/hr, 2–4 h):** **≈ $0.70–$1.40** — often **2–4× cheaper than HF L4**, at the cost of Hub integration and ops time.

## 3. Local hybrid: RTX 4080 16GB + 48GB RAM

### Can 27B QLoRA fit in 16 GB?

**Borderline.** Expect success only with aggressive settings; plan an OOM fallback.

| Setting | L4 24GB (cloud) | 4080 16GB (local) |
|---|---|---|
| load | 4-bit NF4 | 4-bit NF4 + Unsloth if possible |
| `max_seq_length` | 2048 (ok) | **512–1024** first |
| `per_device_train_batch_size` | 1 | 1 |
| `gradient_accumulation_steps` | 8 | **16–32** (keep effective batch) |
| `lora_r` | 16 | **8–16** |
| gradient checkpointing | on | **required** |
| merge full 54GB on GPU | no | **never** — CPU merge or skip |

Peak VRAM sketch (order-of-magnitude): ~14 GB weights + 1–3 GB activations/opt → **often 15–18 GB** at seq 2048 → OOM on 16 GB; seq 1024 often lands ~14–16 GB.

### Task split (CPU vs GPU)

| Task | Where | Why |
|---|---|---|
| Dataset prep / TeichAI download / JSONL | **CPU** | Already done; re-runs free |
| PTQ1_0 dequant → HF shards | **CPU + 48GB RAM** | Already done (~51 GB disk); 48 GB RAM is enough with chunked f16 path |
| Tokenizer / packing / length filter | **CPU** | Parallelize; write packed Arrow/parquet |
| QLoRA forward/backward | **GPU 16GB** | Only step that needs CUDA |
| Optimizer states | GPU (LoRA only) | Keep adapters small |
| Checkpoint save | CPU disk | Save LoRA only |
| Offline eval harness | **CPU** | Current `eval` stub/live client |
| Baseline Swift GGUF inference | **GPU** (Prism llama.cpp) or CPU slow | Separate from training VRAM |
| Optional teacher distill (DavidAU GGUF) | **GPU** after train frees VRAM | Don’t colocate with 27B QLoRA |
| Full BF16 merge | **CPU RAM** (slow) or skip | 54 GB model + workspace ≫ 16 GB VRAM |

### Suggested local recipe

```text
1) CPU: prepare_all (done) + finalize unpacked base (done)
2) GPU: iterate with configs override:
     max_seq_length: 1024
     lora_r: 8
     gradient_accumulation_steps: 16
     max_steps: 50   # smoke
3) If smoke OK → max_steps: 100–200
4) CPU: eval offline; optional llama.cpp baseline when GPU free
```

Wall time on 4080 for 100 steps: often **1.5–4 h** (Ada is strong; limited by VRAM → shorter seq). **Cash cost: $0** (electricity only).

### Hybrid with HF (best of both)

1. **Local 4080**: smoke 20–50 steps to validate no NaNs / loss curve  
2. **HF L4**: full 100–250 steps if local OOMs or you want longer context  
3. Budget: local $0 + HF **$2–$4**

## 4. Recommendation

| Goal | Pick | Budget |
|---|---|---|
| Cheapest finish | Local 4080 hybrid (seq 1024) | **$0** (+ power) |
| Least ops friction | HF Jobs 1× L4, Hub-staged weights | **$3–$5** (cap **$8**) |
| Best $/hr if renting | RunPod 4090 / Vast 4090 | **~$1–$2** |
| If 16GB/24GB keep OOM | HF L40S one short job | **~$2–$4** for 1–2 h |

**Bottom line for “only iterate left”:** set aside **~$5 on HF L4** or run **local 4080 with seq≤1024**; reserve L40S/A100 only if QLoRA cannot stabilize under 24 GB.
