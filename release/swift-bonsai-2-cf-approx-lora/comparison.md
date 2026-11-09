# Swift-Bonsai Cold Fusion Approx — comparison

- Prompts: 6
- Adapter: `/workspace/artifacts/train/lora_adapter`

| Model | Accuracy | Mean thinking tokens | Median thinking tokens | Mode |
|---|---:|---:|---:|---|
| Baseline Swift GGUF | 1.000 | 56.8 | 57 | offline_stub |
| Tuned LoRA | 1.000 | 11.5 | 12 | offline_stub |

**Accuracy Δ**: +0.000
**Thinking token reduction**: 79.8%

## Notes

- Offline stub mode exercises the harness without 27B GPUs.
- Point `endpoints.baseline_gguf` / `endpoints.tuned_hf` in `configs/eval.yaml` and pass `--live` for real scores.
- This is an open approximation of Cold Fusion goals, not DavidAU proprietary GAIN.
- Created using Bonsai by Prism ML (Swift / Bonsai lineage).
