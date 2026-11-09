# Distill data (Hub-hosted)

Training JSONL is **not** stored in GitHub. Pull from Hugging Face:

```bash
huggingface-cli download cybort/swift-bonsai-2-cf-approx-distill \
  --repo-type dataset \
  --local-dir data/distill
```

- Dataset: https://huggingface.co/datasets/cybort/swift-bonsai-2-cf-approx-distill  
- Unpacked base weights: https://huggingface.co/cybort/swift-bonsai-2-ptq1-unpacked-hf  

Prompt banks used to *generate* distill rows remain in `data/prompts/` in this repo.
