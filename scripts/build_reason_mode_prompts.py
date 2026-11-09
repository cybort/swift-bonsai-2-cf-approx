#!/usr/bin/env python3
"""Build mode-tagged distill prompts from configs/reason_modes.json."""

from __future__ import annotations

import json
from pathlib import Path

TASKS = {
    "logic": "Why is comparison-based sorting Ω(n log n) in the worst case?",
    "coder": "Write a Python function that validates a nested JSON schema of depth ≤3.",
    "teacher": "Teach Bayes' rule with one concrete numeric medical-test example.",
    "brainstorm": "Give 5 product ideas for local LLM tooling for GGUF users.",
    "business": "Outline a 30-day GTM plan for an open-source LoRA training kit.",
    "socrates": "Help me decide whether to train LoRA on 16GB VRAM or rent a cloud L4.",
    "deeptree": "Compare QLoRA vs full FT for a 27B base when the goal is shorter thinking.",
    "hyper": "Decompose 'Cold Fusion approx' into MECE workstreams for a weekend sprint.",
    "omni": "Design an eval harness that measures thinking-token reduction without quality loss.",
    "007": "Plan a fault-tolerant pipeline: distill → QLoRA → eval → upload LoRA.",
    "prompt": "Improve this prompt: 'make my model smarter and shorter'.",
    "story": "Write a 120-word microfiction about a ternary neural net waking up.",
    "comedy": "One-minute monologue about GPU OOMs and hope.",
    "house": "Diagnose why a 27B QLoRA job OOMs at seq=2048 on 16GB VRAM.",
    "musk": "First-principles path to fusion-adjacent energy storage for a lab bench.",
    "curie": "How would you rigorously validate a claimed 1/3 reduction in thinking tokens?",
    "spoon": "Panel of 5 experts: should Swift-Bonsai keep ternary weights for QLoRA?",
    "einstein": "Generate 6 novel angles to cut reasoning tokens without dulling IQ.",
    "ultra": "Produce a bullet checklist for shipping a safe HF upload of 54GB weights.",
    "medium": "Summarize GAIN-proxy in 5 bullets.",
    "low": "One sentence: what is PTQ1_0?",
    "xhigh": "Explain tradeoffs of default xhigh thinking vs explicit mode tags.",
    "xlogic": "Does shorter thinking always reduce answer quality? Argue carefully.",
    "xcoder": "Implement FizzBuzz in Rust for 1..=30 with tests sketched.",
    "xteacher": "Teach gradient accumulation vs larger batch for QLoRA.",
    "agi-teacher-coder": "Explain recursion vs iteration to a junior with one code sample.",
    "agi-musk-curie": "Invent a bench-scale path toward aneutronic fusion diagnosis (conceptual).",
    "agi-brainstorm-coder": "Ideate then sketch an architecture for mode-tagged SFT data generation.",
    "ihelp": "List reasoning modes useful for debugging flaky tests.",
    "icoder": "Write is_palindrome(s: str) -> bool in Python.",
    "ilogic": "State the comparison-sort lower bound in one paragraph.",
    "iteacher": "Define precision vs recall with one example.",
    "off": "What is 17*19? Answer only the number.",
}


def main() -> None:
    catalog = json.loads(Path("configs/reason_modes.json").read_text(encoding="utf-8"))
    known = set(catalog.get("modes", {})) | {
        "xhigh",
        "medium",
        "low",
        "high",
        "ihelp",
        "icoder",
        "ilogic",
        "iteacher",
        "off",
        "xlogic",
        "xcoder",
        "xteacher",
    }
    out = Path("data/prompts/reason_mode_distill.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for mode, task in TASKS.items():
            tag = f"{{REASON:{mode}}}"
            prompt = f"{tag} {task}"
            row = {
                "id": f"reason-{mode}",
                "prompt": prompt,
                "mode": mode,
                "known_in_catalog": mode.lstrip("i") in known or mode in known or mode.startswith("agi-"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"out": str(out), "n": n}))


if __name__ == "__main__":
    main()
