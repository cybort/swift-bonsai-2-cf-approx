#!/usr/bin/env python3
"""Write expanded open Turbo-Brilliance-style mode primers (not proprietary construct texts)."""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM = (
    "You are a precise assistant. Keep internal reasoning short and concrete. "
    "Prefer a brief thinking block, then a clear final answer. "
    "Avoid filler like 'wait' or long restatements."
)

# Short open primers keyed by mode tag. Answers are original, not DavidAU construct dumps.
PRIMERS: list[tuple[str, str, str, str]] = [
    ("logic", "{REASON:logic} Why is comparison sorting Ω(n log n)?",
     "First principles: n! permutations; binary comparisons; decision-tree height ≥ log2(n!).",
     "Comparison sorts need Ω(n log n) worst-case comparisons because there are n! orderings and each comparison yields ≤2 outcomes, so any binary decision tree has height ≥ log2(n!)."),
    ("ilogic", "{REASON:ilogic} Why is comparison sorting Ω(n log n)?",
     "Instruct: state the lower bound tightly.",
     "Ω(n log n) comparisons: n! leaves needed; binary tree height ≥ log2(n!)."),
    ("coder", "{REASON:coder} Write balanced_parens(s) in Python.",
     "API + stack + edges, then code.",
     "def balanced_parens(s):\n    st=[]\n    pair={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in pair:\n            if not st or st[-1]!=pair[c]: return False\n            st.pop()\n    return not st"),
    ("icoder", "{REASON:icoder} Write balanced_parens(s) in Python.",
     "Instruct: code only with minimal comment.",
     "def balanced_parens(s):\n    st=[]; m={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in m.values(): st.append(c)\n        elif c in m:\n            if not st or st.pop()!=m[c]: return False\n    return not st"),
    ("teacher", "{REASON:teacher} Teach Bayes with one numeric example.",
     "Define, formula, plug numbers, interpret.",
     "Bayes: P(A|B)=P(B|A)P(A)/P(B). If prior P(A)=0.01, P(B|A)=0.9, P(B|¬A)=0.05, then P(B)=0.0585 and P(A|B)≈0.154."),
    ("brainstorm", "{REASON:brainstorm} 5 product ideas for local LLM tooling.",
     "Diverge briefly, then list five concrete ideas.",
     "1) GGUF eval harness 2) thinking-token budget meter 3) LoRA A/B chat 4) prompt-mode router 5) local dataset distiller UI"),
    ("socrates", "{REASON:socrates} Should I train on 16GB VRAM or rent L4?",
     "Ask clarifying constraints, then a decision frame.",
     "Clarify: seq length needed, deadline, budget. If seq≤1024 and you can iterate overnight, 16GB QLoRA is fine; if you need seq 2048 or fewer retries, rent L4."),
    ("deeptree", "{REASON:deeptree} Paths to shorter thinking on a 27B base.",
     "Branch: data / loss / template / decode; prune weak paths.",
     "Strongest path: SFT on short-think teachers + length penalty. Secondary: mode tags that force instruct. Weak alone: greedy decode without data."),
    ("hyper", "{REASON:hyper} MECE workstreams for Cold Fusion approx.",
     "Partition without overlap.",
     "1) Base unpack 2) Proxy/distill data 3) QLoRA train 4) Eval/compare 5) Hub release. No Heretic/GAIN IP in scope."),
    ("omni", "{REASON:omni} Design an eval for thinking reduction.",
     "Multi-axis: tokens, accuracy, refusal, latency.",
     "Measure median <think> tokens, task accuracy delta, format compliance, and wall latency on a fixed prompt set; gate ship on accuracy ≥ baseline−2% and tokens ≤70% of baseline."),
    ("007", "{REASON:007} Fault-tolerant distill→train→eval→upload plan.",
     "Steps with retries and checkpoints.",
     "1) Smoke distill n=5 2) Full distill resume JSONL 3) QLoRA checkpoint every 25 steps 4) Offline eval 5) Upload LoRA then weights with resume."),
    ("prompt", "{REASON:prompt} Improve: 'make my model smarter and shorter'.",
     "Rewrite into measurable objectives.",
     "Revised: 'Fine-tune so median thinking tokens drop ≥30% on our 50-prompt suite while task accuracy stays within 2 points of baseline; prefer {REASON:logic}/{REASON:icoder} tags.'"),
    ("house", "{REASON:house} Diagnose 27B QLoRA OOM at seq=2048 on 16GB.",
     "Hypothesis differential.",
     "Most likely: activations+4bit weights exceed 16GB at 2048. Fix: seq 512–1024, gradient checkpointing, smaller LoRA r, Unsloth. Less likely: accidental BF16 load."),
    ("xlogic", "{REASON:xlogic} Does shorter thinking always hurt quality?",
     "xhigh+logic: validate claim carefully.",
     "No. Excess tokens often restate. Targeted short plans preserve quality when the model already knows the path; unguided truncation can hurt novel problems."),
    ("xcoder", "{REASON:xcoder} FizzBuzz in Rust for 1..=30.",
     "xhigh+coder: correct code.",
     "fn main(){for i in 1..=30{let s=match(i%3==0,i%5==0){(true,true)=>\"FizzBuzz\".into(),(true,false)=>\"Fizz\".into(),(false,true)=>\"Buzz\".into(),_=>i.to_string()};println!(\"{s}\")}}"),
    ("xteacher", "{REASON:xteacher} Teach grad accumulation vs larger batch for QLoRA.",
     "xhigh+teacher: plain language.",
     "Larger batch needs more VRAM for activations. Grad accumulation runs micro-batches, sums grads, then steps—same effective batch, lower peak VRAM, slightly noisier timing."),
    ("agi-teacher-coder", "{REASON:agi-teacher-coder} Recursion vs iteration for a junior.",
     "Teach + code sample.",
     "Recursion: function calls itself with a smaller case; iteration: loop. Example factorial: recursive n*fact(n-1) vs loop acc*=i. Prefer iteration when depth is large to avoid stack overflow."),
    ("agi-brainstorm-coder", "{REASON:agi-brainstorm-coder} Architecture for mode-tagged SFT generation.",
     "Ideate then sketch.",
     "Components: mode catalog → prompt templates → LM Studio teacher → JSONL normalizer → length filter → train pack. Keep constructs out of repo; only tags + short completions."),
    ("ihelp", "{REASON:ihelp} Modes useful for debugging flaky tests.",
     "Instruct checklist.",
     "Useful: logic, house, 007, coder, socrates. Prefer instruct `i*` when you want a short checklist without long chain-of-thought."),
    ("off", "{REASON:off} What is 17*19?",
     "No enhancements; direct answer.",
     "323"),
]


def main() -> None:
    out = Path("data/distill/mode_primers_expanded.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for mode, prompt, think, answer in PRIMERS:
            content = f"<think>\n{think}\n</think>\n{answer}"
            row = {
                "id": f"primer-{mode}",
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                ],
                "thinking": think,
                "answer": answer,
                "thinking_tokens_est": len(think.split()),
                "teacher": "open-mode-primer-expanded",
                "meta": {"turbo_brilliance_approx": True, "mode": mode, "from_catalog": True},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "n": len(PRIMERS)}))


if __name__ == "__main__":
    main()
