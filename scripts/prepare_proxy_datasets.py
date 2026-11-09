#!/usr/bin/env python3
"""Pull public TeichAI proxy datasets and normalize to distill JSONL schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _est_tokens(text: str) -> int:
    return max(0, len(text.split()))


def _split_thinking(text: str) -> tuple[str, str]:
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), THINK_RE.sub("", text).strip()
    return "", text.strip()


def _as_messages(prompt: str, assistant: str, system: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]


SYSTEM = (
    "You are a precise assistant. Keep internal reasoning short and concrete. "
    "Prefer a brief thinking block, then a clear final answer. "
    "Avoid filler like 'wait' or long restatements."
)


def _row(ex_id: str, prompt: str, assistant: str, teacher: str, meta: dict[str, Any]) -> dict[str, Any]:
    thinking, answer = _split_thinking(assistant)
    if thinking:
        content = f"<think>\n{thinking}\n</think>\n{answer}"
    else:
        # Encourage short think even when source has none
        thinking = "Plan briefly, then answer."
        content = f"<think>\n{thinking}\n</think>\n{assistant.strip()}"
        answer = assistant.strip()
    return {
        "id": ex_id,
        "prompt": prompt,
        "messages": _as_messages(prompt, content, SYSTEM),
        "thinking": thinking,
        "answer": answer,
        "thinking_tokens_est": _est_tokens(thinking),
        "teacher": teacher,
        "meta": meta,
    }


def _pick_text(obj: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            # chat messages
            if isinstance(v[0], dict) and "content" in v[0]:
                user = next((m.get("content") for m in v if m.get("role") == "user"), None)
                asst = next((m.get("content") for m in v if m.get("role") == "assistant"), None)
                if user and asst:
                    return json.dumps({"user": user, "assistant": asst})
    return None


def convert_teichai(repo: str, limit: int, out: Path, tag: str) -> int:
    ds = load_dataset(repo, split="train")
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if n >= limit:
                break
            # Common TeichAI layouts vary; try several fields
            if "messages" in row and isinstance(row["messages"], list):
                msgs = row["messages"]
                user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
                asst = next((m.get("content") for m in msgs if m.get("role") == "assistant"), None)
                if not user or not asst:
                    continue
                rec = _row(f"{tag}-{i}", user, asst, repo, {"source_tag": tag})
            else:
                prompt = _pick_text(row, ["prompt", "question", "input", "user", "instruction"])
                answer = _pick_text(row, ["response", "output", "answer", "completion", "assistant"])
                if not prompt or not answer:
                    # last resort: dump first string fields
                    strs = [v for v in row.values() if isinstance(v, str) and len(v) > 20]
                    if len(strs) < 2:
                        continue
                    prompt, answer = strs[0], strs[1]
                rec = _row(f"{tag}-{i}", prompt, answer, repo, {"source_tag": tag})
            # Prefer shorter thinking for Cold Fusion approx: trim long thinks
            if rec["thinking_tokens_est"] > 256:
                short = " ".join(rec["thinking"].split()[:80])
                rec["thinking"] = short
                rec["thinking_tokens_est"] = _est_tokens(short)
                rec["messages"][-1]["content"] = f"<think>\n{short}\n</think>\n{rec['answer']}"
                rec["meta"]["thinking_truncated"] = True
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def add_mode_primers(out: Path) -> int:
    """Compact open Turbo-Brilliance-style mode primers (not proprietary construct texts)."""
    primers = [
        ("mode-logic", "{REASON:logic} Explain why sorting is O(n log n) for comparison sorts.", "Use first principles: comparison lower bound, decision tree."),
        ("mode-coder", "{REASON:coder} Write a Python function to check balanced parentheses.", "Specify API, edge cases, then code."),
        ("mode-teacher", "{REASON:teacher} Teach Bayes' rule with one numeric example.", "Define terms, show formula, plug numbers, interpret."),
        ("mode-brainstorm", "{REASON:brainstorm} Give 5 product ideas for local LLM tooling.", "Diverge briefly, then list 5 concrete ideas."),
        ("mode-ihelp", "{REASON:ihelp} List reasoning modes useful for debugging flaky tests.", "Instruct mode: no long chain-of-thought; bullet the modes."),
    ]
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for pid, prompt, think_hint in primers:
            answer_map = {
                "mode-logic": "Comparison-based sorting needs Ω(n log n) comparisons in the worst case because there are n! orderings and each comparison has 2 outcomes, so the decision-tree height is at least log2(n!).",
                "mode-coder": "def balanced(s):\n    st=[]\n    pair={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in pair:\n            if not st or st[-1]!=pair[c]: return False\n            st.pop()\n    return not st",
                "mode-teacher": "Bayes: P(A|B)=P(B|A)P(A)/P(B). If P(A)=0.01, P(B|A)=0.9, P(B|¬A)=0.05, then P(B)=0.9*0.01+0.05*0.99=0.0585, so P(A|B)≈0.154.",
                "mode-brainstorm": "1) GGUF eval harness 2) thinking-token budget meter 3) LoRA A/B chat 4) prompt-mode router 5) local dataset distiller UI",
                "mode-ihelp": "Useful modes: logic, coder, socrates, medium, low. Prefer instruct (`i*`) when you need a quick checklist.",
            }
            ans = answer_map[pid]
            content = f"<think>\n{think_hint}\n</think>\n{ans}"
            rec = {
                "id": pid,
                "prompt": prompt,
                "messages": _as_messages(prompt, content, SYSTEM),
                "thinking": think_hint,
                "answer": ans,
                "thinking_tokens_est": _est_tokens(think_hint),
                "teacher": "open-mode-primer",
                "meta": {"turbo_brilliance_approx": True},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/distill/prepared_train.jsonl")
    ap.add_argument("--per-source", type=int, default=120)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    sources = [
        ("TeichAI/polaris-alpha-1000x", "polaris"),
        ("TeichAI/Claude-Opus-4.6-Reasoning-887x", "reasoning"),
        ("TeichAI/Fable-5-Cursor-Traces", "fable5"),
    ]
    total = 0
    for repo, tag in sources:
        try:
            c = convert_teichai(repo, args.per_source, out, tag)
            print(f"{repo}: {c}")
            total += c
        except Exception as exc:  # noqa: BLE001
            print(f"WARN {repo}: {exc}")

    total += add_mode_primers(out)
    # Merge existing synthetic distill if present
    synth = Path("data/distill/train.jsonl")
    if synth.exists():
        with out.open("a", encoding="utf-8") as dest, synth.open(encoding="utf-8") as src:
            for line in src:
                dest.write(line)
                total += 1
    print(json.dumps({"out": str(out), "n_lines": total, "bytes": out.stat().st_size}))


if __name__ == "__main__":
    main()
