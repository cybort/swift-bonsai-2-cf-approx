"""Distill short-thinking SFT data from a DavidAU (or OpenAI-compatible) teacher."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class DistillExample:
    id: str
    prompt: str
    messages: list[dict[str, str]]
    thinking: str
    answer: str
    thinking_tokens_est: int
    teacher: str
    meta: dict[str, Any]


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    items: list[dict[str, str]] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"id": str(obj.get("id", len(items))), "prompt": obj["prompt"]})
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, row in enumerate(data):
            if isinstance(row, str):
                items.append({"id": str(i), "prompt": row})
            else:
                items.append({"id": str(row.get("id", i)), "prompt": row["prompt"]})
    return items


def _split_thinking(text: str) -> tuple[str, str]:
    m = THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        answer = THINK_RE.sub("", text).strip()
        return thinking, answer
    # Fallback: treat first half as thinking if marked by common Qwen style
    if "Final answer:" in text:
        thinking, _, answer = text.partition("Final answer:")
        return thinking.strip(), answer.strip()
    return "", text.strip()


def _estimate_tokens(text: str) -> int:
    # Rough whitespace/punctuation heuristic; good enough for length penalties.
    return max(1, len(text.split()))


SYSTEM_DISTILL = (
    "You are a precise assistant. Keep internal reasoning short and concrete. "
    "Prefer a brief thinking block, then a clear final answer. "
    "Avoid filler phrases like 'wait', 'let me reconsider', or long restatements."
)


def _chat_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    system: str | None = None,
) -> str:
    headers = {"Content-Type": "application/json"}
    # LM Studio accepts any/non-empty bearer; default placeholder keeps clients happy.
    headers["Authorization"] = f"Bearer {api_key or 'lm-studio'}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    # Mode tags like {REASON:coder} must stay in the user message so the
    # Turbo Brilliance chat_template can strip/inject them.
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"].get("content") or ""
    return content


def _synthetic_teacher(prompt: str) -> str:
    """CPU / offline fallback when no teacher endpoint is available."""
    thinking = (
        f"Goal: answer the user promptly.\n"
        f"Key ask: {prompt[:180]}\n"
        f"Plan: state the answer, then one short justification."
    )
    answer = (
        f"Answer: Here is a concise response to: {prompt[:240]}\n\n"
        f"Justification: Stay direct, avoid redundant chain-of-thought, and cover the ask."
    )
    return f"<think>\n{thinking}\n</think>\n{answer}"


def generate_distill_dataset(
    prompts: Iterable[dict[str, str]],
    out_path: str | Path,
    *,
    teacher_name: str = "DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance",
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "default",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    use_synthetic_if_unreachable: bool = True,
) -> list[DistillExample]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples: list[DistillExample] = []

    for row in prompts:
        pid = row["id"]
        prompt = row["prompt"]
        raw: str
        used_synthetic = False
        if base_url:
            try:
                # Keep SYSTEM_DISTILL only for non-mode prompts so REASON tags
                # are not diluted by an extra system prompt on Turbo templates.
                sys_msg = None if "{REASON:" in prompt else SYSTEM_DISTILL
                raw = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    system=sys_msg,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Teacher call failed for %s: %s", pid, exc)
                if not use_synthetic_if_unreachable:
                    raise
                raw = _synthetic_teacher(prompt)
                used_synthetic = True
        else:
            raw = _synthetic_teacher(prompt)
            used_synthetic = True

        thinking, answer = _split_thinking(raw)
        messages = [
            {"role": "system", "content": SYSTEM_DISTILL},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": (f"<think>\n{thinking}\n</think>\n{answer}" if thinking else answer),
            },
        ]
        ex = DistillExample(
            id=pid,
            prompt=prompt,
            messages=messages,
            thinking=thinking,
            answer=answer,
            thinking_tokens_est=_estimate_tokens(thinking) if thinking else 0,
            teacher=teacher_name + ("#synthetic" if used_synthetic else ""),
            meta={"used_synthetic": used_synthetic},
        )
        examples.append(ex)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
    logger.info("Wrote %d distill examples to %s", len(examples), out_path)
    return examples
