#!/usr/bin/env python3
"""Probe DavidAU REASON modes via local LM Studio OpenAI API.

Cloud agents cannot reach your host localhost — run this on the machine
where LM Studio has the Ultimate Heretic Turbo Brilliance GGUF loaded and
the local server enabled (default http://127.0.0.1:1234/v1).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import httpx

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

DEFAULT_PROBES = [
    ("help", "{REASON:help} Show the reasoning/instruct mode menu for coding and research."),
    ("ihelp", "{REASON:ihelp} List the best modes for debugging flaky tests. Be concise."),
    ("logic", "{REASON:logic} In one short paragraph: why is comparison sorting Ω(n log n)?"),
    ("coder", "{REASON:coder} Write a Python function is_palindrome(s: str) -> bool."),
    ("icoder", "{REASON:icoder} Write a Python function is_palindrome(s: str) -> bool."),
    ("teacher", "{REASON:teacher} Teach Bayes' rule with one numeric example."),
    ("xlogic", "{REASON:xlogic} Why does reducing thinking tokens not always hurt answer quality?"),
    ("agi-teacher-coder", "{REASON:agi-teacher-coder} Explain recursion vs iteration to a junior with one code sample."),
    ("off", "{REASON:off} What is 17*19? Answer only the number."),
]


def chat(base_url: str, model: str, prompt: str, *, max_tokens: int, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def analyze(text: str) -> dict:
    m = THINK_RE.search(text or "")
    thinking = m.group(1).strip() if m else ""
    answer = THINK_RE.sub("", text or "").strip()
    return {
        "has_think_block": bool(m),
        "thinking_chars": len(thinking),
        "thinking_tokens_est": len(thinking.split()) if thinking else 0,
        "answer_chars": len(answer),
        "answer_preview": answer[:400],
        "thinking_preview": thinking[:400],
    }


def list_models(base_url: str, timeout: float) -> list[str]:
    with httpx.Client(timeout=timeout) as client:
        r = client.get(base_url.rstrip("/") + "/models")
        r.raise_for_status()
        data = r.json()
    return [m.get("id", "") for m in data.get("data", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    ap.add_argument("--model", default=os.environ.get("LMSTUDIO_MODEL", ""))
    ap.add_argument("--out-dir", default="artifacts/lmstudio")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of probes (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        models = list_models(args.base_url, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"Cannot reach LM Studio at {args.base_url}: {exc}"}))
        return 2

    model = args.model or (models[0] if models else "local-model")
    probes = DEFAULT_PROBES[: args.limit] if args.limit else DEFAULT_PROBES
    results = []
    for mode, prompt in probes:
        t0 = time.time()
        try:
            raw = chat(args.base_url, model, prompt, max_tokens=args.max_tokens, timeout=args.timeout)
            content = raw["choices"][0]["message"].get("content") or ""
            row = {
                "mode": mode,
                "prompt": prompt,
                "latency_s": round(time.time() - t0, 2),
                "content": content,
                "analysis": analyze(content),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "mode": mode,
                "prompt": prompt,
                "latency_s": round(time.time() - t0, 2),
                "content": "",
                "analysis": {},
                "error": str(exc),
            }
        results.append(row)
        (out_dir / f"probe_{mode}.txt").write_text(row.get("content") or row.get("error") or "", encoding="utf-8")
        print(json.dumps({"mode": mode, "ok": row["error"] is None, "analysis": row["analysis"]}, ensure_ascii=False))

    summary = {
        "base_url": args.base_url,
        "model": model,
        "available_models": models,
        "n": len(results),
        "results": results,
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Extract help body for catalog merge
    help_row = next((r for r in results if r["mode"] == "help" and r["content"]), None)
    if help_row:
        (out_dir / "help_menu_live.md").write_text(help_row["content"], encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir), "model": model, "n": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
