#!/usr/bin/env python3
"""Merge LM Studio distill + expanded mode primers into prepared_train.jsonl."""

from __future__ import annotations

import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    base = Path("data/distill/prepared_train.jsonl")
    extras = [
        Path("data/distill/lmstudio_davidau.jsonl"),
        Path("data/distill/mode_primers_expanded.jsonl"),
    ]
    seen: set[str] = set()
    out_rows: list[dict] = []
    for path in [base, *extras]:
        for row in _load_jsonl(path):
            key = str(row.get("id") or row.get("prompt"))
            if key in seen:
                continue
            seen.add(key)
            if path.name.startswith("lmstudio"):
                row["teacher"] = row.get("teacher") or "lmstudio-davidau"
                row.setdefault("meta", {})["lmstudio"] = True
            out_rows.append(row)
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(base), "n": len(out_rows), "bytes": base.stat().st_size}))


if __name__ == "__main__":
    main()
