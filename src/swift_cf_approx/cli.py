"""Unified CLI: python -m swift_cf_approx.cli ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("swift_cf_approx")


def _cmd_dequant(args: argparse.Namespace) -> int:
    from swift_cf_approx.dequant import unpack_swift_ptq1_to_hf

    report = unpack_swift_ptq1_to_hf(
        args.gguf,
        args.out,
        max_shard_bytes=args.max_shard_bytes,
        max_tensors=args.max_tensors,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    from swift_cf_approx.distill import generate_distill_dataset, load_prompts

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    examples = generate_distill_dataset(
        prompts,
        args.out,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        use_synthetic_if_unreachable=not args.require_teacher,
    )
    print(json.dumps({"n": len(examples), "out": args.out}, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from swift_cf_approx.train import run_train

    summary = run_train(
        config_path=args.config,
        gain_config_path=args.gain_config,
        data_path=args.data,
        output_dir=args.out,
        backend=args.backend,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from swift_cf_approx.eval import run_eval_compare

    result = run_eval_compare(
        config_path=args.config,
        output_dir=args.out,
        adapter_path=args.adapter,
        use_live_endpoints=args.live,
    )
    print(json.dumps({"accuracy_delta": result["deltas"], "out": args.out}, indent=2))
    return 0


def _cmd_hf_jobs(args: argparse.Namespace) -> int:
    from swift_cf_approx.backends.hf_jobs import try_submit

    result = try_submit(args.config, dry_run=not args.submit)
    print(json.dumps({k: result[k] for k in result if k != "config"}, indent=2))
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """CPU/GPU convenience: distill (synthetic ok) → train → eval."""
    root = Path(args.root)
    distill_out = root / "data" / "distill" / "train.jsonl"
    train_out = root / "artifacts" / "train"
    eval_out = root / "reports"

    ns_distill = argparse.Namespace(
        prompts=str(root / "data" / "prompts" / "distill_seed.jsonl"),
        out=str(distill_out),
        limit=args.limit,
        base_url=args.base_url,
        api_key=None,
        model="default",
        require_teacher=False,
    )
    _cmd_distill(ns_distill)

    dry = True if args.backend == "cpu" else args.dry_run
    ns_train = argparse.Namespace(
        config=str(root / "configs" / "train_qlora.yaml"),
        gain_config=str(root / "configs" / "gain_proxy.yaml"),
        data=str(distill_out),
        out=str(train_out),
        backend=args.backend,
        dry_run=dry,
    )
    summary = None
    # run_train via _cmd_train prints; capture adapter from disk convention
    _cmd_train(ns_train)
    adapter = train_out / "lora_adapter"

    ns_eval = argparse.Namespace(
        config=str(root / "configs" / "eval.yaml"),
        out=str(eval_out),
        adapter=str(adapter),
        live=False,
    )
    _cmd_eval(ns_eval)
    print(json.dumps({"pipeline": "ok", "backend": args.backend, "adapter": str(adapter)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swift-cf", description="Swift-Bonsai Cold Fusion approx scaffold")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dequant", help="Unpack Swift PTQ1_0 GGUF to HF safetensors")
    d.add_argument("--gguf", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-shard-bytes", type=int, default=5_000_000_000)
    d.add_argument("--max-tensors", type=int, default=None)
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(func=_cmd_dequant)

    di = sub.add_parser("distill", help="Generate distill JSONL from DavidAU teacher or synthetic")
    di.add_argument("--prompts", required=True)
    di.add_argument("--out", required=True)
    di.add_argument("--limit", type=int, default=None)
    di.add_argument("--base-url", default=None, help="OpenAI-compatible teacher base URL")
    di.add_argument("--api-key", default=None)
    di.add_argument("--model", default="default")
    di.add_argument("--require-teacher", action="store_true")
    di.set_defaults(func=_cmd_distill)

    t = sub.add_parser("train", help="QLoRA / dry-run train with GAIN proxy")
    t.add_argument("--config", default="configs/train_qlora.yaml")
    t.add_argument("--gain-config", default="configs/gain_proxy.yaml")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="artifacts/train")
    t.add_argument("--backend", choices=["gpu", "cpu", "hf-jobs"], default="cpu")
    t.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("eval", help="Compare baseline vs tuned scores")
    e.add_argument("--config", default="configs/eval.yaml")
    e.add_argument("--out", default="reports")
    e.add_argument("--adapter", default=None)
    e.add_argument("--live", action="store_true")
    e.set_defaults(func=_cmd_eval)

    h = sub.add_parser("hf-jobs", help="Write / submit HF Jobs spec")
    h.add_argument("--config", default="configs/hf_jobs.yaml")
    h.add_argument("--submit", action="store_true")
    h.set_defaults(func=_cmd_hf_jobs)

    pipe = sub.add_parser("pipeline", help="distill → train → eval for a backend")
    pipe.add_argument("--backend", choices=["gpu", "cpu"], default="cpu")
    pipe.add_argument("--root", default=".")
    pipe.add_argument("--limit", type=int, default=8)
    pipe.add_argument("--base-url", default=None)
    pipe.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    pipe.set_defaults(func=_cmd_pipeline)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
