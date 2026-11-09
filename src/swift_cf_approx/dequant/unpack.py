"""Stream-dequantize Swift-Bonsai PTQ1_0 GGUF into HF-style safetensors shards."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from .gguf_reader import GgufTensorInfo, open_gguf, read_tensor_bytes
from .ptq1 import GGML_TYPE_PTQ1_0, PTQ1_0_BLOCK_BYTES, QK_PTQ1_0, dequant_ptq1_0, ptq1_0_bytes

logger = logging.getLogger(__name__)

# Common llama.cpp → HF name fragments for Qwen-family GGUFs.
# Full mapping is best-effort; remaining tensors keep GGUF names under `gguf.`.
_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def _hf_name(gguf_name: str) -> str:
    if gguf_name in _NAME_MAP:
        return _NAME_MAP[gguf_name]
    # blk.N.attn_q.weight → model.layers.N.self_attn.q_proj.weight (approx)
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            return f"gguf.{gguf_name}"
        rest = ".".join(parts[2:])
        layer = f"model.layers.{idx}"
        mapping = {
            "attn_q.weight": f"{layer}.self_attn.q_proj.weight",
            "attn_k.weight": f"{layer}.self_attn.k_proj.weight",
            "attn_v.weight": f"{layer}.self_attn.v_proj.weight",
            "attn_output.weight": f"{layer}.self_attn.o_proj.weight",
            "attn_norm.weight": f"{layer}.input_layernorm.weight",
            "ffn_norm.weight": f"{layer}.post_attention_layernorm.weight",
            "ffn_gate.weight": f"{layer}.mlp.gate_proj.weight",
            "ffn_up.weight": f"{layer}.mlp.up_proj.weight",
            "ffn_down.weight": f"{layer}.mlp.down_proj.weight",
        }
        return mapping.get(rest, f"gguf.{gguf_name}")
    return f"gguf.{gguf_name}"


def _dtype_nbytes_estimate(tensor: GgufTensorInfo) -> int | None:
    if tensor.ggml_type == GGML_TYPE_PTQ1_0:
        return ptq1_0_bytes(tensor.n_elements)
    # F32 / F16 / BF16 pass-through sizes
    type_sizes = {0: 4, 1: 2, 30: 2}  # F32, F16, BF16
    el = type_sizes.get(tensor.ggml_type)
    if el is None:
        return None
    return tensor.n_elements * el


def _decode_tensor(gguf_path_bytes: bytes, tensor: GgufTensorInfo) -> np.ndarray:
    shape = tuple(reversed(tensor.dims)) if len(tensor.dims) > 1 else tensor.dims
    if tensor.ggml_type == GGML_TYPE_PTQ1_0:
        nbytes = ptq1_0_bytes(tensor.n_elements)
        if nbytes is None:
            raise ValueError(f"{tensor.name}: n_elements not multiple of {QK_PTQ1_0}")
        # Decode straight to float16 in block chunks to keep peak RAM low (~15GB hosts).
        arr = dequant_ptq1_0(
            gguf_path_bytes,
            tensor.n_elements,
            dtype=np.float16,
            block_chunk=32768,
        )
        return arr.reshape(shape)

    type_map = {0: np.float32, 1: np.float16, 30: np.float16}
    dt = type_map.get(tensor.ggml_type)
    if dt is None:
        raise ValueError(f"{tensor.name}: unsupported ggml type {tensor.ggml_type}")
    arr = np.frombuffer(gguf_path_bytes, dtype=dt).copy()
    return arr.reshape(shape).astype(np.float16, copy=False)


def _write_config(out_dir: Path, meta: dict[str, Any], n_tensors: int) -> None:
    arch = meta.get("general.architecture", "qwen35")
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "gguf_architecture": arch,
        "torch_dtype": "float16",
        "transformers_version": "4.46.0",
        "swift_cf_approx": {
            "source": "ukisai/Swift-Bonsai-2-GGUF",
            "packing": "PTQ1_0",
            "note": (
                "Dequantized ternary weights for QLoRA experiments. "
                "Hadamard/hybrid runtime may make HF inference non-faithful; "
                "prefer original GGUF for baseline scoring."
            ),
            "n_tensors_unpacked": n_tensors,
        },
    }
    # Prefer numeric metadata when present
    for src, dst in (
        ("qwen35.embedding_length", "hidden_size"),
        ("qwen35.block_count", "num_hidden_layers"),
        ("qwen35.attention.head_count", "num_attention_heads"),
        ("qwen35.attention.head_count_kv", "num_key_value_heads"),
        ("qwen35.feed_forward_length", "intermediate_size"),
        ("qwen35.context_length", "max_position_embeddings"),
        ("qwen35.vocab_size", "vocab_size"),
    ):
        if src in meta:
            cfg[dst] = int(meta[src])
    # Fallback architecture keys used by some packs
    for src, dst in (
        ("qwen3.embedding_length", "hidden_size"),
        ("qwen3.block_count", "num_hidden_layers"),
    ):
        if dst not in cfg and src in meta:
            cfg[dst] = int(meta[src])
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def unpack_swift_ptq1_to_hf(
    gguf_path: str | Path,
    out_dir: str | Path,
    *,
    max_shard_bytes: int = 5_000_000_000,
    max_tensors: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Dequantize PTQ1_0 (and pass through float) tensors to safetensors shards."""
    gguf_path = Path(gguf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf = open_gguf(gguf_path)
    tensors = list(gguf.tensors)
    if max_tensors is not None:
        tensors = tensors[:max_tensors]

    report: dict[str, Any] = {
        "source": str(gguf_path),
        "n_tensors": len(tensors),
        "architecture": gguf.metadata.get("general.architecture"),
        "shards": [],
        "skipped": [],
        "dry_run": dry_run,
    }

    if dry_run:
        for t in tensors:
            nbytes = _dtype_nbytes_estimate(t)
            report["shards"].append(
                {
                    "name": t.name,
                    "hf_name": _hf_name(t.name),
                    "ggml_type": t.ggml_type,
                    "dims": list(t.dims),
                    "packed_bytes": nbytes,
                }
            )
        _write_config(out_dir, gguf.metadata, len(tensors))
        (out_dir / "unpack_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    shard: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}
    # Smaller shards flush sooner to release RAM on 16GB hosts.
    if max_shard_bytes > 1_500_000_000:
        max_shard_bytes = 1_500_000_000

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        path = out_dir / name
        save_file(shard, str(path))
        for k in shard:
            weight_map[k] = name
        report["shards"].append({"file": name, "n_tensors": len(shard), "approx_bytes": shard_bytes})
        shard = {}
        shard_bytes = 0
        shard_idx += 1
        # Encourage timely release of large arrays
        import gc

        gc.collect()

    for ti, t in enumerate(tensors):
        nbytes = _dtype_nbytes_estimate(t)
        if nbytes is None:
            report["skipped"].append({"name": t.name, "ggml_type": t.ggml_type, "reason": "unsupported"})
            logger.warning("Skipping %s (ggml type %s)", t.name, t.ggml_type)
            continue
        logger.info("[%d/%d] reading %s (%s bytes, type=%s)", ti + 1, len(tensors), t.name, nbytes, t.ggml_type)
        raw = read_tensor_bytes(gguf, t, nbytes)
        try:
            arr = _decode_tensor(raw, t)
        except Exception as exc:  # noqa: BLE001
            report["skipped"].append({"name": t.name, "error": str(exc)})
            logger.exception("Failed to decode %s", t.name)
            continue
        del raw
        key = _hf_name(t.name)
        est = int(arr.nbytes)
        # Large tensors get their own shard immediately.
        if est > max_shard_bytes // 2:
            flush()
            shard[key] = arr
            shard_bytes = est
            flush()
            logger.info("Unpacked large %s -> %s %s (%.2f MiB)", t.name, key, arr.shape, est / 1024 / 1024)
            continue
        if shard and shard_bytes + est > max_shard_bytes:
            flush()
        shard[key] = arr
        shard_bytes += est
        logger.info("Unpacked %s -> %s %s (%.2f MiB)", t.name, key, arr.shape, est / 1024 / 1024)

    flush()

    # Fix shard filenames with total count
    total = shard_idx
    renamed_map: dict[str, str] = {}
    for old in list(weight_map.values()):
        if "XXXXX" in old:
            new = old.replace("XXXXX", f"{total:05d}")
            src = out_dir / old
            dst = out_dir / new
            if src.exists():
                src.rename(dst)
            renamed_map[old] = new
    weight_map = {k: renamed_map.get(v, v) for k, v in weight_map.items()}
    for s in report["shards"]:
        if "XXXXX" in s["file"]:
            s["file"] = renamed_map.get(s["file"], s["file"].replace("XXXXX", f"{total:05d}"))

    index = {"metadata": {"total_size": sum(x.get("approx_bytes", 0) for x in report["shards"])}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    _write_config(out_dir, gguf.metadata, len(weight_map))
    (out_dir / "unpack_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
