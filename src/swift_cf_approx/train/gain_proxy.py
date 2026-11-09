"""Open GAIN proxy: per-sample loss scaling (NOT DavidAU proprietary GAIN)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GainProxyConfig:
    enabled: bool = True
    # Scale down loss for already-short thinking (encourage keeping it short)
    short_thinking_boost: float = 1.15
    long_thinking_penalty: float = 0.85
    short_threshold_tokens: int = 64
    long_threshold_tokens: int = 512
    # Clamp final weight
    min_weight: float = 0.5
    max_weight: float = 1.5


def compute_sample_weight(thinking_tokens_est: int, cfg: GainProxyConfig) -> float:
    if not cfg.enabled:
        return 1.0
    w = 1.0
    if thinking_tokens_est <= cfg.short_threshold_tokens:
        w *= cfg.short_thinking_boost
    elif thinking_tokens_est >= cfg.long_threshold_tokens:
        w *= cfg.long_thinking_penalty
    return float(max(cfg.min_weight, min(cfg.max_weight, w)))
