"""Utilities for extracting KV-cache parameters and estimating memory usage."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from . import formula

# Regular expressions that capture integers after simple separators.
_FIELD_PATTERNS = {
    "n_layers": re.compile(r"num_hidden_layers\s*[:=]\s*(\d+)", re.IGNORECASE),
    "n_kv_heads": re.compile(r"num_key_value_heads\s*[:=]\s*(\d+)", re.IGNORECASE),
    "n_heads": re.compile(r"num_attention_heads\s*[:=]\s*(\d+)", re.IGNORECASE),
    "hidden_size": re.compile(r"hidden_size\s*[:=]\s*(\d+)", re.IGNORECASE),
    "head_dim": re.compile(r"head_dim\s*[:=]\s*(\d+)", re.IGNORECASE),
}


@dataclass
class KVConfig:
    """Normalized KV-cache configuration."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2
    source: str | None = None
    hidden_size: int | None = None
    n_heads: int | None = None

    @property
    def kv_per_token_gib(self) -> float:
        return kv_per_token_gib(
            self.n_layers,
            self.n_kv_heads,
            self.head_dim,
            dtype_bytes=self.dtype_bytes,
        )


def parse_from_text(text: str) -> MutableMapping[str, int]:
    """Extract KV related integers from a blob of text."""

    text = text or ""
    data: MutableMapping[str, int] = {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, Mapping):
        key_map = {
            "num_hidden_layers": "n_layers",
            "n_layer": "n_layers",
            "num_key_value_heads": "n_kv_heads",
            "num_kv_heads": "n_kv_heads",
            "num_attention_heads": "n_heads",
            "hidden_size": "hidden_size",
            "head_dim": "head_dim",
        }
        for src, dst in key_map.items():
            value = parsed.get(src)
            if isinstance(value, (int, float)):
                data[dst] = int(value)

    for key, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(text)
        if match:
            data[key] = int(match.group(1))

    return data


def derive_head_dim(hidden_size: int, n_heads: int) -> int:
    """Derive head dimension from hidden size and attention heads."""

    if n_heads <= 0:
        raise ValueError("Number of attention heads must be positive.")
    if hidden_size % n_heads != 0:
        raise ValueError("hidden_size is not divisible by num_attention_heads.")
    return hidden_size // n_heads


def kv_per_token_gib(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    *,
    dtype_bytes: int = 2,
) -> float:
    """Return the KV cache footprint per token in GiB."""

    if min(n_layers, n_kv_heads, head_dim, dtype_bytes) <= 0:
        raise ValueError("KV parameters must be positive integers.")
    bytes_per_token = 2 * dtype_bytes * n_layers * n_kv_heads * head_dim
    return bytes_per_token / float(formula.GIB)


def kv_total_gib(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    *,
    dtype_bytes: int = 2,
) -> float:
    """Total KV cache usage in GiB for the given context length."""

    return kv_per_token_gib(
        n_layers,
        n_kv_heads,
        head_dim,
        dtype_bytes=dtype_bytes,
    ) * context_tokens


def consolidate_configs(
    parts: Sequence[tuple[Mapping[str, int], str]],
    *,
    overrides: Mapping[str, int | None] | None = None,
    dtype_bytes: int = 2,
) -> tuple[KVConfig | None, str]:
    """Merge multiple partial configs and apply user overrides."""

    merged: dict[str, int] = {}
    sources: list[str] = []
    for data, source in parts:
        if not data:
            continue
        sources.append(source)
        for key, value in data.items():
            if value is None:
                continue
            merged.setdefault(key, int(value))

    if overrides:
        sources.append("override")
        for key, value in overrides.items():
            if value is None:
                continue
            merged[key] = int(value)

    required_keys = {"n_layers", "n_kv_heads"}
    if not required_keys.issubset(merged):
        return None, "unknown"

    head_dim = merged.get("head_dim")
    hidden_size = merged.get("hidden_size")
    n_heads = merged.get("n_heads")
    if head_dim is None and hidden_size is not None and n_heads is not None:
        head_dim = derive_head_dim(hidden_size, n_heads)
    if head_dim is None:
        return None, "unknown"

    config = KVConfig(
        n_layers=int(merged["n_layers"]),
        n_kv_heads=int(merged["n_kv_heads"]),
        head_dim=int(head_dim),
        dtype_bytes=int(dtype_bytes),
        hidden_size=int(hidden_size) if hidden_size is not None else None,
        n_heads=int(n_heads) if n_heads is not None else None,
        source=" > ".join(sources) if sources else None,
    )
    mode = "override" if overrides else "auto"
    return config, mode
