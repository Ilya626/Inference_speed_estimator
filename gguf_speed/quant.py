"""Quantization helpers for GGUF metadata."""
from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

_NumberPattern = re.compile(r"(\d+(?:\.\d+)?)")


def infer_quant(rfilename: str) -> str:
    """Return the quantization label extracted from a GGUF filename."""

    name = Path(rfilename or "").name
    if not name.lower().endswith(".gguf"):
        return "unknown"
    stem = name[:-5]
    if "-" in stem:
        quant = stem.split("-")[-1]
    else:
        quant = stem
    return quant


def nominal_bpw(label: str | None) -> float | None:
    """Best-effort conversion of a quant label into nominal bits-per-weight."""

    if not label:
        return None
    text = str(label).strip().lower()
    if not text:
        return None
    # Common floating-point formats.
    if text in {"f16", "fp16", "bf16"}:
        return 16.0
    if text in {"f32", "fp32"}:
        return 32.0
    if text in {"f64", "fp64"}:
        return 64.0
    match = _NumberPattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def estimate_param_count(ggufs: Sequence[Mapping[str, object]]) -> float | None:
    """Estimate the number of model parameters from GGUF artefacts."""

    candidates: list[float] = []
    for entry in ggufs:
        size_value = entry.get("size")
        if size_value is None:
            continue
        try:
            size_bytes = int(size_value)
        except (TypeError, ValueError):
            continue
        if size_bytes <= 0:
            continue
        quant_label = entry.get("quant")
        if isinstance(quant_label, str) and quant_label:
            quant = quant_label
        else:
            quant = infer_quant(str(entry.get("rfilename", "")))
        nominal = nominal_bpw(quant)
        if nominal is None or nominal <= 0:
            continue
        params = (size_bytes * 8.0) / nominal
        if params > 0:
            candidates.append(params)
    if not candidates:
        return None
    return median(candidates)


def bpw_from_size(size_bytes: int, param_count: float) -> float:
    """Compute effective bits-per-weight from a file size and parameter count."""

    if param_count <= 0:
        raise ValueError("Parameter count must be positive.")
    if size_bytes < 0:
        raise ValueError("File size cannot be negative.")
    return (float(size_bytes) * 8.0) / float(param_count)


__all__ = [
    "infer_quant",
    "nominal_bpw",
    "estimate_param_count",
    "bpw_from_size",
]
