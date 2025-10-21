"""Core formulas used by the GGUF speed estimator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

# 1 GiB in bytes.
GIB = 1024 ** 3


def bytes_to_gib(size_bytes: int | float) -> float:
    """Convert a byte count to gibibytes (GiB)."""
    return float(size_bytes) / GIB


@dataclass(frozen=True)
class Calibration:
    """Calibration constants used by the speed model."""

    C: float = 206.0
    beta: float = 0.000512

    def speed(self, size_gib: float, context_tokens: int) -> float:
        """Convenience wrapper around :func:`speed`."""
        return speed(size_gib, context_tokens, self.C, self.beta)


def speed(size_gib: float, context_tokens: int, C: float, beta: float) -> float:
    """Estimate tokens/s for a GGUF file at a given context length."""

    denominator = size_gib + beta * float(context_tokens)
    if denominator <= 0:
        raise ValueError("Denominator of speed formula must be positive.")
    return C / denominator


def mem_need(size_gib: float, kv_gib: float, overhead_gib: float = 0.0) -> float:
    """Estimate the total memory requirement in GiB."""

    return size_gib + kv_gib + overhead_gib


def fit_calibration(
    samples: Iterable[Tuple[float, int, float]],
    *,
    fix_beta: float | None = None,
) -> Calibration:
    """Fit ``C`` and ``beta`` from empirical measurements."""

    samples = list(samples)
    if not samples:
        raise ValueError("At least one sample is required to fit calibration.")

    if fix_beta is not None:
        total = 0.0
        for size_gib, context_tokens, measured_speed in samples:
            if measured_speed <= 0:
                raise ValueError("Measured speed must be positive.")
            total += measured_speed * (size_gib + fix_beta * context_tokens)
        C_est = total / len(samples)
        return Calibration(C=C_est, beta=float(fix_beta))

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("numpy is required to fit calibration without fix_beta") from exc

    sizes: list[float] = []
    contexts: list[float] = []
    inv_speeds: list[float] = []
    for size_gib, context_tokens, measured_speed in samples:
        if measured_speed <= 0:
            raise ValueError("Measured speed must be positive.")
        sizes.append(float(size_gib))
        contexts.append(float(context_tokens))
        inv_speeds.append(1.0 / float(measured_speed))

    X = np.column_stack([sizes, contexts])
    y = np.array(inv_speeds)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    inv_C, beta_over_C = coeffs
    if inv_C <= 0:
        raise ValueError("Calibration fit produced invalid C.")
    C_est = 1.0 / inv_C
    beta_est = beta_over_C * C_est
    return Calibration(C=float(C_est), beta=float(beta_est))
