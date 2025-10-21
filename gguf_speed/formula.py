"""Core formulas used by the GGUF speed estimator."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

# 1 GiB in bytes.
GIB = 1024 ** 3


def bytes_to_gib(size_bytes: int | float) -> float:
    """Convert a byte count to gibibytes (GiB)."""
    return float(size_bytes) / GIB


class ModelType(Enum):
    """Supported architecture categories for speed estimation."""

    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True)
class DenseCalibration:
    """Calibration constants used by the dense speed model."""

    C: float = 206.0
    beta: float = 0.000512

    def speed(self, size_gib: float, context_tokens: int) -> float:
        """Convenience wrapper around :func:`dense_speed`."""
        return dense_speed(size_gib, context_tokens, self.C, self.beta)


@dataclass(frozen=True)
class MoECalibration:
    """Calibration constants for the MoE approximation."""

    coeff_total: float = 2593.21
    exp_total: float = -1.51
    coeff_active: float = 44.9
    exp_active: float = -0.47

    def speed(self, total_gib: float, active_gib: float) -> float:
        """Evaluate :func:`moe_speed` with stored calibration constants."""

        return moe_speed(
            total_gib,
            active_gib,
            self.coeff_total,
            self.exp_total,
            self.coeff_active,
            self.exp_active,
        )


Calibration = DenseCalibration
DEFAULT_MOE_RATIO = 0.113


def dense_speed(size_gib: float, context_tokens: int, C: float, beta: float) -> float:
    """Estimate tokens/s for a dense GGUF file at a given context length."""

    denominator = size_gib + beta * float(context_tokens)
    if denominator <= 0:
        raise ValueError("Denominator of dense speed formula must be positive.")
    return C / denominator


def speed(size_gib: float, context_tokens: int, C: float, beta: float) -> float:
    """Backward compatible wrapper around :func:`dense_speed`."""

    return dense_speed(size_gib, context_tokens, C, beta)


def moe_speed(
    total_gib: float,
    active_gib: float,
    coeff_total: float,
    exp_total: float,
    coeff_active: float,
    exp_active: float,
) -> float:
    """Approximate tokens/s for a MoE model using total and active footprint."""

    if total_gib <= 0:
        raise ValueError("Total model size must be positive for the MoE formula.")
    if active_gib <= 0:
        raise ValueError("Active model size must be positive for the MoE formula.")
    total_term = coeff_total * pow(total_gib, exp_total)
    active_term = coeff_active * pow(active_gib, exp_active)
    return total_term + active_term


def moe_speed_from_ratio(total_gib: float, ratio: float, calibration: MoECalibration) -> float:
    """Convenience helper that derives the active size from ``ratio``."""

    if ratio <= 0 or ratio > 1:
        raise ValueError("MoE active ratio must be within (0, 1].")
    active_gib = total_gib * ratio
    return calibration.speed(total_gib, active_gib)


def mem_need(size_gib: float, kv_gib: float, overhead_gib: float = 0.0) -> float:
    """Estimate the total memory requirement in GiB."""

    return size_gib + kv_gib + overhead_gib


def fit_dense_calibration(
    samples: Iterable[Tuple[float, int, float]],
    *,
    fix_beta: float | None = None,
) -> DenseCalibration:
    """Fit ``C`` and ``beta`` for the dense model from empirical measurements."""

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
        return DenseCalibration(C=C_est, beta=float(fix_beta))

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
    return DenseCalibration(C=float(C_est), beta=float(beta_est))


fit_calibration = fit_dense_calibration
