"""Assemble tabular outputs for the GGUF speed estimator."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from . import formula, kv, quant


def build(
    repo_id: str,
    ggufs: Sequence[Mapping[str, object]],
    contexts: Sequence[int],
    C: float,
    beta: float,
    kv_config: kv.KVConfig | None,
    *,
    mem_gib: float | None = None,
    overhead_gib: float = 0.0,
) -> pd.DataFrame:
    """Produce a pandas DataFrame with the computed statistics."""

    rows: list[dict[str, object]] = []
    kv_per_token = kv_config.kv_per_token_gib if kv_config else None
    param_estimate = quant.estimate_param_count(ggufs)
    params_billions = (
        round(param_estimate / 1e9, 4) if param_estimate and param_estimate > 0 else None
    )
    for file_info in ggufs:
        name = str(file_info.get("rfilename"))
        size_bytes = int(file_info.get("size", 0))
        size_gib = formula.bytes_to_gib(size_bytes)
        raw_quant = file_info.get("quant")
        quant_label = (
            str(raw_quant)
            if isinstance(raw_quant, str) and raw_quant.strip()
            else quant.infer_quant(name)
        )
        bpw = None
        if param_estimate and param_estimate > 0 and size_bytes > 0:
            bpw = round(quant.bpw_from_size(size_bytes, param_estimate), 4)

        for context in contexts:
            context_tokens = int(context)
            speed_est = formula.speed(size_gib, context_tokens, C, beta)

            kv_gib = None
            est_mem = None
            oom_state = "unknown"
            n_layers = None
            n_kv_heads = None
            head_dim = None
            kv_mode = "unknown"

            if kv_config:
                kv_mode = kv_config.source or "auto"
                kv_gib = kv_per_token * context_tokens if kv_per_token else None
                est_mem = (
                    formula.mem_need(size_gib, kv_gib, overhead_gib)
                    if kv_gib is not None
                    else None
                )
                n_layers = kv_config.n_layers
                n_kv_heads = kv_config.n_kv_heads
                head_dim = kv_config.head_dim
                if est_mem is not None and mem_gib is not None:
                    oom_state = "OK" if est_mem <= mem_gib else "OOM"
                else:
                    oom_state = "unknown"

            rows.append(
                {
                    "repo": repo_id,
                    "file": name,
                    "quant": quant_label,
                    "bpw": bpw,
                    "params_b": params_billions,
                    "size_gib": round(size_gib, 4),
                    "context_tokens": context_tokens,
                    "pred_speed_toks_per_s": round(speed_est, 3),
                    "kv_mode": kv_mode,
                    "n_layers": n_layers,
                    "n_kv": n_kv_heads,
                    "head_dim": head_dim,
                    "kv_gib": round(kv_gib, 4) if kv_gib is not None else None,
                    "est_mem_gib": round(est_mem, 4) if est_mem is not None else None,
                    "oom": oom_state,
                }
            )

    df = pd.DataFrame(rows)
    df = df[
        [
            "repo",
            "file",
            "quant",
            "bpw",
            "params_b",
            "size_gib",
            "context_tokens",
            "pred_speed_toks_per_s",
            "kv_mode",
            "n_layers",
            "n_kv",
            "head_dim",
            "kv_gib",
            "est_mem_gib",
            "oom",
        ]
    ]
    return df


def to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV and return the path."""

    path = Path(path)
    df.to_csv(path, index=False)
    return path
