from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guard
    pd = None  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guard
    yaml = None  # type: ignore

from gguf_speed import formula, hf, kv, local_db, table

DEFAULT_CONTEXTS = [4096, 8192, 16384, 24576, 32768]
DEFAULT_DENSE_CALIBRATION = formula.DenseCalibration()
DEFAULT_MOE_CALIBRATION = formula.MoECalibration()
DEFAULT_MODEL_TYPE = formula.ModelType.DENSE
PRESETS_PATH = Path("app/presets.yaml")
DEFAULT_MOE_RATIO = formula.DEFAULT_MOE_RATIO


def load_presets(path: Path = PRESETS_PATH) -> Mapping[str, Mapping[str, object]]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("presets", {})


def _to_float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _extract_dense_calibration(
    preset: Mapping[str, object], base: formula.DenseCalibration
) -> formula.DenseCalibration:
    section = preset.get("dense")
    if isinstance(section, Mapping):
        return formula.DenseCalibration(
            C=_to_float(section.get("C"), base.C),
            beta=_to_float(section.get("beta"), base.beta),
        )
    return formula.DenseCalibration(
        C=_to_float(preset.get("C"), base.C),
        beta=_to_float(preset.get("beta"), base.beta),
    )


def _extract_moe_calibration(
    preset: Mapping[str, object], base: formula.MoECalibration
) -> formula.MoECalibration:
    section = preset.get("moe")
    if isinstance(section, Mapping):
        return formula.MoECalibration(
            coeff_total=_to_float(section.get("coeff_total"), base.coeff_total),
            exp_total=_to_float(section.get("exp_total"), base.exp_total),
            coeff_active=_to_float(section.get("coeff_active"), base.coeff_active),
            exp_active=_to_float(section.get("exp_active"), base.exp_active),
        )
    return formula.MoECalibration(
        coeff_total=_to_float(preset.get("moe_coeff_total"), base.coeff_total),
        exp_total=_to_float(preset.get("moe_exp_total"), base.exp_total),
        coeff_active=_to_float(preset.get("moe_coeff_active"), base.coeff_active),
        exp_active=_to_float(preset.get("moe_exp_active"), base.exp_active),
    )


def _extract_moe_ratio(preset: Mapping[str, object], default: float) -> float:
    section = preset.get("moe")
    if isinstance(section, Mapping):
        if "ratio" in section:
            return _to_float(section.get("ratio"), default)
        if "moe_ratio" in section:
            return _to_float(section.get("moe_ratio"), default)
    if "moe_ratio" in preset:
        return _to_float(preset.get("moe_ratio"), default)
    return default


def gather_kv_config(
    repo_id: str,
    *,
    token: str | None,
    overrides: Mapping[str, int | None] | None,
    dtype_bytes: int,
    max_depth: int,
):
    parts: list[tuple[Mapping[str, int], str]] = []
    for text, source in hf.iter_potential_kv_texts(repo_id, token=token, max_depth=max_depth):
        data = kv.parse_from_text(text)
        if data:
            parts.append((data, source))
    entry = local_db.lookup(repo_id)
    if entry and entry.kv_fields:
        parts.append((entry.kv_fields, "local-db"))
    config, mode = kv.consolidate_configs(parts, overrides=overrides, dtype_bytes=dtype_bytes)
    return config, mode


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate GGUF inference speed and memory usage.")
    parser.add_argument("repo", help="Hugging Face repo ID or URL")
    parser.add_argument("--contexts", nargs="*", type=int, default=DEFAULT_CONTEXTS, help="Context lengths in tokens")
    parser.add_argument(
        "--C",
        dest="C",
        type=float,
        default=DEFAULT_DENSE_CALIBRATION.C,
        help="Dense calibration constant C",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_DENSE_CALIBRATION.beta,
        help="Dense calibration constant beta",
    )
    parser.add_argument(
        "--model-type",
        choices=[item.value for item in formula.ModelType],
        default=DEFAULT_MODEL_TYPE.value,
        help="Model type for the speed approximation (dense or moe)",
    )
    parser.add_argument(
        "--moe-ratio",
        type=float,
        default=None,
        help="Active parameter ratio for MoE models (0-1)",
    )
    parser.add_argument(
        "--moe-total-coeff",
        type=float,
        default=DEFAULT_MOE_CALIBRATION.coeff_total,
        help="Coefficient for the total size term in the MoE formula",
    )
    parser.add_argument(
        "--moe-total-exp",
        type=float,
        default=DEFAULT_MOE_CALIBRATION.exp_total,
        help="Exponent for the total size term in the MoE formula",
    )
    parser.add_argument(
        "--moe-active-coeff",
        type=float,
        default=DEFAULT_MOE_CALIBRATION.coeff_active,
        help="Coefficient for the active size term in the MoE formula",
    )
    parser.add_argument(
        "--moe-active-exp",
        type=float,
        default=DEFAULT_MOE_CALIBRATION.exp_active,
        help="Exponent for the active size term in the MoE formula",
    )
    parser.add_argument("--preset", help="Name of a preset defined in app/presets.yaml")
    parser.add_argument("--mem-gib", type=float, default=None, help="Memory budget in GiB")
    parser.add_argument("--overhead-gib", type=float, default=4.0, help="Additional overhead in GiB")
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--hf-token", help="Hugging Face access token")
    parser.add_argument("--csv", help="Path to write CSV output")
    parser.add_argument("--max-depth", type=int, default=1, help="Depth of base-model recursion when collecting KV fields")
    return parser


def ensure_dependencies() -> None:
    missing: list[str] = []
    if pd is None:
        missing.append("pandas")
    if yaml is None:
        missing.append("pyyaml")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required dependencies: {joined}. Install them with 'pip install -r requirements.txt'."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    ensure_dependencies()

    repo_id = hf.normalize_repo_id(args.repo)

    dense_calibration = formula.DenseCalibration(C=args.C, beta=args.beta)
    moe_calibration = formula.MoECalibration(
        coeff_total=args.moe_total_coeff,
        exp_total=args.moe_total_exp,
        coeff_active=args.moe_active_coeff,
        exp_active=args.moe_active_exp,
    )
    model_type = formula.ModelType(args.model_type)
    moe_ratio = args.moe_ratio
    presets = load_presets()
    if args.preset:
        preset = presets.get(args.preset)
        if not preset:
            parser.error(f"Unknown preset: {args.preset}")
        dense_calibration = _extract_dense_calibration(preset, dense_calibration)
        moe_calibration = _extract_moe_calibration(preset, moe_calibration)
        if moe_ratio is None:
            moe_ratio = _extract_moe_ratio(preset, DEFAULT_MOE_RATIO)
        if "model_type" in preset and args.model_type == parser.get_default("model_type"):
            try:
                model_type = formula.ModelType(str(preset["model_type"]))
            except ValueError:
                pass
        if "mem_gib" in preset and args.mem_gib is None:
            args.mem_gib = float(preset["mem_gib"])
        if "overhead_gib" in preset and args.overhead_gib == parser.get_default("overhead_gib"):
            args.overhead_gib = float(preset["overhead_gib"])

    if model_type is formula.ModelType.MOE and moe_ratio is None:
        moe_ratio = DEFAULT_MOE_RATIO

    contexts = args.contexts or DEFAULT_CONTEXTS
    try:
        ggufs = hf.list_gguf(repo_id, token=args.hf_token)
    except hf.HFError as exc:
        print(f"Failed to fetch GGUF metadata from Hugging Face: {exc}")
        ggufs = []

    if not ggufs:
        ggufs = local_db.list_gguf(repo_id)
        if ggufs:
            print("Using local database for GGUF metadata.")
        else:
            print("No GGUF files found.")
            return 1

    overrides = {
        "n_layers": args.n_layers,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "hidden_size": args.hidden_size,
        "n_heads": args.n_heads,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config, kv_mode = gather_kv_config(
        repo_id,
        token=args.hf_token,
        overrides=overrides or None,
        dtype_bytes=args.dtype_bytes,
        max_depth=args.max_depth,
    )

    if config is None and overrides:
        required = {"n_layers", "n_kv_heads", "head_dim"}
        if not required.issubset(overrides):
            missing = ", ".join(sorted(required - overrides.keys()))
            parser.error(f"Override requires values for: {missing}")
        config = kv.KVConfig(
            n_layers=int(overrides["n_layers"]),
            n_kv_heads=int(overrides["n_kv_heads"]),
            head_dim=int(overrides["head_dim"]),
            dtype_bytes=int(args.dtype_bytes),
            source="override",
        )
        kv_mode = "override"

    if config:
        print(f"KV source: {config.source or kv_mode}")
        print(
            f"KV per token: {config.kv_per_token_gib:.6f} GiB | n_layers={config.n_layers} "
            f"n_kv={config.n_kv_heads} head_dim={config.head_dim}"
        )
    else:
        print("KV parameters not found; OOM estimation disabled.")

    print(f"Model type: {model_type.value}")
    if model_type is formula.ModelType.MOE:
        ratio_display = moe_ratio if moe_ratio is not None else DEFAULT_MOE_RATIO
        print(
            "MoE calibration → "
            f"coeff_total={moe_calibration.coeff_total:.4f} exp_total={moe_calibration.exp_total:.4f} "
            f"coeff_active={moe_calibration.coeff_active:.4f} exp_active={moe_calibration.exp_active:.4f} "
            f"ratio={ratio_display:.4f}"
        )
    else:
        print(
            "Dense calibration → "
            f"C={dense_calibration.C:.4f} beta={dense_calibration.beta:.6g}"
        )

    df = table.build(
        repo_id,
        ggufs,
        contexts,
        dense_calibration,
        moe_calibration,
        model_type,
        config,
        mem_gib=args.mem_gib,
        overhead_gib=args.overhead_gib,
        moe_ratio=moe_ratio,
    )

    assert pd is not None
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    if args.csv:
        table.to_csv(df, args.csv)
        print(f"CSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
