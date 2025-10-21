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
DEFAULT_CALIBRATION = formula.Calibration()
PRESETS_PATH = Path("app/presets.yaml")


def load_presets(path: Path = PRESETS_PATH) -> Mapping[str, Mapping[str, float]]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("presets", {})


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
    parser.add_argument("--C", dest="C", type=float, default=DEFAULT_CALIBRATION.C, help="Calibration constant C")
    parser.add_argument("--beta", type=float, default=DEFAULT_CALIBRATION.beta, help="Calibration constant beta")
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

    calibration = formula.Calibration(C=args.C, beta=args.beta)
    presets = load_presets()
    if args.preset:
        preset = presets.get(args.preset)
        if not preset:
            parser.error(f"Unknown preset: {args.preset}")
        calibration = formula.Calibration(
            C=float(preset.get("C", calibration.C)),
            beta=float(preset.get("beta", calibration.beta)),
        )
        if "mem_gib" in preset and args.mem_gib is None:
            args.mem_gib = float(preset["mem_gib"])
        if "overhead_gib" in preset and args.overhead_gib == parser.get_default("overhead_gib"):
            args.overhead_gib = float(preset["overhead_gib"])

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

    df = table.build(
        repo_id,
        ggufs,
        contexts,
        calibration.C,
        calibration.beta,
        config,
        mem_gib=args.mem_gib,
        overhead_gib=args.overhead_gib,
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
