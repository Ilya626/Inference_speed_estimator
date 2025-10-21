from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping

import gradio as gr
import pandas as pd
import yaml

from gguf_speed import formula, hf, kv, table

DEFAULT_CONTEXTS = [4096, 8192, 16384, 24576, 32768]
PRESETS_PATH = Path(__file__).with_name("presets.yaml")


def load_presets() -> Mapping[str, Mapping[str, float]]:
    if not PRESETS_PATH.exists():
        return {}
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("presets", {})


PRESETS = load_presets()
DEFAULT_CALIBRATION = formula.Calibration()


def parse_contexts(selected: list[str] | None, extra_text: str | None) -> list[int]:
    contexts: set[int] = set()
    for item in selected or []:
        try:
            contexts.add(int(item))
        except (TypeError, ValueError):
            continue
    if extra_text:
        for chunk in extra_text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                contexts.add(int(chunk))
            except ValueError:
                raise ValueError(f"Invalid context token: {chunk}")
    return sorted(contexts) or DEFAULT_CONTEXTS


def gather_kv(repo_id: str, token: str | None, overrides: dict[str, int | None], dtype_bytes: int, max_depth: int):
    parts: list[tuple[Mapping[str, int], str]] = []
    for text, source in hf.iter_potential_kv_texts(repo_id, token=token, max_depth=max_depth):
        data = kv.parse_from_text(text)
        if data:
            parts.append((data, source))
    config, mode = kv.consolidate_configs(parts, overrides=overrides or None, dtype_bytes=dtype_bytes)
    if config is None and overrides:
        required = {"n_layers", "n_kv_heads", "head_dim"}
        if required.issubset({k for k, v in overrides.items() if v is not None}):
            config = kv.KVConfig(
                n_layers=int(overrides["n_layers"]),
                n_kv_heads=int(overrides["n_kv_heads"]),
                head_dim=int(overrides["head_dim"]),
                dtype_bytes=int(dtype_bytes),
                source="override",
            )
            mode = "override"
    return config, mode, parts


def run_estimator(
    repo_input: str,
    contexts_selected: list[str] | None,
    extra_contexts: str | None,
    C: float,
    beta: float,
    mem_gib: float | None,
    overhead_gib: float,
    n_layers: float | None,
    n_kv_heads: float | None,
    head_dim: float | None,
    hidden_size: float | None,
    n_heads: float | None,
    dtype_bytes: int,
    hf_token: str | None,
    max_depth: int,
) -> tuple[pd.DataFrame | None, str | None, str]:
    logs: list[str] = []
    if not repo_input:
        return None, None, "Please provide a Hugging Face repository URL or ID."

    try:
        repo_id = hf.normalize_repo_id(repo_input)
    except ValueError as exc:
        return None, None, f"❌ {exc}"

    try:
        contexts = parse_contexts(contexts_selected, extra_contexts)
    except ValueError as exc:
        return None, None, f"❌ {exc}"

    try:
        ggufs = hf.list_gguf(repo_id, token=hf_token)
    except Exception as exc:  # pragma: no cover - UI layer
        return None, None, f"❌ Failed to list GGUF files: {exc}"

    if not ggufs:
        return None, None, "⚠️ No GGUF files found in the repository."

    overrides = {
        "n_layers": int(n_layers) if n_layers is not None else None,
        "n_kv_heads": int(n_kv_heads) if n_kv_heads is not None else None,
        "head_dim": int(head_dim) if head_dim is not None else None,
        "hidden_size": int(hidden_size) if hidden_size is not None else None,
        "n_heads": int(n_heads) if n_heads is not None else None,
    }

    config, mode, parts = gather_kv(repo_id, hf_token, overrides, dtype_bytes, max_depth)
    if config:
        logs.append(f"KV source: {config.source or mode}")
        logs.append(
            f"KV per token: {config.kv_per_token_gib:.6f} GiB | n_layers={config.n_layers} "
            f"n_kv={config.n_kv_heads} head_dim={config.head_dim}"
        )
    else:
        logs.append("KV parameters not resolved; OOM will be reported as unknown.")

    calibration = formula.Calibration(C=float(C), beta=float(beta))
    df = table.build(
        repo_id,
        ggufs,
        contexts,
        calibration.C,
        calibration.beta,
        config,
        mem_gib=mem_gib,
        overhead_gib=overhead_gib,
    )

    temp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8")
    table.to_csv(df, temp.name)
    temp.close()
    logs.append(f"Processed {len(ggufs)} GGUF files across {len(contexts)} context lengths.")
    return df, temp.name, "\n".join(logs)


def apply_preset(name: str):
    preset = PRESETS.get(name) or {}
    updates = []
    for key in ("C", "beta", "mem_gib", "overhead_gib"):
        if key in preset:
            updates.append(gr.update(value=float(preset[key])))
        else:
            updates.append(gr.update())
    return updates


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GGUF Speed Estimator") as demo:
        gr.Markdown("""# GGUF Speed Estimator\nProvide a Hugging Face repository to predict token generation speed and memory usage.""")

        with gr.Row():
            repo_input = gr.Textbox(label="Hugging Face repo or URL", placeholder="https://huggingface.co/owner/repo", lines=1)
            preset = gr.Dropdown(sorted(PRESETS.keys()), label="Preset", value="strix-halo" if "strix-halo" in PRESETS else None)

        with gr.Row():
            contexts = gr.CheckboxGroup(
                [str(v) for v in DEFAULT_CONTEXTS],
                value=[str(v) for v in DEFAULT_CONTEXTS],
                label="Contexts (tokens)",
            )
            extra_contexts = gr.Textbox(label="Extra contexts", placeholder="Comma separated e.g. 12288, 40960")

        with gr.Row():
            C = gr.Number(label="C", value=DEFAULT_CALIBRATION.C)
            beta = gr.Number(label="β", value=DEFAULT_CALIBRATION.beta)
            dtype_bytes = gr.Dropdown([1, 2, 4], value=2, label="KV dtype bytes")
            max_depth = gr.Slider(0, 3, value=1, step=1, label="Base-model depth")

        with gr.Row():
            mem_gib = gr.Number(label="Memory budget (GiB)", value=64)
            overhead_gib = gr.Number(label="Overhead (GiB)", value=4)
            hf_token = gr.Textbox(label="HF token", type="password")

        with gr.Accordion("KV overrides", open=False):
            n_layers = gr.Number(label="n_layers", precision=0)
            n_kv_heads = gr.Number(label="n_kv_heads", precision=0)
            head_dim = gr.Number(label="head_dim", precision=0)
            hidden_size = gr.Number(label="hidden_size", precision=0)
            n_heads = gr.Number(label="n_heads", precision=0)

        run_button = gr.Button("Fetch & Compute", variant="primary")

        result_table = gr.Dataframe(label="Predictions", interactive=False)
        csv_file = gr.File(label="Download CSV")
        log_output = gr.Markdown()

        run_button.click(
            run_estimator,
            inputs=[
                repo_input,
                contexts,
                extra_contexts,
                C,
                beta,
                mem_gib,
                overhead_gib,
                n_layers,
                n_kv_heads,
                head_dim,
                hidden_size,
                n_heads,
                dtype_bytes,
                hf_token,
                max_depth,
            ],
            outputs=[result_table, csv_file, log_output],
        )

        if PRESETS:
            preset.change(
                apply_preset,
                inputs=preset,
                outputs=[C, beta, mem_gib, overhead_gib],
            )

    return demo


if __name__ == "__main__":
    build_ui().launch()
