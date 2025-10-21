from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping

import gradio as gr
import pandas as pd
import yaml

from gguf_speed import formula, hf, kv, local_db, strix, table

DEFAULT_CONTEXTS = [4096, 8192, 16384, 24576, 32768]
PRESETS_PATH = Path(__file__).with_name("presets.yaml")
DEFAULT_DENSE_CALIBRATION = formula.DenseCalibration()
DEFAULT_MOE_CALIBRATION = formula.MoECalibration()
DEFAULT_MODEL_TYPE = formula.ModelType.DENSE
DEFAULT_MOE_RATIO = formula.DEFAULT_MOE_RATIO


def load_presets() -> Mapping[str, Mapping[str, object]]:
    if not PRESETS_PATH.exists():
        return {}
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("presets", {})


PRESETS = load_presets()


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


def _extract_model_type(preset: Mapping[str, object], fallback: formula.ModelType) -> formula.ModelType:
    value = preset.get("model_type")
    if isinstance(value, str):
        try:
            return formula.ModelType(value)
        except ValueError:
            return fallback
    return fallback


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

    entry = local_db.lookup(repo_id)
    if entry and entry.kv_fields:
        parts.append((entry.kv_fields, "local-db"))

    config, mode = kv.consolidate_configs(parts, overrides=overrides or None, dtype_bytes=dtype_bytes)

    if config is None and entry and entry.kv_config is not None:
        config = local_db.kv_override(repo_id, dtype_bytes=dtype_bytes)
        mode = "local-db"

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


def _parse_context(value: float | None, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required when providing speeds.")
    context = int(value)
    if abs(context - float(value)) > 1e-6:
        raise ValueError(f"{label} must be an integer number of tokens.")
    if context <= 0:
        raise ValueError(f"{label} must be positive.")
    return context


def _parse_speed(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required for calibration.")
    speed = float(value)
    if speed <= 0:
        raise ValueError(f"{label} must be positive.")
    return speed


def calibrate_from_measurements(
    gguf_ref: str,
    context_a: float | None,
    avg_speed_a: float | None,
    best_speed_a: float | None,
    context_b: float | None,
    avg_speed_b: float | None,
    best_speed_b: float | None,
    hf_token: str | None,
):
    try:
        repo_id, file_path = hf.parse_gguf_reference(gguf_ref)
    except ValueError as exc:
        return None, None, "", f"❌ {exc}"

    try:
        file_info = hf.get_gguf_file(repo_id, file_path, token=hf_token)
    except Exception as exc:  # pragma: no cover - UI layer
        return None, None, "", f"❌ Failed to resolve GGUF file: {exc}"

    size_bytes = int(file_info.get("size", 0))
    size_gib = formula.bytes_to_gib(size_bytes)

    samples_avg: list[tuple[float, int, float]] = []
    samples_best: list[tuple[float, int, float]] = []

    try:
        if any(x is not None for x in (avg_speed_a, best_speed_a)):
            context_a_tokens = _parse_context(context_a, "Context A")
            if avg_speed_a is not None:
                samples_avg.append((size_gib, context_a_tokens, _parse_speed(avg_speed_a, "Average speed A")))
            if best_speed_a is not None:
                samples_best.append((size_gib, context_a_tokens, _parse_speed(best_speed_a, "Best speed A")))
        if any(x is not None for x in (avg_speed_b, best_speed_b)):
            context_b_tokens = _parse_context(context_b, "Context B")
            if avg_speed_b is not None:
                samples_avg.append((size_gib, context_b_tokens, _parse_speed(avg_speed_b, "Average speed B")))
            if best_speed_b is not None:
                samples_best.append((size_gib, context_b_tokens, _parse_speed(best_speed_b, "Best speed B")))
    except ValueError as exc:
        return None, None, "", f"❌ {exc}"

    summary_lines = [
        f"Resolved: {repo_id}/{file_info.get('rfilename')} ({size_gib:.4f} GiB)",
    ]

    avg_cal: formula.Calibration | None = None
    best_cal: formula.Calibration | None = None

    if len(samples_avg) >= 2:
        try:
            avg_cal = formula.fit_calibration(samples_avg)
            summary_lines.append(
                "Average calibration → "
                f"C = {avg_cal.C:.3f}, β = {avg_cal.beta:.6g}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            summary_lines.append(f"Average calibration failed: {exc}")
    else:
        summary_lines.append("Average calibration → need at least two measurements.")

    if len(samples_best) >= 2:
        try:
            best_cal = formula.fit_calibration(samples_best)
            summary_lines.append(
                "Best calibration → "
                f"C = {best_cal.C:.3f}, β = {best_cal.beta:.6g}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            summary_lines.append(f"Best calibration failed: {exc}")
    else:
        summary_lines.append("Best calibration → need at least two measurements.")

    if avg_cal is None and best_cal is None:
        return None, None, "", "⚠️ Unable to compute calibration constants."

    return avg_cal, best_cal, "\n".join(summary_lines), "✅ Calibration computed."


def apply_calibration(cal: formula.DenseCalibration | None, label: str):
    if cal is None:
        return gr.update(), gr.update(), f"❌ No {label} calibration available."
    return (
        gr.update(value=cal.C),
        gr.update(value=cal.beta),
        f"✅ Applied {label} calibration: C = {cal.C:.3f}, β = {cal.beta:.6g}",
    )


def load_strix_measurements(perf_url: str):
    if not perf_url or not perf_url.strip():
        return (
            gr.update(value=None),
            "",
            "⚠️ Provide a Strix Halo performance link to import measurements.",
            *(gr.update() for _ in range(6)),
        )

    try:
        measurements, label = strix.fetch_performance(perf_url)
    except ValueError as exc:
        return (
            gr.update(value=None),
            "",
            f"❌ {exc}",
            *(gr.update() for _ in range(6)),
        )
    except strix.PerformanceError as exc:
        return (
            gr.update(value=None),
            "",
            f"❌ {exc}",
            *(gr.update() for _ in range(6)),
        )
    except Exception as exc:  # pragma: no cover - defensive for network errors
        return (
            gr.update(value=None),
            "",
            f"❌ Failed to load performance data: {exc}",
            *(gr.update() for _ in range(6)),
        )

    rows = [
        {
            "Context (tokens)": item.context,
            "Average tok/s": item.avg,
            "Best tok/s": item.best,
        }
        for item in measurements
    ]
    table_df = pd.DataFrame(rows)

    contexts = ", ".join(str(item.context) for item in measurements)
    summary_lines = []
    if label:
        summary_lines.append(f"Source: {label}")
    summary_lines.append(f"Contexts available: {contexts}")
    summary = "\n".join(summary_lines)

    if len(measurements) < 2:
        status = "⚠️ Only one context detected; provide another measurement before calibrating."
    else:
        status = f"✅ Loaded {len(measurements)} contexts from the Strix Halo dataset."

    first = measurements[0] if measurements else None
    last = measurements[-1] if len(measurements) > 1 else None

    def _update(measurement: strix.Measurement | None, attr: str):
        if measurement is None:
            return gr.update()
        value = getattr(measurement, attr)
        if value is None:
            return gr.update()
        return gr.update(value=value)

    context_updates = []
    if first is not None:
        context_updates.append(gr.update(value=first.context))
        context_updates.append(_update(first, "avg"))
        context_updates.append(_update(first, "best"))
    else:
        context_updates.extend(gr.update() for _ in range(3))

    if last is not None:
        context_updates.append(gr.update(value=last.context))
        context_updates.append(_update(last, "avg"))
        context_updates.append(_update(last, "best"))
    else:
        context_updates.extend(gr.update() for _ in range(3))

    return (table_df, summary, status, *context_updates)


def run_estimator(
    repo_input: str,
    contexts_selected: list[str] | None,
    extra_contexts: str | None,
    model_type_value: str,
    dense_C: float,
    dense_beta: float,
    moe_total_coeff: float,
    moe_total_exp: float,
    moe_active_coeff: float,
    moe_active_exp: float,
    moe_ratio: float | None,
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
        model_type = formula.ModelType(model_type_value)
    except ValueError:
        return None, None, f"❌ Unsupported model type: {model_type_value}"

    dense_calibration = formula.DenseCalibration(C=float(dense_C), beta=float(dense_beta))
    moe_calibration = formula.MoECalibration(
        coeff_total=float(moe_total_coeff),
        exp_total=float(moe_total_exp),
        coeff_active=float(moe_active_coeff),
        exp_active=float(moe_active_exp),
    )
    ratio_value = None
    if moe_ratio is not None:
        try:
            ratio_value = float(moe_ratio)
        except (TypeError, ValueError):
            return None, None, "❌ MoE ratio must be numeric."
        if ratio_value <= 0 or ratio_value > 1:
            return None, None, "❌ MoE ratio must be within (0, 1]."

    hf_error: Exception | None = None
    try:
        ggufs = hf.list_gguf(repo_id, token=hf_token)
    except Exception as exc:  # pragma: no cover - UI layer
        hf_error = exc
        ggufs = []

    if not ggufs:
        fallback = local_db.list_gguf(repo_id)
        if fallback:
            ggufs = fallback
            logs.append("Loaded GGUF metadata from local database.")
            if hf_error is not None:
                logs.append(f"Hugging Face lookup failed: {hf_error}")
        elif hf_error is not None:
            return None, None, f"❌ Failed to list GGUF files: {hf_error}"

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

    if model_type is formula.ModelType.MOE and ratio_value is None:
        ratio_value = DEFAULT_MOE_RATIO

    logs.append(f"Model type: {model_type.value}")
    if model_type is formula.ModelType.MOE:
        logs.append(
            "MoE calibration → "
            f"coeff_total={moe_calibration.coeff_total:.4f} exp_total={moe_calibration.exp_total:.4f} "
            f"coeff_active={moe_calibration.coeff_active:.4f} exp_active={moe_calibration.exp_active:.4f} "
            f"ratio={(ratio_value if ratio_value is not None else DEFAULT_MOE_RATIO):.4f}"
        )
    else:
        logs.append(
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
        mem_gib=mem_gib,
        overhead_gib=overhead_gib,
        moe_ratio=ratio_value,
    )

    temp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8")
    table.to_csv(df, temp.name)
    temp.close()
    logs.append(f"Processed {len(ggufs)} GGUF files across {len(contexts)} context lengths.")
    return df, temp.name, "\n".join(logs)


def apply_preset(name: str):
    preset = PRESETS.get(name) or {}
    updates: list[object] = []

    if "model_type" in preset:
        model_type = _extract_model_type(preset, DEFAULT_MODEL_TYPE)
        updates.append(gr.update(value=model_type.value))
    else:
        updates.append(gr.update())

    dense_cal = _extract_dense_calibration(preset, DEFAULT_DENSE_CALIBRATION)
    updates.append(gr.update(value=dense_cal.C))
    updates.append(gr.update(value=dense_cal.beta))

    moe_section = preset.get("moe")
    has_moe = isinstance(moe_section, Mapping) or any(
        key in preset
        for key in ("moe_coeff_total", "moe_exp_total", "moe_coeff_active", "moe_exp_active")
    )
    if has_moe:
        moe_cal = _extract_moe_calibration(preset, DEFAULT_MOE_CALIBRATION)
        updates.extend(
            [
                gr.update(value=moe_cal.coeff_total),
                gr.update(value=moe_cal.exp_total),
                gr.update(value=moe_cal.coeff_active),
                gr.update(value=moe_cal.exp_active),
            ]
        )
    else:
        updates.extend(gr.update() for _ in range(4))

    has_ratio = False
    if isinstance(moe_section, Mapping):
        has_ratio = "ratio" in moe_section or "moe_ratio" in moe_section
    has_ratio = has_ratio or ("moe_ratio" in preset)
    if has_ratio:
        updates.append(gr.update(value=_extract_moe_ratio(preset, DEFAULT_MOE_RATIO)))
    else:
        updates.append(gr.update())

    for key in ("mem_gib", "overhead_gib"):
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

        avg_state = gr.State()
        best_state = gr.State()

        with gr.Row():
            contexts = gr.CheckboxGroup(
                [str(v) for v in DEFAULT_CONTEXTS],
                value=[str(v) for v in DEFAULT_CONTEXTS],
                label="Contexts (tokens)",
            )
            extra_contexts = gr.Textbox(label="Extra contexts", placeholder="Comma separated e.g. 12288, 40960")

        with gr.Row():
            model_type = gr.Radio(
                [
                    ("Dense", formula.ModelType.DENSE.value),
                    ("MoE", formula.ModelType.MOE.value),
                ],
                value=DEFAULT_MODEL_TYPE.value,
                label="Model type",
            )
            dense_C = gr.Number(label="Dense C", value=DEFAULT_DENSE_CALIBRATION.C)
            dense_beta = gr.Number(label="Dense β", value=DEFAULT_DENSE_CALIBRATION.beta)

        with gr.Row():
            moe_total_coeff = gr.Number(
                label="MoE total coeff",
                value=DEFAULT_MOE_CALIBRATION.coeff_total,
            )
            moe_total_exp = gr.Number(
                label="MoE total exponent",
                value=DEFAULT_MOE_CALIBRATION.exp_total,
            )
            moe_active_coeff = gr.Number(
                label="MoE active coeff",
                value=DEFAULT_MOE_CALIBRATION.coeff_active,
            )
            moe_active_exp = gr.Number(
                label="MoE active exponent",
                value=DEFAULT_MOE_CALIBRATION.exp_active,
            )

        with gr.Row():
            moe_ratio = gr.Number(
                label="MoE active ratio (0-1)",
                value=DEFAULT_MOE_RATIO,
                minimum=0.0,
                maximum=1.0,
            )
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

        with gr.Accordion("Calibration helper", open=False):
            gguf_ref = gr.Textbox(
                label="GGUF reference",
                placeholder="https://huggingface.co/owner/repo/resolve/main/model.gguf",
            )
            perf_url = gr.Textbox(
                label="Strix Halo performance link",
                placeholder="https://strixhalo-homelab.d7.wtf/AI/llamacpp-performance/...",
            )
            load_perf_button = gr.Button("Load Strix Halo data", variant="secondary")
            performance_table = gr.Dataframe(
                label="Imported performance measurements",
                interactive=False,
            )
            performance_summary = gr.Markdown()
            performance_status = gr.Markdown()
            with gr.Row():
                context_a = gr.Number(label="Context A (tokens)", precision=0)
                avg_speed_a = gr.Number(label="Average speed A (tok/s)")
                best_speed_a = gr.Number(label="Best speed A (tok/s)")
            with gr.Row():
                context_b = gr.Number(label="Context B (tokens)", precision=0)
                avg_speed_b = gr.Number(label="Average speed B (tok/s)")
                best_speed_b = gr.Number(label="Best speed B (tok/s)")
            calibrate_button = gr.Button("Compute calibration", variant="secondary")
            with gr.Row():
                use_avg_button = gr.Button("Use average calibration")
                use_best_button = gr.Button("Use best calibration")
            calibration_summary = gr.Markdown()
            calibration_status = gr.Markdown()

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
                model_type,
                dense_C,
                dense_beta,
                moe_total_coeff,
                moe_total_exp,
                moe_active_coeff,
                moe_active_exp,
                moe_ratio,
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

        calibrate_button.click(
            calibrate_from_measurements,
            inputs=[
                gguf_ref,
                context_a,
                avg_speed_a,
                best_speed_a,
                context_b,
                avg_speed_b,
                best_speed_b,
                hf_token,
            ],
            outputs=[avg_state, best_state, calibration_summary, calibration_status],
        )

        load_perf_button.click(
            load_strix_measurements,
            inputs=[perf_url],
            outputs=[
                performance_table,
                performance_summary,
                performance_status,
                context_a,
                avg_speed_a,
                best_speed_a,
                context_b,
                avg_speed_b,
                best_speed_b,
            ],
        )

        use_avg_button.click(
            lambda cal: apply_calibration(cal, "average"),
            inputs=[avg_state],
            outputs=[dense_C, dense_beta, calibration_status],
        )

        use_best_button.click(
            lambda cal: apply_calibration(cal, "best"),
            inputs=[best_state],
            outputs=[dense_C, dense_beta, calibration_status],
        )

        if PRESETS:
            preset.change(
                apply_preset,
                inputs=preset,
                outputs=[
                    model_type,
                    dense_C,
                    dense_beta,
                    moe_total_coeff,
                    moe_total_exp,
                    moe_active_coeff,
                    moe_active_exp,
                    moe_ratio,
                    mem_gib,
                    overhead_gib,
                ],
            )

    return demo


if __name__ == "__main__":
    build_ui().launch()
