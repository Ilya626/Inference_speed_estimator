# GGUF Speed Estimator (Strix Halo calibrated)

GGUF Speed Estimator is a utility that predicts llama.cpp-style inference
throughput for dense and Mixture-of-Experts (MoE) GGUF models hosted on the
Hugging Face Hub.  The default calibration targets Ryzen AI 9 HX-395 (Strix
Halo) in best-of mode with FA/RPC enabled.  The tool automatically analyses
Hugging Face repositories, evaluates expected token generation speed for
multiple context lengths, and—when architectural metadata is
available—estimates KV cache usage to flag potential out-of-memory scenarios.

## Quick start: Gradio UI

The fastest way to explore models is via the bundled Gradio interface.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app/ui.py
```

Gradio launches on <http://127.0.0.1:7860> by default.  The UI remembers your
previous inputs while the server is running, making it easy to iterate on
repositories, calibration constants, and memory constraints.

### UI tour

**Repository & presets**

* Paste a Hugging Face URL or `owner/repo` slug into **Hugging Face repo or URL**.
* Pick a **Preset** (from `app/presets.yaml`) to preload calibration constants
  and default memory budgets—`strix-halo` ships out of the box.

**Contexts & calibration controls**

* Use the **Contexts (tokens)** checkboxes for common lengths (4k–32k) or add
  custom values in **Extra contexts** (comma or semicolon separated).
* **Model type** toggles between dense and MoE approximations.  The **Dense C**
  and **Dense β** fields expose the dense calibration constants, while the four
  **MoE** inputs (total/active coefficients and exponents) shape the MoE
  approximation.  **MoE active ratio (0-1)** defines the share of active
  parameters (`ratio × GGUF size`).  Adjust any of them manually or use the
  Calibration helper to refine dense constants.
* **KV dtype bytes** and **Base-model depth** control how deeply the app searches
  linked repositories for architecture metadata.

**Memory budgets & overrides**

* Specify **Memory budget (GiB)** and **Overhead (GiB)** to enable OOM
  annotations in the results table.
* Provide a private **HF token** if the repository is gated or to raise the rate
  limit.
* Expand **KV overrides** to manually fill `n_layers`, `n_kv_heads`, `head_dim`,
  `hidden_size`, or `n_heads` when metadata is missing.

**Results, CSV export & activity log**

* Press **Fetch & Compute** to enumerate `.gguf` files, evaluate each requested
  context, and populate the **Predictions** dataframe.
* The **Download CSV** widget provides a ready-to-share export; Gradio generates
  a fresh file every time the estimator runs.
* Review the **log** panel for KV cache source information, measurement
  assumptions, and progress summaries.

**Calibration helper**

* Expand the **Calibration helper** accordion to calibrate directly inside the
  UI.
* Paste a direct GGUF link into **GGUF reference** so the helper can infer the
  model size.
* Optionally load historical throughput from the Strix Halo portal via
  **Strix Halo performance link** and **Load Strix Halo data**—detected contexts
  pre-fill the measurement fields and surface a preview table.
* Provide at least two contexts (average and/or best token rates) and click
  **Compute calibration**.  The helper outputs textual summaries plus separate
  average/best calibration objects.
* Apply the new constants to the main form with **Use average calibration** or
  **Use best calibration**.

> ℹ️ Presets live in `app/presets.yaml`; drop in your own calibration constants
> and memory defaults to share team-wide setups.  The UI also auto-loads
> `gguf_speed/data/model_database.csv` when Hugging Face metadata is incomplete,
> so you keep KV insights even while offline.

## Features at a glance

* **Automatic GGUF discovery:** fetches the list of `.gguf` files directly from
  the Hugging Face model page (via `https://huggingface.co/api/models/{repo}`).
* **Speed estimates:** applies the calibrated dense or MoE formula for each
  available quant and context length (4k, 8k, 16k, 24k, 32k by default; arbitrary
  lengths are supported).
* **Memory checks:** parses architectural fields from the model card or
  `config.json`, follows "base model" links when necessary, and reports KV cache
  footprint together with OOM status for a given memory budget.
* **Interactive UI workflows:** presets, checkbox-driven contexts, inline
  overrides, CSV export, logs, and built-in calibration tools streamline ad-hoc
  exploration.
* **Quant insight:** infers nominal bits-per-weight (bpw) for each GGUF file and
  reports an approximate parameter count derived from the artefact sizes.
* **Local metadata fallback:** includes a curated CSV with KV-cache and GGUF
  size data for popular models so you can work offline or when repository
  metadata is incomplete.

> ℹ️ The bundled dense and MoE calibrations target Strix Halo.  For other
> hardware backends, gather a few empirical points and refit the constants.

## Speed model

The estimator ships with two analytic approximations calibrated on Strix Halo
measurements.

### Dense models

\[
\text{speed}(L) = \frac{C}{\text{size}_{\text{GiB}} + \beta \cdot L}
\]

Where:

* `speed(L)` – predicted tokens per second for context length `L`.
* `size_GiB` – GGUF file size in gibibytes (GiB), reported by Hugging Face.
* `L` – context length in tokens (e.g. 4096, 8192, 16384, 24576, 32768).
* `C`, `β` – dense calibration constants (default `C = 206`, `β = 0.000512`).

If KV-cache parameters are known, an equivalent formulation with
`α = β / \text{KV-per-token}_{\text{GiB}}` is used to reason about architecture
impact.

### MoE models

\[
\text{speed} = 2593.21 \times \text{total}^{-1.51} + 44.9 \times \text{active}^{-0.47}
\]

Where:

* `speed` – predicted tokens per second.
* `total` – GGUF artefact size in GiB (after quantisation).
* `active` – active parameter footprint in GiB, computed as `ratio × total` for
  a given activation sparsity (default ratio `0.113`).

The four MoE coefficients are exposed in both the CLI and the UI so you can
re-fit them for other hardware profiles.

The same metadata allows us to estimate total memory usage:

\[
\text{mem}_{\text{need}}(L) \approx \text{size}_{\text{GiB}} + \text{KV}_{\text{GiB}}(L) + \text{overhead}_{\text{GiB}}
\]

The KV cache term is computed as:

\[
\text{KV}_{\text{GiB}}(L) = \left(\frac{2 \cdot \text{dtype}_{\text{bytes}} \cdot n_{\ell} \cdot n_{\text{kv}} \cdot d_{\text{head}}}{2^{30}}\right) L
\]

The default `dtype_bytes` is 2 (FP16/BF16) and the multiplier accounts for both
K and V tensors.

## Local metadata CSV

When network access is unavailable—or when Hugging Face metadata omits key
fields—the tool falls back to `gguf_speed/data/model_database.csv`.  The CSV
contains rows for popular repositories, including KV-cache parameters and
typical GGUF artefact sizes across common quantizations.  Both the CLI and
Gradio UI automatically consult this file when online lookups fail, so you can
still estimate KV requirements and memory usage for those models.  Extend or
replace the CSV with your own entries to cover additional architectures or
quant variants.

## How it works

1. Call `https://huggingface.co/api/models/{repo_id}` to obtain the list of
   repository siblings and filter `.gguf` artefacts (including file sizes).
2. Load `config.json` and `README` files from the repository.  When the README
   references a base model, follow the link(s) and inspect their `config.json`
   as well.
3. Parse the available metadata for KV-cache parameters (`num_hidden_layers`,
   `num_key_value_heads`, `num_attention_heads`/`hidden_size`, `head_dim`).
4. Compute predicted speeds for each GGUF/context pair using the formula above.
5. If KV data is present, estimate memory usage, apply the configured budget,
   and mark rows as `OK` or `OOM`.
6. Present the result as a table, optionally emit a CSV file, and expose the
   workflow through a Gradio interface.

## Calibration workflow in Gradio

1. Open the **Calibration helper** accordion in the UI.
2. Provide a direct GGUF reference (`https://huggingface.co/.../model.gguf` or
   `owner/repo:path`) so the tool can resolve the artefact size automatically.
3. Either load measurements from the Strix Halo portal with **Load Strix Halo
   data** or type in your own average/best token rates for two distinct context
   lengths.
4. Click **Compute calibration** to generate constants; review the textual
   summary for validation details.
5. Apply the preferred set with **Use average calibration** or **Use best
   calibration**—the buttons immediately update the main **Dense C** and **Dense β** fields.
6. Optionally persist your calibration by adding a new entry to
   `app/presets.yaml` so the preset dropdown includes your hardware profile.

For scripted or bulk calibration, you can still call
`gguf_speed.formula.fit_calibration` directly from Python.  See the CLI section
below for automation-friendly options.

## Additional: command-line interface

The CLI remains available for batch processing, CI pipelines, or environments
without a browser.  It mirrors the UI functionality with flag-based controls.

```bash
# Basic run (uses default contexts and calibration)
python cli.py "https://huggingface.co/mradermacher/Agatha-111B-v1-i1-GGUF"

# Specify contexts explicitly and save CSV
python cli.py owner/repo --contexts 4096 8192 16384 32768 --csv out.csv

# Provide KV overrides when metadata is missing and set memory budget
python cli.py owner/repo \
  --n-layers 96 --n-kv-heads 8 --head-dim 128 --mem-gib 64

# Adjust dense calibration constants or load a preset
python cli.py owner/repo --C 180 --beta 0.00042
python cli.py owner/repo --preset strix-halo

# Switch to the MoE estimator and tune its parameters
python cli.py owner/repo --model-type moe --moe-ratio 0.12 \
  --moe-total-coeff 2700 --moe-active-coeff 50
```

Supply `--hf-token` if you need to access private repositories or want to avoid
rate limiting.

## Caveats

* The shipped dense and MoE calibrations target Strix Halo measurements.  Gather
  hardware-specific samples and refit the constants for other devices.
* KV cache calculations assume FP16/BF16 storage.  If your runtime stores KV in
  a different format (e.g. FP8), adjust `dtype_bytes` accordingly.
* Metadata quality on Hugging Face varies; when architectural fields are
  missing, OOM detection is disabled until you provide explicit overrides.
* API calls rely on the public Hugging Face endpoints.  Handle HTTP errors and
  rate limiting as needed (the CLI accepts an `--hf-token`).

## Roadmap ideas

* Device presets and one-click switching of calibration constants.
* Assisted calibration by ingesting CSV logs with empirical measurements.
* Support for alternative KV data types (FP8 and beyond).
* MoE detection with specialised heuristics.
* Additional export targets (Markdown/HTML dashboards).

## License

This project is released under the terms of the MIT License.  See [LICENSE](LICENSE).
