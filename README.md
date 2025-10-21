# GGUF Speed Estimator (Strix Halo calibrated)

GGUF Speed Estimator is a utility that predicts llama.cpp-style inference
throughput for dense GGUF models hosted on the Hugging Face Hub.  The default
calibration targets Ryzen AI 9 HX-395 (Strix Halo) in best-of mode with FA/RPC
enabled.  The tool automatically analyses Hugging Face repositories, evaluates
expected token generation speed for multiple context lengths, and—when
architectural metadata is available—estimates KV cache usage to flag potential
out-of-memory scenarios.

## Features

* **Automatic GGUF discovery:** fetches the list of `.gguf` files directly from
  the Hugging Face model page (via `https://huggingface.co/api/models/{repo}`).
* **Speed estimates:** applies the calibrated formula for each available quant
  and context length (4k, 8k, 16k, 24k, 32k by default; arbitrary lengths are
  supported).
* **Memory checks:** parses architectural fields from the model card or
  `config.json`, follows "base model" links when necessary, and reports KV cache
  footprint together with OOM status for a given memory budget.
* **Multiple outputs:** produces a tabular summary, lets you export to CSV, and
  ships with a Gradio UI for point-and-click exploration.
* **Quant insight:** infers nominal bits-per-weight (bpw) for each GGUF file and
  reports an approximate parameter count derived from the artefact sizes.
* **Local metadata fallback:** includes a curated CSV with KV-cache and GGUF
  size data for popular models so you can work offline or when repository
  metadata is incomplete.

> ℹ️ The current calibration is meant for dense models on Strix Halo.  For other
> hardware backends, gather a few empirical points and refit the constants.

## Speed model

The estimator uses the following expression for token throughput:

\[
\text{speed}(L) = \frac{C}{\text{size}_{\text{GiB}} + \beta \cdot L}
\]

Where:

* `speed(L)` – predicted tokens per second for context length `L`.
* `size_GiB` – GGUF file size in gibibytes (GiB), reported by Hugging Face.
* `L` – context length in tokens (e.g. 4096, 8192, 16384, 24576, 32768).
* `C`, `β` – calibration constants (default `C = 206`, `β = 0.000512`).

If KV-cache parameters are known, an equivalent formulation with
`α = β / \text{KV-per-token}_{\text{GiB}}` is used to reason about architecture
impact.

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
typical GGUF artefact sizes across common quantizations.  The CLI and Gradio UI
automatically consult this file when online lookups fail, so you can still
estimate KV requirements and memory usage for those models.  Extend or replace
the CSV with your own entries to cover additional architectures or quant
variants.

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

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Command-line usage

```bash
# Basic run (uses default contexts and calibration)
python cli.py "https://huggingface.co/mradermacher/Agatha-111B-v1-i1-GGUF"

# Specify contexts explicitly and save CSV
python cli.py owner/repo --contexts 4096 8192 16384 32768 --csv out.csv

# Provide KV overrides when metadata is missing and set memory budget
python cli.py owner/repo \
  --n-layers 96 --n-kv-heads 8 --head-dim 128 --mem-gib 64

# Adjust calibration constants or load a preset
python cli.py owner/repo --C 180 --beta 0.00042
python cli.py owner/repo --preset strix-halo
```

Supply `--hf-token` if you need to access private repositories or want to avoid
rate limiting.

## Gradio UI

```bash
python app/ui.py
```

The interface lets you select context lengths via checkboxes, tweak calibration
constants, set memory and overhead budgets, and override KV parameters.  Results
are shown in an interactive table with a download button for the generated CSV.

## Calibration guidance

* Collect a few measurements `(size_GiB, context_tokens, speed)` for your
  hardware.
* Use the CLI or a separate script to fit `C` and `β` with
  `gguf_speed.formula.fit_calibration`.
* Update the defaults or create a new preset in `app/presets.yaml` for quick
  reuse.

### Gradio calibration helper

The Gradio UI ships with a **Calibration helper** accordion.  Provide a direct
link (or `owner/repo:path`) to a GGUF file together with empirical speed
measurements for two different context lengths (both average and best results
are supported).  The tool resolves the model size automatically, fits the
calibration constants, and lets you apply either set of coefficients to the main
estimator with a single click.

If you already have results recorded on the Strix Halo llama.cpp performance
portal, paste the public link into the **Strix Halo performance link** field and
press **Load Strix Halo data**.  The UI will fetch the available contexts (both
average and best throughput), pre-populate the measurement fields, and render a
mini table so you can double-check the import before computing calibration
constants.

## Caveats

* The calibration targets dense models.  Mixture-of-Experts architectures with
  sparse activation may require a different model.
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
