import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gguf_speed import formula, hf, kv, quant, table


class HuggingFaceQwenCalculationTests(unittest.TestCase):
    REPO_URL = "https://huggingface.co/unsloth/Qwen3-14B-GGUF"
    CONTEXTS = [4096, 8192]
    MEM_BUDGET_GIB = 18.5
    KV_CONFIG = kv.KVConfig(
        n_layers=40,
        n_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        source="unit-test",
    )

    def _fetch_gguf_entries(self) -> tuple[str, list[dict[str, object]]]:
        """Download GGUF metadata for the configured repository."""

        repo_id = hf.normalize_repo_id(self.REPO_URL)
        try:
            entries = hf.list_gguf(repo_id)
        except hf.HFError as exc:  # pragma: no cover - network guard
            message = str(exc)
            # Network or service level issues should not fail CI runs.
            if message.startswith("Repository not found"):
                raise
            self.skipTest(f"Unable to reach Hugging Face Hub: {message}")
        self.assertTrue(entries, "Hugging Face repository does not contain GGUF files")
        return repo_id, list(entries)

    def test_speed_estimation_works_for_qwen_repo(self):
        repo_id, gguf_entries = self._fetch_gguf_entries()

        contexts = list(self.CONTEXTS)
        kv_config = self.KV_CONFIG
        mem_budget_gib = self.MEM_BUDGET_GIB

        df = table.build(
            repo_id,
            gguf_entries,
            contexts,
            formula.DenseCalibration(),
            formula.MoECalibration(),
            formula.ModelType.DENSE,
            kv_config=kv_config,
            mem_gib=mem_budget_gib,
            overhead_gib=4.0,
        )

        files_from_repo = {str(entry["rfilename"]) for entry in gguf_entries}
        self.assertSetEqual(set(df["file"].unique()), files_from_repo)
        self.assertEqual(len(df), len(gguf_entries) * len(contexts))
        self.assertEqual(sorted(df["context_tokens"].unique().tolist()), sorted(contexts))
        self.assertEqual(df["repo"].unique().tolist(), [repo_id])

        target_file = "Qwen3-14B-Q4_K_M.gguf"
        self.assertIn(target_file, files_from_repo)
        target_entry = next(entry for entry in gguf_entries if entry["rfilename"] == target_file)
        self.assertIsInstance(target_entry.get("size"), int)

        param_estimate = quant.estimate_param_count(gguf_entries)
        self.assertIsNotNone(param_estimate)

        display_columns = [
            "file",
            "quant",
            "context_tokens",
            "size_gib",
            "pred_speed_toks_per_s",
            "kv_gib",
            "est_mem_gib",
            "oom",
        ]
        display_df = (
            df[display_columns]
            .sort_values(["file", "context_tokens"])
            .reset_index(drop=True)
        )
        formatters = {
            "context_tokens": lambda value: f"{int(value):d}",
            "size_gib": lambda value: f"{float(value):.4f}",
            "pred_speed_toks_per_s": lambda value: f"{float(value):.3f}",
            "kv_gib": lambda value: f"{float(value):.4f}",
            "est_mem_gib": lambda value: f"{float(value):.4f}",
        }
        calculation_report = display_df.to_string(index=False, formatters=formatters)
        print(f"\nРасчёт скоростей для {repo_id}:\n" + calculation_report)

        target_rows = df[df["file"] == target_file].set_index("context_tokens")
        self.assertEqual(len(target_rows), len(contexts))

        size_q4_gib = formula.bytes_to_gib(int(target_entry["size"]))
        expected_q4_size = round(size_q4_gib, 4)
        self.assertTrue((target_rows["size_gib"] == expected_q4_size).all())

        expected_quant_label = quant.infer_quant(target_file)
        self.assertEqual(set(target_rows["quant"]), {expected_quant_label})

        expected_params_b = round(param_estimate / 1e9, 4)
        self.assertTrue((target_rows["params_b"] == expected_params_b).all())

        expected_bpw = round(
            quant.bpw_from_size(int(target_entry["size"]), float(param_estimate)),
            4,
        )
        self.assertTrue((target_rows["bpw"] == expected_bpw).all())

        kv_per_token = kv_config.kv_per_token_gib
        for context in contexts:
            expected_speed = round(
                formula.DenseCalibration().speed(size_q4_gib, context),
                3,
            )
            observed_speed = target_rows.loc[context, "pred_speed_toks_per_s"]
            self.assertEqual(observed_speed, expected_speed)

            expected_kv_gib = round(kv_per_token * context, 4)
            self.assertEqual(target_rows.loc[context, "kv_gib"], expected_kv_gib)

            expected_mem_gib = round(
                formula.mem_need(size_q4_gib, kv_per_token * context, 4.0),
                4,
            )
            self.assertEqual(target_rows.loc[context, "est_mem_gib"], expected_mem_gib)

            expected_oom = "OK" if expected_mem_gib <= mem_budget_gib else "OOM"
            self.assertEqual(target_rows.loc[context, "oom"], expected_oom)


if __name__ == "__main__":
    unittest.main()
