import unittest
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gguf_speed import formula, hf, kv, table


class HuggingFaceQwenCalculationTests(unittest.TestCase):
    def test_speed_estimation_works_for_qwen_repo(self):
        repo_url = "https://huggingface.co/unsloth/Qwen3-14B-GGUF"
        repo_id = hf.normalize_repo_id(repo_url)
        self.assertEqual(repo_id, "unsloth/Qwen3-14B-GGUF")

        size_q4 = 14_602_888_832
        size_q5 = 17_179_869_184
        api_payload = {
            "siblings": [
                {"rfilename": "Qwen3-14B-Q4_K_M.gguf", "size": size_q4},
                {"rfilename": "notes/usage.txt", "size": 1024},
                {"rfilename": "Qwen3-14B-Q6_K.gguf", "size": None},
                {"rfilename": "Qwen3-14B-Q5_K_M.gguf", "size": size_q5},
            ]
        }

        fake_response = mock.Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = api_payload
        fake_response.text = ""

        with mock.patch("gguf_speed.hf.requests.get", return_value=fake_response) as mock_get:
            with self.assertLogs("gguf_speed.hf", level="WARNING") as hf_logs:
                gguf_entries = hf.list_gguf(repo_id)

        self.assertEqual(mock_get.call_count, 1)
        requested_url = mock_get.call_args[0][0]
        self.assertEqual(requested_url, hf._API_BASE + repo_id)
        self.assertTrue(
            any("Skipping Qwen3-14B-Q6_K.gguf with unknown size" in message for message in hf_logs.output),
            msg=f"Missing warning about skipped GGUF entries: {hf_logs.output}",
        )

        self.assertEqual(len(gguf_entries), 2)
        self.assertEqual(
            [entry["rfilename"] for entry in gguf_entries],
            ["Qwen3-14B-Q4_K_M.gguf", "Qwen3-14B-Q5_K_M.gguf"],
        )

        contexts = [4096, 8192]
        kv_config = kv.KVConfig(
            n_layers=40,
            n_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
            source="unit-test",
        )

        mem_budget_gib = 18.5

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

        # Отображаем таблицу расчёта, чтобы автотест явно печатал параметры в консоль.
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
        print("\nРасчёт скоростей для unsloth/Qwen3-14B-GGUF:\n" + calculation_report)

        self.assertEqual(len(df), len(gguf_entries) * len(contexts))
        self.assertEqual(sorted(df["context_tokens"].unique().tolist()), sorted(contexts))
        self.assertEqual(df["repo"].unique().tolist(), [repo_id])

        self.assertTrue(df["pred_speed_toks_per_s"].notna().all())
        self.assertTrue(df["kv_gib"].notna().all())
        self.assertTrue(df["est_mem_gib"].notna().all())
        self.assertTrue(df["oom"].isin({"OK", "OOM"}).all())

        size_q4_gib = formula.bytes_to_gib(size_q4)
        expected_q4_size = round(size_q4_gib, 4)
        observed_q4_size = (
            df.loc[df["file"] == "Qwen3-14B-Q4_K_M.gguf", "size_gib"].iloc[0]
        )
        self.assertEqual(observed_q4_size, expected_q4_size)

        expected_q4_speed = round(
            formula.DenseCalibration().speed(size_q4_gib, contexts[0]),
            3,
        )
        observed_q4_speed = df.loc[
            (df["file"] == "Qwen3-14B-Q4_K_M.gguf")
            & (df["context_tokens"] == contexts[0]),
            "pred_speed_toks_per_s",
        ].iloc[0]
        self.assertEqual(observed_q4_speed, expected_q4_speed)

        # Validate KV and memory calculations for a specific row.
        expected_kv_gib = round(kv_config.kv_per_token_gib * contexts[0], 4)
        observed_kv_gib = df.loc[
            (df["file"] == "Qwen3-14B-Q4_K_M.gguf")
            & (df["context_tokens"] == contexts[0]),
            "kv_gib",
        ].iloc[0]
        self.assertEqual(observed_kv_gib, expected_kv_gib)

        expected_mem_gib = round(
            formula.mem_need(size_q4_gib, kv_config.kv_per_token_gib * contexts[0], 4.0),
            4,
        )
        observed_mem_gib = df.loc[
            (df["file"] == "Qwen3-14B-Q4_K_M.gguf")
            & (df["context_tokens"] == contexts[0]),
            "est_mem_gib",
        ].iloc[0]
        self.assertEqual(observed_mem_gib, expected_mem_gib)

        oom_states = df.set_index(["file", "context_tokens"])["oom"].to_dict()
        self.assertEqual(
            oom_states[("Qwen3-14B-Q4_K_M.gguf", contexts[0])],
            "OK",
        )
        self.assertEqual(
            oom_states[("Qwen3-14B-Q4_K_M.gguf", contexts[1])],
            "OOM",
        )
        self.assertEqual(
            oom_states[("Qwen3-14B-Q5_K_M.gguf", contexts[0])],
            "OOM",
        )
        self.assertEqual(
            oom_states[("Qwen3-14B-Q5_K_M.gguf", contexts[1])],
            "OOM",
        )

        quant_labels = {
            row_file: df.loc[df["file"] == row_file, "quant"].iloc[0]
            for row_file in ["Qwen3-14B-Q4_K_M.gguf", "Qwen3-14B-Q5_K_M.gguf"]
        }
        self.assertEqual(quant_labels["Qwen3-14B-Q4_K_M.gguf"], "Q4_K_M")
        self.assertEqual(quant_labels["Qwen3-14B-Q5_K_M.gguf"], "Q5_K_M")


if __name__ == "__main__":
    unittest.main()
