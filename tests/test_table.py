import unittest

from gguf_speed import formula, table


class TableBuildTests(unittest.TestCase):
    def test_invalid_size_metadata_is_handled(self):
        gguf_entries = [{"rfilename": "model.gguf", "size": "unknown"}]
        contexts = [4096]

        with self.assertLogs("gguf_speed.table", level="WARNING") as logs:
            df = table.build(
                "repo/test",
                gguf_entries,
                contexts,
                formula.DenseCalibration(),
                formula.MoECalibration(),
                formula.ModelType.DENSE,
                None,
            )

        self.assertTrue(any("Unable to parse size" in message for message in logs.output))
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.loc[0, "size_gib"])
        self.assertIsNone(df.loc[0, "pred_speed_toks_per_s"])


if __name__ == "__main__":
    unittest.main()
