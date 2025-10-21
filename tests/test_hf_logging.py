import json
import unittest
from unittest import mock

from gguf_speed import hf


class ListGGUFLoggingTests(unittest.TestCase):
    def test_html_fallback_used_when_api_fails(self):
        repo_id = "someone/model"
        next_data = {
            "props": {
                "pageProps": {
                    "modelRepo": {
                        "siblings": [
                            {"rfilename": "model.Q4_K_M.gguf", "size": 1024},
                            {"rfilename": "README.md", "size": 2048},
                            {"rfilename": "model.Q5_K_M.GGUF", "size": 2048},
                        ]
                    }
                }
            }
        }
        html = (
            "<html><head><script id=\"__NEXT_DATA__\" type=\"application/json\">"
            + json.dumps(next_data)
            + "</script></head></html>"
        )

        api_error = hf.HFError("api offline")

        class DummyResponse:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

        with mock.patch("gguf_speed.hf.get_model_info", side_effect=api_error):
            with mock.patch("requests.get", return_value=DummyResponse(html)):
                with self.assertLogs("gguf_speed.hf", level="ERROR") as captured:
                    entries = hf.list_gguf(repo_id)

        self.assertTrue(captured.output, "Expected an error log entry to be emitted")
        log_entry = "\n".join(captured.output)
        self.assertIn(repo_id, log_entry)
        self.assertIn("api offline", log_entry)

        gguf_files = {entry["rfilename"] for entry in entries}
        self.assertEqual(gguf_files, {"model.Q4_K_M.gguf", "model.Q5_K_M.GGUF"})
        for entry in entries:
            self.assertIsInstance(entry.get("size"), int)

    def test_logs_error_when_fallback_fails(self):
        repo_id = "someone/model"
        api_error = hf.HFError("network down")
        fallback_error = hf.HFError("html missing")

        with mock.patch("gguf_speed.hf.get_model_info", side_effect=api_error):
            with mock.patch(
                "gguf_speed.hf._fetch_model_page_metadata", side_effect=fallback_error
            ):
                with self.assertLogs("gguf_speed.hf", level="ERROR") as captured:
                    with self.assertRaises(hf.HFError):
                        hf.list_gguf(repo_id)

        self.assertTrue(captured.output, "Expected error log entries to be emitted")
        log_entry = "\n".join(captured.output)
        self.assertIn(repo_id, log_entry)
        self.assertIn("network down", log_entry)
        self.assertIn("html missing", log_entry)


if __name__ == "__main__":
    unittest.main()
