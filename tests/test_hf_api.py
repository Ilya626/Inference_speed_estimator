import unittest
from unittest import mock

from gguf_speed import hf


class GetModelInfoTests(unittest.TestCase):
    def test_prefers_siblings_expansion(self):
        repo_id = "someone/model"

        class DummyResponse:
            status_code = 200
            text = "{}"

            def json(self):
                return {"siblings": []}

        with mock.patch("requests.get", return_value=DummyResponse()) as mock_get:
            info = hf.get_model_info(repo_id)

        self.assertEqual(info, {"siblings": []})
        self.assertEqual(mock_get.call_count, 1)
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs.get("params"), {"expand": ["siblings"]})

    def test_falls_back_when_api_rejects_expand(self):
        repo_id = "someone/model"

        class DummyResponse:
            def __init__(self, status_code, payload=None, text=""):
                self.status_code = status_code
                self._payload = payload or {}
                self.text = text or "{}"

            def json(self):
                return dict(self._payload)

        error = DummyResponse(400, text="{\"error\": \"bad expand\"}")
        success = DummyResponse(200, payload={"siblings": []})

        with mock.patch("requests.get", side_effect=[error, success]) as mock_get:
            info = hf.get_model_info(repo_id)

        self.assertEqual(info, {"siblings": []})
        self.assertEqual(mock_get.call_count, 2)
        first_call = mock_get.call_args_list[0]
        second_call = mock_get.call_args_list[1]
        self.assertEqual(first_call.kwargs.get("params"), {"expand": ["siblings"]})
        self.assertEqual(second_call.kwargs.get("params"), {"expand": ["files"]})


if __name__ == "__main__":
    unittest.main()
