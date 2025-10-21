import json

import pytest

from gguf_speed import hf


@pytest.mark.parametrize(
    "value, expected",
    [
        (123, 123),
        (123.0, 123),
        ("123456", 123456),
        ("123,456", 123456),
        ("123_456", 123456),
        ("123 456", 123456),
        ("15 GB", int(15 * 1000**3)),
        ("15.5GB", int(15.5 * 1000**3)),
        ("7.25 GiB", int(7.25 * 1024**3)),
        ("1024 KiB", int(1024 * 1024)),
        ("size 4096", 4096),
        ("SIZE=8192", 8192),
        (b"size 2048", 2048),
        ("version https://git-lfs.github.com/spec/v1\nsize 16384", 16384),
        (json.dumps({"size": "1.5 GB"}), int(1.5 * 1000**3)),
        (json.dumps({"sizeBytes": 512}), 512),
    ],
)
def test_parse_size_candidate_success(value, expected):
    assert hf._parse_size_candidate(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, "", "sha256", "NaN", float("nan"), {"unexpected": "value"}]
)
def test_parse_size_candidate_returns_none(value):
    assert hf._parse_size_candidate(value) is None


def test_extract_size_handles_multiple_sources():
    entry = {
        "size": None,
        "metadata": {"size": "14.5 GB"},
        "lfs": {"text": "size 1048576"},
    }
    assert hf._extract_size(entry) == int(14.5 * 1000**3)

