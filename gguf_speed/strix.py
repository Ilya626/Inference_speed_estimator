"""Helpers for ingesting Strix Halo llama.cpp performance datasets."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import requests

_USER_AGENT = "gguf-speed-estimator/1.0"
_JSON_MIME_TYPES = {"application/json", "text/json", "application/ld+json"}
_CSV_MIME_TYPES = {"text/csv", "application/csv", "application/vnd.ms-excel"}
_CONTEXT_KEYS = [
    "context",
    "context_length",
    "context_len",
    "context_tokens",
    "ctx_len",
    "ctx",
    "n_ctx",
    "tokens",
    "n_tokens",
    "sequence_length",
    "prompt_length",
    "prompt_tokens",
    "length",
]
_AVG_KEYS = [
    "avg",
    "average",
    "mean",
    "avg_tps",
    "avg_tok_s",
    "avg_tokens_per_second",
    "avg_token_per_second",
    "avg_token_per_sec",
    "avg_t/s",
    "median",
    "p50",
]
_BEST_KEYS = [
    "best",
    "max",
    "peak",
    "best_tps",
    "best_tok_s",
    "best_tokens_per_second",
    "best_t/s",
    "p95",
    "p99",
]
_NUMBER_RE = re.compile(r"[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?")


class PerformanceError(RuntimeError):
    """Raised when Strix Halo performance data cannot be parsed."""


@dataclass(frozen=True)
class Measurement:
    """Single inference measurement from the Strix Halo dataset."""

    context: int
    avg: float | None
    best: float | None


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if number > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = text.replace(",", "")
        match = _NUMBER_RE.search(text)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
        return number if number > 0 else None
    return None


def _parse_context(value: Any) -> int | None:
    number = _parse_number(value)
    if number is None:
        return None
    context = int(round(number))
    if context <= 0:
        return None
    if abs(number - context) > 1e-6:
        return None
    return context


def _row_from_mapping(mapping: Mapping[str, Any]) -> Measurement | None:
    context: int | None = None
    avg: float | None = None
    best: float | None = None

    keys = {str(key).lower(): key for key in mapping.keys()}

    for candidate in _CONTEXT_KEYS:
        key = keys.get(candidate)
        if key is not None:
            context = _parse_context(mapping[key])
            if context is not None:
                break
    if context is None:
        for key, original in keys.items():
            if "ctx" in key or "context" in key:
                context = _parse_context(mapping[original])
                if context is not None:
                    break
    if context is None:
        return None

    for candidate in _AVG_KEYS:
        key = keys.get(candidate)
        if key is not None:
            avg = _parse_number(mapping[key])
            if avg is not None:
                break
    if avg is None:
        for key, original in keys.items():
            if "avg" in key or "mean" in key or "median" in key or "p50" in key:
                avg = _parse_number(mapping[original])
                if avg is not None:
                    break

    for candidate in _BEST_KEYS:
        key = keys.get(candidate)
        if key is not None:
            best = _parse_number(mapping[key])
            if best is not None:
                break
    if best is None:
        for key, original in keys.items():
            if "best" in key or "max" in key or "peak" in key or "p9" in key:
                best = _parse_number(mapping[original])
                if best is not None:
                    break

    if avg is None and best is None:
        return None

    return Measurement(context=context, avg=avg, best=best)


def _merge_measurements(rows: Iterable[Measurement]) -> list[Measurement]:
    merged: dict[int, Measurement] = {}
    for row in rows:
        existing = merged.get(row.context)
        if existing:
            avg = row.avg if row.avg is not None else existing.avg
            best = row.best if row.best is not None else existing.best
            merged[row.context] = Measurement(context=row.context, avg=avg, best=best)
        else:
            merged[row.context] = row
    return sorted(merged.values(), key=lambda item: item.context)


def _collect_from_json(data: Any) -> list[Measurement]:
    rows: list[Measurement] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, Mapping):
            candidate = _row_from_mapping(obj)
            if candidate is not None:
                rows.append(candidate)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                visit(item)

    visit(data)
    return _merge_measurements(rows)


def _parse_csv(text: str) -> list[Measurement]:
    rows: list[Measurement] = []
    reader = csv.DictReader(io.StringIO(text))
    for record in reader:
        candidate = _row_from_mapping(record)
        if candidate is not None:
            rows.append(candidate)
    return _merge_measurements(rows)


def _extract_json_blob(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{[]", text):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_label(data: Any) -> str | None:
    if isinstance(data, Mapping):
        for key in ("model", "model_name", "name", "title", "gguf", "file"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            label = _extract_label(value)
            if label:
                return label
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for item in data:
            label = _extract_label(item)
            if label:
                return label
    return None


def fetch_performance(url: str, *, timeout: float = 15.0) -> tuple[list[Measurement], str | None]:
    """Fetch and parse Strix Halo llama.cpp performance data from ``url``."""

    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Performance URL is empty.")

    response = requests.get(cleaned, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    if response.status_code >= 400:
        raise PerformanceError(
            f"HTTP {response.status_code} while fetching performance data from {cleaned}"
        )

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    text = response.text

    if content_type in _JSON_MIME_TYPES or cleaned.lower().endswith(".json"):
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - depends on remote payloads
            raise PerformanceError("Failed to decode JSON performance data") from exc
        rows = _collect_from_json(data)
        label = _extract_label(data)
    elif content_type in _CSV_MIME_TYPES or cleaned.lower().endswith(".csv"):
        rows = _parse_csv(text)
        label = None
    else:
        data: Any | None = None
        try:
            data = json.loads(text)
        except ValueError:
            data = _extract_json_blob(text)
        if data is not None:
            rows = _collect_from_json(data)
            label = _extract_label(data)
        else:
            rows = _parse_csv(text)
            label = None

    if not rows:
        raise PerformanceError("No inference measurements found in the supplied data.")

    return rows, label
