"""Lightweight CSV-backed metadata for popular GGUF models."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from . import kv

_DATA_PATH = Path(__file__).resolve().parent / "data" / "model_database.csv"


@dataclass(frozen=True)
class GGUFVariant:
    """Representation of a GGUF artefact stored in the local database."""

    rfilename: str
    quant: str | None
    size: int | None


@dataclass(frozen=True)
class RepoEntry:
    """Metadata bundle extracted from the CSV for a given repository."""

    repo: str
    model_family: str | None
    variant: str | None
    kv_fields: Mapping[str, int]
    kv_config: kv.KVConfig | None
    gguf_variants: tuple[GGUFVariant, ...]
    sources: tuple[str, ...]

def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_rows(path: Path) -> Iterable[MutableMapping[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def _build_kv_fields(row: Mapping[str, str]) -> Mapping[str, int]:
    fields: dict[str, int] = {}
    for key in ("n_layers", "n_kv_heads", "head_dim", "hidden_size", "n_heads"):
        value = _parse_int(row.get(key))
        if value is not None:
            fields[key] = value
    return fields

def _make_kv_config(repo: str, fields: Mapping[str, int], dtype_bytes: int | None) -> kv.KVConfig | None:
    required = {"n_layers", "n_kv_heads"}
    if not required.issubset(fields):
        return None
    head_dim = fields.get("head_dim")
    hidden_size = fields.get("hidden_size")
    n_heads = fields.get("n_heads")
    if head_dim is None and hidden_size is not None and n_heads is not None:
        try:
            head_dim = kv.derive_head_dim(hidden_size, n_heads)
        except ValueError:
            head_dim = None
    if head_dim is None:
        return None
    dtype = dtype_bytes if dtype_bytes is not None else 2
    hidden = fields.get("hidden_size")
    heads = fields.get("n_heads")
    return kv.KVConfig(
        n_layers=int(fields["n_layers"]),
        n_kv_heads=int(fields["n_kv_heads"]),
        head_dim=int(head_dim),
        dtype_bytes=int(dtype),
        hidden_size=int(hidden) if hidden is not None else None,
        n_heads=int(heads) if heads is not None else None,
        source="local-db",
    )


@lru_cache()
def _load_database() -> dict[str, RepoEntry]:
    entries: dict[str, RepoEntry] = {}
    by_repo: dict[str, list[Mapping[str, str]]] = {}
    for row in _load_rows(_DATA_PATH):
        repo = (row.get("repo") or "").strip()
        if not repo:
            continue
        by_repo.setdefault(repo, []).append(row)

    for repo, rows in by_repo.items():
        model_family = None
        variant = None
        kv_fields: dict[str, int] = {}
        dtype_bytes: int | None = None
        sources: list[str] = []
        gguf_variants: list[GGUFVariant] = []

        for row in rows:
            if model_family is None:
                model_family = (row.get("model_family") or "").strip() or None
            if variant is None:
                variant = (row.get("variant") or "").strip() or None
            if dtype_bytes is None:
                dtype_bytes = _parse_int(row.get("dtype_bytes"))
            source_value = (row.get("source") or "").strip()
            if source_value and source_value not in sources:
                sources.append(source_value)
            fields = _build_kv_fields(row)
            for key, value in fields.items():
                kv_fields.setdefault(key, value)

            file_name = (row.get("gguf_file") or "").strip()
            quant = (row.get("quant") or "").strip() or None
            size_bytes = _parse_int(row.get("size_bytes"))
            if file_name:
                gguf_variants.append(
                    GGUFVariant(rfilename=file_name, quant=quant, size=size_bytes)
                )

        kv_config = _make_kv_config(repo, kv_fields, dtype_bytes)
        entries[repo] = RepoEntry(
            repo=repo,
            model_family=model_family,
            variant=variant,
            kv_fields=kv_fields,
            kv_config=kv_config,
            gguf_variants=tuple(gguf_variants),
            sources=tuple(sources),
        )

    return entries

def lookup(repo_id: str) -> RepoEntry | None:
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return None
    entries = _load_database()
    if repo_id in entries:
        return entries[repo_id]
    lowered = repo_id.lower()
    for key, value in entries.items():
        if key.lower() == lowered:
            return value
    return None


def kv_override(repo_id: str, *, dtype_bytes: int | None = None) -> kv.KVConfig | None:
    entry = lookup(repo_id)
    if not entry:
        return None
    if entry.kv_config is None:
        return None
    config = entry.kv_config
    if dtype_bytes is None or dtype_bytes == config.dtype_bytes:
        return config
    return kv.KVConfig(
        n_layers=config.n_layers,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype_bytes=int(dtype_bytes),
        hidden_size=config.hidden_size,
        n_heads=config.n_heads,
        source=config.source,
    )


def list_gguf(repo_id: str) -> list[Mapping[str, object]]:
    entry = lookup(repo_id)
    if not entry:
        return []
    result: list[Mapping[str, object]] = []
    for variant in entry.gguf_variants:
        if not variant.rfilename:
            continue
        data: dict[str, object] = {"rfilename": variant.rfilename}
        if variant.size is not None:
            data["size"] = int(variant.size)
        if variant.quant is not None:
            data["quant"] = variant.quant
        result.append(data)
    return result
