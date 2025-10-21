"""Helpers for talking to the Hugging Face Hub."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger(__name__)
_API_BASE = "https://huggingface.co/api/models/"
_RESOLVE_TEMPLATE = "https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


class HFError(RuntimeError):
    """Raised when the Hugging Face API returns an unexpected response."""


def normalize_repo_id(repo_or_url: str) -> str:
    """Return a canonical ``owner/repo`` identifier."""

    repo_or_url = repo_or_url.strip()
    if not repo_or_url:
        raise ValueError("Repository identifier is empty.")

    if "://" in repo_or_url:
        parsed = urlparse(repo_or_url)
        if parsed.netloc and "huggingface.co" not in parsed.netloc:
            raise ValueError(f"Unsupported host: {parsed.netloc}")
        path = parsed.path.strip("/")
        if not path:
            raise ValueError("Invalid Hugging Face URL.")
        parts = [segment for segment in path.split("/") if segment]
        if not parts:
            raise ValueError("Invalid Hugging Face URL path.")
        if parts[0] in {"models", "spaces", "datasets"} and len(parts) >= 3:
            parts = parts[1:3]
        if parts[0] == "blob" and len(parts) >= 3:
            parts = parts[1:3]
        if len(parts) >= 3 and parts[1] == "tree":
            parts = [parts[0], parts[2]]
        if len(parts) < 2:
            raise ValueError("Unable to determine repository identifier from URL.")
        repo_id = "/".join(parts[:2])
        return repo_id

    if repo_or_url.count("/") != 1:
        raise ValueError("Repository identifier must be in the form owner/name.")
    return repo_or_url


def _build_headers(token: str | None) -> MutableMapping[str, str]:
    headers: MutableMapping[str, str] = {
        "Accept": "application/json",
        "User-Agent": "gguf-speed-estimator/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_model_info(repo_id: str, token: str | None = None) -> Mapping[str, object]:
    url = _API_BASE + repo_id
    try:
        response = requests.get(url, headers=_build_headers(token))
    except requests.RequestException as exc:  # pragma: no cover - network guard
        raise HFError(f"Error fetching model info for {repo_id}: {exc}") from exc
    if response.status_code == 404:
        raise HFError(f"Repository not found: {repo_id}")
    if response.status_code >= 300:
        raise HFError(f"Unexpected API error {response.status_code}: {response.text}")
    return response.json()


def list_gguf(repo_id: str, token: str | None = None) -> list[Mapping[str, object]]:
    info = get_model_info(repo_id, token=token)
    files: list[Mapping[str, object]] = []
    for sibling in info.get("siblings", []):
        name = sibling.get("rfilename")
        if not isinstance(name, str):
            continue
        if not name.lower().endswith(".gguf"):
            continue
        size = sibling.get("size")
        if size is None:
            LOGGER.warning("Skipping %s with unknown size", name)
            continue
        files.append({"rfilename": name, "size": int(size)})
    files.sort(key=lambda item: item["rfilename"])
    return files


def parse_gguf_reference(value: str) -> tuple[str, str]:
    """Parse a GGUF reference into ``(repo_id, file_path)``."""

    value = value.strip()
    if not value:
        raise ValueError("GGUF reference is empty.")

    if "://" in value:
        parsed = urlparse(value)
        if parsed.netloc and "huggingface.co" not in parsed.netloc:
            raise ValueError(f"Unsupported host for GGUF reference: {parsed.netloc}")
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) < 3:
            raise ValueError("GGUF URL is missing path information.")
        repo_id = normalize_repo_id("/".join(segments[:2]))
        tail = segments[2:]
        if tail and tail[0] in {"blob", "resolve", "tree"}:
            if len(tail) < 3:
                raise ValueError("GGUF URL is missing revision or filename.")
            tail = tail[2:]
        file_path = "/".join(tail)
    elif ":" in value:
        repo_part, file_part = value.split(":", 1)
        repo_id = normalize_repo_id(repo_part)
        file_path = file_part.lstrip("/")
    else:
        parts = [segment for segment in value.split("/") if segment]
        if len(parts) < 3:
            raise ValueError("GGUF reference must include repo and filename.")
        repo_id = normalize_repo_id("/".join(parts[:2]))
        file_path = "/".join(parts[2:])

    file_path = file_path.lstrip("/")
    if not file_path:
        raise ValueError("GGUF reference does not include a filename.")
    return repo_id, file_path


def get_gguf_file(
    repo_id: str,
    file_path: str,
    *,
    token: str | None = None,
) -> Mapping[str, object]:
    """Return metadata for a GGUF file within ``repo_id``.

    Parameters
    ----------
    repo_id:
        Hugging Face repository identifier (``owner/repo``).
    file_path:
        Path to the GGUF file within the repository.  May include directories.
    token:
        Optional Hugging Face token used for authentication.
    """

    file_path = file_path.lstrip("/")
    ggufs = list_gguf(repo_id, token=token)
    for entry in ggufs:
        name = str(entry.get("rfilename"))
        if name == file_path:
            return entry

    target = Path(file_path).name
    matches = [entry for entry in ggufs if Path(str(entry.get("rfilename"))).name == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HFError(
            f"Multiple GGUF files named {target!r} found in {repo_id}; provide the full path."
        )
    raise HFError(f"GGUF file not found in {repo_id}: {file_path}")


def read_repo_text(
    repo_id: str,
    path: str,
    *,
    revision: str = "main",
    token: str | None = None,
) -> str | None:
    url = _RESOLVE_TEMPLATE.format(repo_id=repo_id, revision=revision, path=path)
    headers = _build_headers(token)
    headers["Accept"] = "text/plain"
    try:
        response = requests.get(url, headers=headers)
    except requests.RequestException as exc:  # pragma: no cover - network guard
        LOGGER.warning("Failed to fetch %s from %s: %s", path, repo_id, exc)
        return None
    if response.status_code == 404:
        return None
    if response.status_code >= 300:
        raise HFError(
            f"Error fetching {path} from {repo_id}: {response.status_code} {response.text}"
        )
    return response.text


def extract_repo_ids(text: str) -> list[str]:
    """Extract ``owner/repo`` identifiers from free-form text."""

    matches: list[str] = []
    seen: set[str] = set()
    if not text:
        return matches

    base_model_re = re.compile(r"base_model\s*[:=]\s*['\"]?([\w\-.]+/[\w\-.]+)")
    for match in base_model_re.findall(text):
        if match not in seen:
            seen.add(match)
            matches.append(match)

    url_re = re.compile(r"https?://huggingface\.co/([\w\-.]+)/([\w\-.]+)")
    for owner, name in url_re.findall(text):
        candidate = f"{owner}/{name}"
        if candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)

    repo_re = re.compile(r"['\"]([\w\-.]+/[\w\-.]+)['\"]")
    for candidate in repo_re.findall(text):
        if candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)

    return matches


def iter_potential_kv_texts(
    repo_id: str,
    *,
    token: str | None = None,
    max_depth: int = 1,
) -> Iterator[tuple[str, str]]:
    """Yield blobs of text that may contain KV information."""

    visited: set[str] = set()

    def fetch(repo: str, level: int) -> Iterator[tuple[str, str]]:
        if repo in visited or level > max_depth:
            return
        visited.add(repo)

        config = read_repo_text(repo, "config.json", token=token)
        if config:
            yield (config, f"{repo}:config.json")

        readme = None
        for candidate in ("README.md", "README.MD", "README"):
            readme = read_repo_text(repo, candidate, token=token)
            if readme:
                yield (readme, f"{repo}:{candidate}")
                break

        sources: list[str] = []
        info = None
        try:
            info = get_model_info(repo, token=token)
        except HFError as exc:
            LOGGER.warning("Failed to fetch info for %s: %s", repo, exc)
        if info and isinstance(info, Mapping):
            card = info.get("cardData")
            if isinstance(card, Mapping):
                base = card.get("base_model")
                if isinstance(base, str):
                    sources.append(base)
                elif isinstance(base, Sequence):
                    sources.extend(str(item) for item in base if isinstance(item, str))
        if readme:
            sources.extend(extract_repo_ids(readme))

        for candidate in sources:
            try:
                normalized = normalize_repo_id(candidate)
            except ValueError:
                continue
            if normalized in visited:
                continue
            yield from fetch(normalized, level + 1)

    yield from fetch(repo_id, 0)
