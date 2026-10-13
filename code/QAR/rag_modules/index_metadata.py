"""Lightweight metadata used to validate a persisted local index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2


def corpus_fingerprint(records: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for source, content in sorted(records):
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_index_metadata(*, records: Iterable[tuple[str, str]], embedding_model: str,
                        chunking_version: str, document_count: int,
                        chunk_count: int, chunks=(), recipe_metadata_enabled: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_hash": corpus_fingerprint(records),
        "embedding_model": embedding_model,
        "chunking_version": chunking_version,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "chunks_hash": corpus_fingerprint((c.metadata['chunk_id'], json.dumps(
            {'text': c.page_content, 'metadata': c.metadata}, sort_keys=True, ensure_ascii=False
        )) for c in chunks),
        "recipe_metadata_enabled": recipe_metadata_enabled,
    }


def save_index_metadata(path: str, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def load_index_metadata(path: str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def metadata_matches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if not actual:
        return False
    keys = ("schema_version", "corpus_hash", "embedding_model", "chunking_version",
            "document_count", "chunk_count", "chunks_hash", "recipe_metadata_enabled")
    return all(actual.get(key) == expected.get(key) for key in keys)


def index_file_hashes(directory: str) -> dict[str, str]:
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
            for name in ('index.faiss', 'index.pkl')}


def index_files_match(directory: str, metadata: dict) -> bool:
    try:
        return index_file_hashes(directory) == metadata.get('index_files')
    except OSError:
        return False
