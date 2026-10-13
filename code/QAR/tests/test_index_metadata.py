from pathlib import Path

from rag_modules.index_metadata import (build_index_metadata, load_index_metadata,
    metadata_matches, save_index_metadata, index_file_hashes, index_files_match)


def test_index_metadata_round_trip(tmp_path: Path):
    records = [("a.md", "内容 A"), ("b.md", "内容 B")]
    metadata = build_index_metadata(
        records=records,
        embedding_model="test-model",
        chunking_version="v1",
        document_count=2,
        chunk_count=4,
    )
    path = tmp_path / "index_metadata.json"
    save_index_metadata(str(path), metadata)
    assert metadata_matches(load_index_metadata(str(path)), metadata)
    assert not metadata_matches(load_index_metadata(str(path)), {**metadata, "chunk_count": 5})


def test_index_metadata_changes_when_content_changes():
    base = build_index_metadata(
        records=[("a.md", "内容 A")], embedding_model="m",
        chunking_version="v1", document_count=1, chunk_count=1,
    )
    changed = build_index_metadata(
        records=[("a.md", "内容 B")], embedding_model="m",
        chunking_version="v1", document_count=1, chunk_count=1,
    )
    assert base["corpus_hash"] != changed["corpus_hash"]


def test_corrupt_or_old_manifest_is_not_reused(tmp_path):
    path = tmp_path / 'metadata.json'
    assert load_index_metadata(str(path)) is None
    path.write_text('{broken', encoding='utf-8')
    assert load_index_metadata(str(path)) is None
    path.write_text('[]', encoding='utf-8')
    assert load_index_metadata(str(path)) is None
    path.write_bytes(b'\xff\xff')
    assert load_index_metadata(str(path)) is None
    assert not metadata_matches({}, {'schema_version': 1})


def test_file_hashes_detect_partial_or_corrupt_save(tmp_path):
    assert not index_files_match(str(tmp_path), {})
    for name in ('index.faiss', 'index.pkl'):
        (tmp_path / name).write_bytes(b'original')
    metadata = {'index_files': index_file_hashes(str(tmp_path))}
    assert index_files_match(str(tmp_path), metadata)
    (tmp_path / 'index.pkl').write_bytes(b'changed')
    assert not index_files_match(str(tmp_path), metadata)
