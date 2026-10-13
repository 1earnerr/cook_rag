from config import RAGConfig


def test_default_path_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RAGConfig(index_save_path=str(tmp_path / 'custom-index'))
    from pathlib import Path
    assert Path(config.data_path).is_dir()
    assert config.index_metadata_path == str(tmp_path / 'custom-index' / 'index_metadata.json')
    assert config.query_rewrite_enabled is False
    assert config.recipe_metadata_enabled is False
    assert config.retrieval_relevance_threshold == 0.45


def test_relevance_threshold_validation():
    import pytest

    with pytest.raises(ValueError, match='threshold'):
        RAGConfig(retrieval_relevance_threshold=1.1)
