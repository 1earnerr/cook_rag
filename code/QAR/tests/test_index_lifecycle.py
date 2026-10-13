import hashlib

import pytest
from langchain_core.embeddings import Embeddings
from main import MarkdownRAGSystem
from config import RAGConfig
from rag_modules.index_construction import IndexConstructionModule


class LocalTestEmbeddings(Embeddings):
    """Deterministic offline vectors, not BGE and not a retrieval quality metric."""
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:8]]


def system(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir(exist_ok=True)
    config = RAGConfig(data_path=str(corpus), index_save_path=str(tmp_path / 'index'), embedding_model='offline-test')
    app = MarkdownRAGSystem(config)
    app.index_module = IndexConstructionModule(model_name='offline-test', index_save_path=config.index_save_path,
                                              embeddings=LocalTestEmbeddings())
    return app, corpus


def assert_consistent(app):
    current = {c.metadata['chunk_id']: c for c in app.data_module.chunks}
    dense = app.retrieval_module.vectorstore
    assert dense.index.ntotal == len(current)
    assert set(dense.index_to_docstore_id.values()) == set(current)
    for key, doc in current.items():
        stored = dense.documents_by_id[key]
        assert stored.page_content == doc.page_content
        assert stored.metadata == doc.metadata
    sparse = app.retrieval_module.bm25_retriever.docs
    assert {d.metadata['chunk_id'] for d in sparse} == set(current)
    assert all(d.page_content == current[d.metadata['chunk_id']].page_content for d in sparse)


def test_real_faiss_bm25_reuse_and_corpus_changes(tmp_path, monkeypatch):
    app, corpus = system(tmp_path)
    path = corpus / 'a.md'
    path.write_text('# 手册\n## 安装\n原始内容', encoding='utf-8')
    app.build_knowledge_base()
    assert_consistent(app)
    original = app.index_module.build_vector_index
    def fail(_):
        pytest.fail('unchanged corpus should reuse FAISS')
    monkeypatch.setattr(app.index_module, 'build_vector_index', fail)
    app.build_knowledge_base()
    assert_consistent(app)
    monkeypatch.setattr(app.index_module, 'build_vector_index', original)
    for action in ('modify', 'add', 'rename', 'delete'):
        previous = app.index_metadata['corpus_hash']
        if action == 'modify':
            path.write_text('# 手册\n## 安装\n新版内容', encoding='utf-8')
        elif action == 'add':
            (corpus / 'b.md').write_text('新增内容', encoding='utf-8')
        elif action == 'rename':
            path.rename(corpus / 'renamed.md')
        else:
            (corpus / 'b.md').unlink()
        app.build_knowledge_base()
        assert app.index_metadata['corpus_hash'] != previous
        assert_consistent(app)


@pytest.mark.parametrize('change', ['model', 'chunking', 'recipe', 'missing_manifest', 'corrupt_index'])
def test_configuration_or_file_change_rebuilds(tmp_path, monkeypatch, change):
    app, corpus = system(tmp_path)
    (corpus / 'a.md').write_text('正文', encoding='utf-8')
    app.build_knowledge_base()
    if change == 'model':
        app.config.embedding_model = 'other-offline-model'
        app.index_module = IndexConstructionModule(model_name=app.config.embedding_model,
            index_save_path=app.config.index_save_path, embeddings=LocalTestEmbeddings())
    elif change == 'chunking':
        from rag_modules.data_preparation import DataPreparationModule
        monkeypatch.setattr(DataPreparationModule, 'CHUNKING_VERSION', 'changed-version')
    elif change == 'recipe':
        app.config.recipe_metadata_enabled = True
    elif change == 'missing_manifest':
        from pathlib import Path
        Path(app.config.index_metadata_path).unlink()
    else:
        from pathlib import Path
        (Path(app.config.index_save_path) / 'index.pkl').write_bytes(b'invalid pickle')
    calls = []
    original = app.index_module.build_vector_index
    def build(chunks):
        calls.append(True)
        return original(chunks)
    monkeypatch.setattr(app.index_module, 'build_vector_index', build)
    app.build_knowledge_base()
    assert calls == [True]
    assert_consistent(app)


def test_failed_save_invalidates_marker_and_next_build_recovers(tmp_path, monkeypatch):
    from pathlib import Path
    app, corpus = system(tmp_path)
    path = corpus / 'a.md'
    path.write_text('旧内容', encoding='utf-8')
    app.build_knowledge_base()
    previous_retriever = app.retrieval_module
    path.write_text('新内容', encoding='utf-8')
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError('disk failure')
        patch.setattr(app.index_module, '_save_index_files', fail)
        with pytest.raises(OSError):
            app.build_knowledge_base()
    assert not Path(app.config.index_metadata_path).exists()
    assert app.retrieval_module is previous_retriever
    app.build_knowledge_base()
    assert_consistent(app)


def test_empty_rebuild_does_not_replace_active_corpus(tmp_path):
    app, corpus = system(tmp_path)
    path = corpus / 'a.md'
    path.write_text('正文', encoding='utf-8')
    app.build_knowledge_base()
    previous = app.data_module
    path.unlink()
    with pytest.raises(ValueError, match='没有有效'):
        app.build_knowledge_base()
    assert app.data_module is previous


def test_rrf_does_not_merge_distinct_sources(tmp_path):
    app, corpus = system(tmp_path)
    for name in ('a.md', 'b.md'):
        (corpus / name).write_text('同一正文', encoding='utf-8')
    app.build_knowledge_base()
    a, b = app.data_module.chunks
    result = app.retrieval_module._rrf_rerank([a], [b])
    assert len(result) == 2
    assert {d.metadata['source'] for d in result} == {'a.md', 'b.md'}


def test_changed_model_requires_actual_reinitialization(tmp_path):
    app, corpus = system(tmp_path)
    (corpus / 'a.md').write_text('正文', encoding='utf-8')
    app.config.embedding_model = 'different-model'
    with pytest.raises(ValueError, match='重新初始化'):
        app.build_knowledge_base()


def test_offline_pipeline_real_retrieval_to_prompt_and_sources(tmp_path):
    from langchain_core.runnables import RunnableLambda
    from rag_modules.generation_integration import GenerationIntegrationModule
    app, corpus = system(tmp_path)
    (corpus / 'manual.md').write_text('# Manual\nRecovery command: restore --safe.', encoding='utf-8')
    app.build_knowledge_base()
    prompts = []
    def fake_model(prompt):
        prompts.append(prompt.to_string())
        return 'restore --safe [S1]'
    app.generation_module = GenerationIntegrationModule(llm=RunnableLambda(fake_model))
    result = app.ask_question('Recovery command')
    assert result.answer == 'restore --safe [S1]'
    assert result.sources[0].render() in prompts[0]
    assert result.sources[0].chunk_id == app.data_module.chunks[0].metadata['chunk_id']
    assert result.sources[0].source == 'manual.md'


def test_direct_vector_append_is_blocked(tmp_path):
    app, _ = system(tmp_path)
    with pytest.raises(NotImplementedError, match='统一重建'):
        app.index_module.add_documents([])


def test_faiss_save_and_load_supports_unicode_windows_path(tmp_path):
    from langchain_core.documents import Document

    index_path = tmp_path / '中文索引'
    embeddings = LocalTestEmbeddings()
    module = IndexConstructionModule(
        model_name='offline-test', index_save_path=str(index_path), embeddings=embeddings)
    chunk = Document('恢复命令 restore --safe', metadata={'chunk_id': 'stable-chunk'})
    module.build_vector_index([chunk])
    module.save_index()

    loader = IndexConstructionModule(
        model_name='offline-test', index_save_path=str(index_path), embeddings=embeddings)
    assert loader.load_index() is not None
    assert loader.similarity_search('restore', k=1)[0].metadata['chunk_id'] == 'stable-chunk'
