import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_modules.data_preparation import DataPreparationModule
from rag_modules.native_faiss import NativeFAISSIndex
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


class LocalTestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [value / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]


def make_retriever():
    chunks = [
        Document('宫保鸡丁 鸡肉 花生', metadata={'chunk_id': 'c1', 'document_id': 'd1', 'title': '菜谱', 'source': 'a.md', 'section': '原料', 'kind': 'recipe'}),
        Document('服务恢复 restore --safe', metadata={'chunk_id': 'c2', 'document_id': 'd2', 'title': '运维手册', 'source': 'b.md', 'section': '恢复', 'kind': 'manual'}),
        Document('数据库备份 backup', metadata={'chunk_id': 'c3', 'document_id': 'd3', 'title': '运维手册', 'source': 'c.md', 'section': '备份', 'kind': 'manual'}),
    ]
    store = NativeFAISSIndex.from_documents(chunks, LocalTestEmbeddings())
    return RetrievalOptimizationModule(store, chunks, vector_fetch_k=2, bm25_fetch_k=2, final_k=2)


def test_chinese_tokenizer_has_unigrams_bigrams_and_ascii():
    tokens = RetrievalOptimizationModule.tokenize('宫保鸡丁 restore-123')
    assert '宫' in tokens and '宫保' in tokens and '鸡丁' in tokens
    assert 'restore' in tokens and '123' in tokens


def test_vector_bm25_and_hybrid_are_independently_callable():
    retriever = make_retriever()
    assert len(retriever.retrieve('restore', mode='vector', vector_fetch_k=2, final_k=2)) == 2
    vector = retriever.retrieve('restore', mode='vector', vector_fetch_k=2, final_k=2)
    assert [doc.metadata['vector_rank'] for doc in vector] == [1, 2]
    bm25 = retriever.retrieve('restore', mode='bm25', bm25_fetch_k=2, final_k=2)
    assert len(bm25) == 2
    assert [doc.metadata['bm25_rank'] for doc in bm25] == [1, 2]
    hybrid = retriever.retrieve('restore', mode='hybrid', final_k=2)
    assert len(hybrid) == 2
    assert all('rrf_score' in doc.metadata for doc in hybrid)


def test_rrf_preserves_rank_metadata_and_stable_identity():
    retriever = make_retriever()
    docs = retriever._rrf_rerank(retriever.chunks[:2], retriever.chunks[1:])
    by_id = {doc.metadata['chunk_id']: doc for doc in docs}
    assert by_id['c1'].metadata['vector_rank'] == 1
    assert by_id['c2'].metadata['vector_rank'] == 2
    assert by_id['c2'].metadata['bm25_rank'] == 1
    assert by_id['c1'].metadata['rrf_score'] > 0


def test_metadata_filter_is_generic_and_supports_collections():
    retriever = make_retriever()
    manual = retriever.retrieve('restore', filters={'kind': 'manual'}, mode='hybrid', final_k=5)
    assert manual and all(doc.metadata['kind'] == 'manual' for doc in manual)
    selected = retriever.retrieve('restore', filters={'source': ['b.md', 'c.md']}, mode='bm25', final_k=5)
    assert selected and all(doc.metadata['source'] in {'b.md', 'c.md'} for doc in selected)


class FilterCandidateEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[10.0, 0.0] if text == 'filtered target' else [0.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [0.0, 0.0]


@pytest.mark.parametrize('mode', ['vector', 'bm25', 'hybrid'])
def test_metadata_filter_limits_candidate_generation_before_top_k(mode):
    chunks = [
        Document('filtered target', metadata={
            'chunk_id': 'target', 'document_id': 'target-doc', 'source': 'target.md'}),
        Document('generic query generic query', metadata={
            'chunk_id': 'noise', 'document_id': 'noise-doc', 'source': 'noise.md'}),
    ]
    store = NativeFAISSIndex.from_documents(chunks, FilterCandidateEmbeddings())
    retriever = RetrievalOptimizationModule(
        store, chunks, vector_fetch_k=1, bm25_fetch_k=1, final_k=1)

    result = retriever.retrieve(
        'generic query', filters={'source': 'target.md'}, mode=mode, final_k=1)

    assert [document.metadata['chunk_id'] for document in result] == ['target']


class SimilarityEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] if text == 'aligned' else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_native_faiss_reuses_stored_vectors_for_cosine_relevance():
    chunks = [
        Document('aligned', metadata={'chunk_id': 'a'}),
        Document('orthogonal', metadata={'chunk_id': 'b'}),
    ]
    store = NativeFAISSIndex.from_documents(chunks, SimilarityEmbeddings())
    assert store.semantic_similarities('query', chunks) == pytest.approx([1.0, 0.0])


def test_long_section_uses_recursive_fallback_and_stable_subchunks(tmp_path):
    content = '# 运行手册\n## 恢复\n' + ('第一步：执行 restore --safe。\n' * 500)
    path = tmp_path / 'manual.md'
    path.write_text(content, encoding='utf-8')
    module = DataPreparationModule(str(tmp_path), max_section_chars=120, section_chunk_overlap=20)
    module.load_documents()
    chunks = module.chunk_documents()
    assert len(chunks) > 1
    assert max(len(chunk.page_content) for chunk in chunks) <= 120
    assert len({chunk.metadata['chunk_id'] for chunk in chunks}) == len(chunks)
    assert all(chunk.metadata['document_id'] == chunks[0].metadata['document_id'] for chunk in chunks)
    assert all(chunk.metadata['section'] == '运行手册 / 恢复' for chunk in chunks)
    assert all('section_index' in chunk.metadata and 'subchunk_index' in chunk.metadata for chunk in chunks)
    again = DataPreparationModule(str(tmp_path), max_section_chars=120, section_chunk_overlap=20)
    again.load_documents()
    assert [c.metadata['chunk_id'] for c in again.chunk_documents()] == [c.metadata['chunk_id'] for c in chunks]


class CountingTokenizer:
    model_max_length = 512

    def __call__(self, text, add_special_tokens=True, truncation=False):
        # Deterministic test tokenizer: one token per non-space character.
        count = len(''.join(text.split())) + (2 if add_special_tokens else 0)
        return {'input_ids': list(range(count))}


@pytest.mark.parametrize('body', [
    '这是一段用于验证超长中文章节切分的知识库内容。' * 100,
    'restore service build 2026 port 8080 and verify status. ' * 100,
])
def test_token_budget_is_checked_with_injected_embedding_tokenizer(tmp_path, body):
    text = '# Mixed\n## Section\n' + body
    (tmp_path / 'manual.md').write_text(text, encoding='utf-8')
    module = DataPreparationModule(str(tmp_path), max_section_chars=500,
        section_chunk_overlap=30, embedding_chunk_max_tokens=80,
        embedding_tokenizer=CountingTokenizer())
    module.load_documents()
    chunks = module.chunk_documents()
    assert chunks and all(module._token_count(chunk.page_content) <= 80 for chunk in chunks)
    assert all(chunk.metadata['document_id'] and chunk.metadata['parent_id'] for chunk in chunks)
    assert all(chunk.metadata['source'] == 'manual.md' for chunk in chunks)
    assert all(chunk.metadata['title'] == 'Mixed' for chunk in chunks)
    assert all(chunk.metadata['section'] == 'Mixed / Section' for chunk in chunks)
    again = DataPreparationModule(str(tmp_path), max_section_chars=500,
        section_chunk_overlap=30, embedding_chunk_max_tokens=80,
        embedding_tokenizer=CountingTokenizer())
    again.load_documents()
    assert [chunk.metadata['chunk_id'] for chunk in again.chunk_documents()] == [chunk.metadata['chunk_id'] for chunk in chunks]


def test_token_aware_recursive_split_keeps_configured_overlap(tmp_path):
    # Unique characters make the suffix/prefix overlap observable without
    # accidental matches from repeated prose.
    body = ''.join(chr(0x4e00 + index) for index in range(300))
    (tmp_path / 'manual.md').write_text(
        '# Manual\n## Long section\n' + body, encoding='utf-8')
    module = DataPreparationModule(
        str(tmp_path), max_section_chars=160, section_chunk_overlap=24,
        embedding_chunk_max_tokens=70, embedding_tokenizer=CountingTokenizer())
    module.load_documents()
    chunks = module.chunk_documents()

    def overlap(left, right):
        maximum = min(len(left), len(right))
        return max((size for size in range(1, maximum + 1)
                    if left[-size:] == right[:size]), default=0)

    assert all(module._token_count(chunk.page_content) <= 70 for chunk in chunks)
    assert any(overlap(left.page_content, right.page_content) > 0
               for left, right in zip(chunks, chunks[1:]))
