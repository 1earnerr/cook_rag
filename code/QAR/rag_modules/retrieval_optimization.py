"""Dense, sparse and hybrid retrieval for a generic Markdown corpus."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .native_faiss import NativeFAISSIndex, copy_document

logger = logging.getLogger(__name__)


class NativeBM25Retriever:
    """Minimal rank_bm25 adapter with the same corpus order for stable ranking."""

    def __init__(self, documents: List[Document], preprocess_func):
        self.docs = list(documents)
        self.preprocess_func = preprocess_func
        self.vectorizer = BM25Okapi([
            preprocess_func(document.page_content) for document in self.docs
        ])

    def search(self, query: str, k: int,
               allowed_chunk_ids: set[str] | None = None) -> List[Document]:
        scores = self.vectorizer.get_scores(self.preprocess_func(query))
        if allowed_chunk_ids is None:
            candidate_positions = np.arange(len(self.docs))
        else:
            candidate_positions = np.asarray([
                position for position, document in enumerate(self.docs)
                if document.metadata.get("chunk_id") in allowed_chunk_ids
            ], dtype=int)
        if len(candidate_positions) == 0:
            return []
        # Match rank_bm25.get_top_n ordering, including its reverse-index tie order.
        local_order = np.argsort(scores[candidate_positions])[::-1]
        positions = candidate_positions[local_order[:min(k, len(candidate_positions))]]
        return [copy_document(self.docs[int(position)]) for position in positions]


class RetrievalOptimizationModule:
    """Keep retrieval policy in one place; domain metadata remains caller supplied."""

    def __init__(self, vectorstore: NativeFAISSIndex, chunks: List[Document],
                 vector_fetch_k: int = 20, bm25_fetch_k: int = 20,
                 final_k: int = 5, rrf_k: int = 60):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_fetch_k = vector_fetch_k
        self.bm25_fetch_k = bm25_fetch_k
        self.final_k = final_k
        self.rrf_k = rrf_k
        if min(vector_fetch_k, bm25_fetch_k, final_k, rrf_k) < 1:
            raise ValueError('检索参数必须为正整数')
        self.setup_retrievers()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Deterministic mixed-language tokenizer without a new search dependency."""
        tokens: List[str] = []
        for part in re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9_]+', text.lower()):
            if re.fullmatch(r'[\u4e00-\u9fff]+', part):
                tokens.extend(list(part))
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                tokens.append(part)
        return tokens

    def setup_retrievers(self):
        self.bm25_retriever = NativeBM25Retriever(self.chunks, self.tokenize)

    def vector_search(self, query: str, fetch_k: int | None = None,
                      allowed_chunk_ids: set[str] | None = None) -> List[Document]:
        k = fetch_k or self.vector_fetch_k
        documents = self.vectorstore.similarity_search(
            query, k=k, allowed_chunk_ids=allowed_chunk_ids)
        for rank, document in enumerate(documents, 1):
            document.metadata['vector_rank'] = rank
        return documents

    def bm25_search(self, query: str, fetch_k: int | None = None,
                    allowed_chunk_ids: set[str] | None = None) -> List[Document]:
        documents = self.bm25_retriever.search(
            query, fetch_k or self.bm25_fetch_k, allowed_chunk_ids=allowed_chunk_ids)
        for rank, document in enumerate(documents, 1):
            document.metadata['bm25_rank'] = rank
        return documents

    def retrieve(self, query: str, filters: Dict[str, Any] | None = None,
                 mode: str = 'hybrid', vector_fetch_k: int | None = None,
                 bm25_fetch_k: int | None = None, final_k: int | None = None) -> List[Document]:
        """Unified retrieval entry point; mode is vector, bm25 or hybrid."""
        limit = final_k or self.final_k
        allowed_chunk_ids = self._matching_chunk_ids(filters or {})
        if mode == 'vector':
            candidates = self.vector_search(query, vector_fetch_k, allowed_chunk_ids)
        elif mode == 'bm25':
            candidates = self.bm25_search(query, bm25_fetch_k, allowed_chunk_ids)
        elif mode == 'hybrid':
            candidates = self._rrf_rerank(
                self.vector_search(query, vector_fetch_k, allowed_chunk_ids),
                self.bm25_search(query, bm25_fetch_k, allowed_chunk_ids),
            )
        else:
            raise ValueError("mode 必须是 vector、bm25 或 hybrid")
        candidates = self._apply_metadata_filter(candidates, filters or {})
        return candidates[:limit]

    def _matching_chunk_ids(self, filters: Dict[str, Any]) -> set[str] | None:
        if not filters:
            return None
        return {
            str(document.metadata["chunk_id"])
            for document in self.chunks
            if self._metadata_matches(document, filters)
        }

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        return self.retrieve(query, mode='hybrid', final_k=top_k)

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        return self.retrieve(query, filters=filters, mode='hybrid', final_k=top_k)

    @staticmethod
    def _apply_metadata_filter(docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        if not filters:
            return docs
        return [doc for doc in docs if RetrievalOptimizationModule._metadata_matches(doc, filters)]

    @staticmethod
    def _metadata_matches(doc: Document, filters: Dict[str, Any]) -> bool:
        return all(doc.metadata.get(key) in value if isinstance(value, (list, tuple, set))
                   else doc.metadata.get(key) == value for key, value in filters.items())

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int | None = None) -> List[Document]:
        rrf_k = k or self.rrf_k
        scores: Dict[str, float] = {}
        objects: Dict[str, Document] = {}
        ranks: Dict[str, Dict[str, int | float]] = {}
        for source_name, docs in (('vector', vector_docs), ('bm25', bm25_docs)):
            for rank, doc in enumerate(docs, 1):
                chunk_id = doc.metadata.get('chunk_id')
                if not chunk_id:
                    raise ValueError('检索文档缺少稳定 chunk_id')
                objects[chunk_id] = doc
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                ranks.setdefault(chunk_id, {})[f'{source_name}_rank'] = rank
        ordered = sorted(scores, key=lambda item: (-scores[item], item))
        result = []
        for chunk_id in ordered:
            doc = objects[chunk_id]
            doc.metadata.update(ranks[chunk_id])
            doc.metadata['rrf_score'] = scores[chunk_id]
            result.append(doc)
        return result
