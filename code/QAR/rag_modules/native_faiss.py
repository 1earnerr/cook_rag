"""Small native FAISS adapter for the operations QAR actually uses."""

from __future__ import annotations

from typing import Iterable

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def copy_document(document: Document) -> Document:
    """Return an isolated retrieval result so rank metadata cannot leak."""
    return Document(page_content=document.page_content, metadata=dict(document.metadata))


class NativeFAISSIndex:
    """IndexFlatL2 plus stable chunk-id-to-Document mapping."""

    FORMAT_VERSION = "qar-native-faiss-v1"

    def __init__(self, embeddings: Embeddings, index, documents_by_id: dict[str, Document],
                 index_to_docstore_id: dict[int, str]):
        self.embeddings = embeddings
        self.index = index
        self.documents_by_id = documents_by_id
        self.index_to_docstore_id = index_to_docstore_id
        self._validate()

    @classmethod
    def from_documents(cls, documents: Iterable[Document], embeddings: Embeddings):
        docs = list(documents)
        if not docs:
            raise ValueError("文档块列表不能为空")
        chunk_ids = [str(doc.metadata.get("chunk_id", "")) for doc in docs]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("向量索引文档缺少稳定 chunk_id")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("向量索引存在重复 chunk_id")

        vectors = np.asarray(
            embeddings.embed_documents([doc.page_content for doc in docs]),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(docs) or vectors.shape[1] < 1:
            raise ValueError("Embedding 返回的文档向量形状无效")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        documents_by_id = {
            chunk_id: copy_document(doc) for chunk_id, doc in zip(chunk_ids, docs)
        }
        index_to_docstore_id = dict(enumerate(chunk_ids))
        return cls(embeddings, index, documents_by_id, index_to_docstore_id)

    def _validate(self) -> None:
        expected_positions = set(range(self.index.ntotal))
        if set(self.index_to_docstore_id) != expected_positions:
            raise ValueError("FAISS 位置映射与向量数量不一致")
        mapped_ids = list(self.index_to_docstore_id.values())
        if len(set(mapped_ids)) != len(mapped_ids):
            raise ValueError("FAISS 位置映射包含重复 chunk_id")
        if set(mapped_ids) != set(self.documents_by_id):
            raise ValueError("FAISS 位置映射与 Document 映射不一致")

    def _query_vector(self, query: str) -> np.ndarray:
        vector = np.asarray([self.embeddings.embed_query(query)], dtype=np.float32)
        if vector.ndim != 2 or vector.shape[1] != self.index.d:
            raise ValueError("Query Embedding 维度与 FAISS 索引不一致")
        return np.ascontiguousarray(vector)

    def similarity_search(self, query: str, k: int = 5,
                          allowed_chunk_ids: set[str] | None = None) -> list[Document]:
        if k < 1 or self.index.ntotal == 0:
            return []
        vector = self._query_vector(query)
        if allowed_chunk_ids is None:
            _, positions = self.index.search(vector, min(k, self.index.ntotal))
            ranked_positions = positions[0]
        else:
            candidate_positions = [
                position for position, chunk_id in self.index_to_docstore_id.items()
                if chunk_id in allowed_chunk_ids
            ]
            if not candidate_positions:
                return []
            candidate_vectors = np.vstack([
                self.index.reconstruct(position) for position in candidate_positions
            ]).astype(np.float32, copy=False)
            squared_l2 = np.sum((candidate_vectors - vector[0]) ** 2, axis=1)
            order = np.argsort(squared_l2, kind="stable")[:min(k, len(candidate_positions))]
            ranked_positions = [candidate_positions[int(index)] for index in order]
        return [
            copy_document(self.documents_by_id[self.index_to_docstore_id[int(position)]])
            for position in ranked_positions
            if position >= 0
        ]

    def semantic_similarities(self, query: str, documents: Iterable[Document]) -> list[float]:
        """Cosine similarity against the normalized vectors already stored in FAISS."""
        docs = list(documents)
        if not docs:
            return []
        position_by_id = {
            chunk_id: position for position, chunk_id in self.index_to_docstore_id.items()
        }
        positions = []
        for document in docs:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id not in position_by_id:
                raise ValueError("Evidence chunk_id 不在当前 FAISS 索引中")
            positions.append(position_by_id[chunk_id])
        query_vector = self._query_vector(query)[0]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return [0.0] * len(positions)
        scores = []
        for position in positions:
            vector = self.index.reconstruct(position).astype(np.float32, copy=False)
            denominator = query_norm * np.linalg.norm(vector)
            scores.append(float(np.dot(query_vector, vector) / denominator) if denominator else 0.0)
        return scores

    def persistence_payload(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "documents_by_id": self.documents_by_id,
            "index_to_docstore_id": self.index_to_docstore_id,
        }

    @classmethod
    def from_persistence_payload(cls, embeddings: Embeddings, index, payload):
        if not isinstance(payload, dict) or payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("不是受支持的 QAR native FAISS 索引格式")
        return cls(
            embeddings,
            index,
            payload.get("documents_by_id", {}),
            payload.get("index_to_docstore_id", {}),
        )
