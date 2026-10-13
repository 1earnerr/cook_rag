"""Thin application service over the existing RAG core."""

from __future__ import annotations

import logging
from time import perf_counter

from main import MarkdownRAGSystem
from schemas import (
    ChatResponse,
    HealthResponse,
    ReindexResponse,
    RetrievalInfo,
    SourceResponse,
)

logger = logging.getLogger(__name__)


class RAGServiceError(RuntimeError):
    """Base class for errors safe to map at the HTTP boundary."""


class RAGNotReadyError(RAGServiceError):
    pass


class RAGExecutionError(RAGServiceError):
    pass


class LLMInvocationError(RAGServiceError):
    pass


class ReindexError(RAGServiceError):
    pass


def _is_external_llm_error(exc: Exception) -> bool:
    if isinstance(exc, ValueError) and any(
        name in str(exc) for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    ):
        return True
    try:
        from openai import OpenAIError

        return isinstance(exc, OpenAIError)
    except ImportError:
        return False


class RAGService:
    """Manage core lifecycle and convert core results into API schemas."""

    def __init__(self, system: MarkdownRAGSystem | None = None):
        self.system = system or MarkdownRAGSystem()
        self.last_initialization_error: str | None = None

    def initialize(self) -> None:
        """Load the embedding model and build or reuse the current index."""
        try:
            if self.system.index_module is None or self.system.generation_module is None:
                self.system.initialize_system()
            self.system.build_knowledge_base()
            self.last_initialization_error = None
        except Exception as exc:
            self.last_initialization_error = "RAG initialization failed"
            logger.exception("RAG service initialization failed")
            raise

    def ask(self, query: str, filters: dict | None = None) -> ChatResponse:
        if self.system.retrieval_module is None or self.system.generation_module is None:
            raise RAGNotReadyError("RAG service is not initialized")
        try:
            result = self.system.ask_question(query, filters=filters or None)
        except Exception as exc:
            if _is_external_llm_error(exc):
                logger.error("External LLM invocation failed (%s)", type(exc).__name__)
                raise LLMInvocationError("External LLM invocation failed") from exc
            logger.exception("RAG retrieval or generation failed")
            raise RAGExecutionError("RAG request failed") from exc

        sources = [
            SourceResponse(
                source_id=item.evidence_id,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                title=item.title,
                section=item.section,
                source=item.source,
            )
            for item in result.sources
        ]
        return ChatResponse(
            answer=result.answer,
            sources=sources,
            retrieval=RetrievalInfo(count=len(sources)),
        )

    def health(self) -> HealthResponse:
        index_module = self.system.index_module
        data_module = self.system.data_module
        embedding_loaded = bool(index_module and index_module.embeddings is not None)
        index_loaded = bool(
            index_module
            and index_module.vectorstore is not None
            and self.system.retrieval_module is not None
        )
        documents = len(data_module.documents) if data_module is not None else 0
        chunks = len(data_module.chunks) if data_module is not None else 0
        ready = embedding_loaded and index_loaded
        return HealthResponse(
            status="ok" if ready else "degraded",
            embedding_loaded=embedding_loaded,
            index_loaded=index_loaded,
            documents=documents,
            chunks=chunks,
            detail=None if ready else self.last_initialization_error or "RAG service is not initialized",
        )

    def reindex(self) -> ReindexResponse:
        started = perf_counter()
        try:
            if self.system.index_module is None or self.system.generation_module is None:
                self.system.initialize_system()
            self.system.build_knowledge_base(force=True)
            self.last_initialization_error = None
            metadata = self.system.index_metadata or {}
            return ReindexResponse(
                success=True,
                documents=len(self.system.data_module.documents),
                chunks=len(self.system.data_module.chunks),
                index_schema_version=int(metadata["schema_version"]),
                elapsed_seconds=perf_counter() - started,
            )
        except Exception as exc:
            logger.exception("RAG reindex failed")
            raise ReindexError("RAG reindex failed") from exc
