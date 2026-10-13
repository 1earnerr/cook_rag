"""Minimal FastAPI boundary for the QAR Markdown knowledge base."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import ChatRequest, ChatResponse, HealthResponse, ReindexResponse
from services import (
    LLMInvocationError,
    RAGExecutionError,
    RAGNotReadyError,
    RAGService,
    ReindexError,
)

logger = logging.getLogger(__name__)


def create_app(service: RAGService | None = None, *, initialize_on_startup: bool = True) -> FastAPI:
    rag_service = service or RAGService()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if initialize_on_startup:
            try:
                rag_service.initialize()
            except Exception:
                # Keep /health available so the real failed state can be inspected.
                logger.error("API started in degraded state because RAG initialization failed")
        yield

    application = FastAPI(
        title="QAR Markdown Knowledge Base API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.state.rag_service = rag_service

    @application.exception_handler(RAGNotReadyError)
    async def handle_not_ready(_: Request, exc: RAGNotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(LLMInvocationError)
    async def handle_llm_error(_: Request, exc: LLMInvocationError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(RAGExecutionError)
    async def handle_rag_error(_: Request, exc: RAGExecutionError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @application.exception_handler(ReindexError)
    async def handle_reindex_error(_: Request, exc: ReindexError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @application.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return rag_service.health()

    @application.post("/api/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return rag_service.ask(request.query, request.filters)

    @application.post("/api/v1/reindex", response_model=ReindexResponse)
    def reindex() -> ReindexResponse:
        return rag_service.reindex()

    return application


app = create_app()
