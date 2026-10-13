import asyncio
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

from api import create_app
from rag_modules.models import Evidence, RAGAnswer
from rag_modules.generation_integration import NO_EVIDENCE
from services.rag_service import RAGService


class ASGITestClient:
    """Small synchronous facade over HTTPX ASGITransport for endpoint tests."""

    def __init__(self, application):
        self.application = application

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.application)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


def evidence(number=1):
    return Evidence(
        evidence_id=f"S{number}",
        chunk_id=f"chunk-{number}",
        document_id=f"document-{number}",
        title="恢复手册",
        section="服务恢复",
        source="manual/recovery.md",
        content="执行 restore --safe。",
    )


class FakeSystem:
    def __init__(self):
        self.index_module = SimpleNamespace(embeddings=object(), vectorstore=object())
        self.generation_module = object()
        self.retrieval_module = object()
        self.data_module = SimpleNamespace(documents=[object(), object()], chunks=[object()] * 3)
        self.index_metadata = {"schema_version": 2}
        self.answer = RAGAnswer("执行 restore --safe [S1]", [evidence()])
        self.ask_calls = []
        self.reindex_calls = []
        self.ask_error = None
        self.reindex_error = None

    def ask_question(self, query, stream=False, filters=None):
        self.ask_calls.append((query, stream, filters))
        if self.ask_error:
            raise self.ask_error
        return self.answer

    def initialize_system(self):
        raise AssertionError("ready test system must not be reinitialized")

    def build_knowledge_base(self, force=False):
        self.reindex_calls.append(force)
        if self.reindex_error:
            raise self.reindex_error


def client_and_system():
    system = FakeSystem()
    application = create_app(RAGService(system), initialize_on_startup=False)
    return ASGITestClient(application), system


def test_health_reports_real_service_state():
    client, _ = client_and_system()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_loaded": True,
        "index_loaded": True,
        "documents": 2,
        "chunks": 3,
        "detail": None,
    }


def test_health_reports_degraded_state_when_core_is_not_loaded():
    _, system = client_and_system()
    system.index_module = None
    system.retrieval_module = None
    system.data_module = None
    service = RAGService(system)
    service.last_initialization_error = "RAG initialization failed"
    client = ASGITestClient(create_app(service, initialize_on_startup=False))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "embedding_loaded": False,
        "index_loaded": False,
        "documents": 0,
        "chunks": 0,
        "detail": "RAG initialization failed",
    }


def test_chat_response_schema_sources_and_core_delegation():
    client, system = client_and_system()
    response = client.post("/api/v1/chat", json={"query": "  如何恢复？  "})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "执行 restore --safe [S1]",
        "sources": [{
            "source_id": "S1",
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "title": "恢复手册",
            "section": "服务恢复",
            "source": "manual/recovery.md",
        }],
        "retrieval": {"mode": "hybrid", "count": 1},
    }
    assert system.ask_calls == [("如何恢复？", False, None)]


def test_chat_passes_generic_filters_without_changing_retrieval():
    client, system = client_and_system()
    filters = {"source": ["a.md", "b.md"], "published": True}
    response = client.post("/api/v1/chat", json={"query": "恢复", "filters": filters})
    assert response.status_code == 200
    assert system.ask_calls == [("恢复", False, filters)]


def test_empty_query_and_invalid_filters_are_rejected():
    client, system = client_and_system()
    assert client.post("/api/v1/chat", json={"query": "   "}).status_code == 422
    assert client.post("/api/v1/chat", json={"query": "问题", "filters": []}).status_code == 422
    assert client.post("/api/v1/chat", json={"query": "问题", "filters": {"source": []}}).status_code == 422
    assert system.ask_calls == []


def test_no_evidence_is_a_safe_success_without_sources():
    client, system = client_and_system()
    system.answer = RAGAnswer(NO_EVIDENCE, [])
    response = client.post("/api/v1/chat", json={"query": "不存在的问题"})
    assert response.status_code == 200
    assert response.json()["answer"] == NO_EVIDENCE
    assert response.json()["sources"] == []
    assert response.json()["retrieval"] == {"mode": "hybrid", "count": 0}


def test_external_llm_error_is_sanitized():
    client, system = client_and_system()
    system.ask_error = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    response = client.post("/api/v1/chat", json={"query": "恢复"})
    assert response.status_code == 502
    assert response.json() == {"detail": "External LLM invocation failed"}
    assert "example.invalid" not in response.text


def test_uninitialized_and_retrieval_errors_use_distinct_statuses():
    client, system = client_and_system()
    system.retrieval_module = None
    response = client.post("/api/v1/chat", json={"query": "恢复"})
    assert response.status_code == 503
    system.retrieval_module = object()
    system.ask_error = RuntimeError("private retrieval detail")
    response = client.post("/api/v1/chat", json={"query": "恢复"})
    assert response.status_code == 500
    assert response.json() == {"detail": "RAG request failed"}
    assert "private retrieval detail" not in response.text


def test_reindex_success_returns_current_counts_and_schema():
    client, system = client_and_system()
    response = client.post("/api/v1/reindex")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["documents"] == 2 and body["chunks"] == 3
    assert body["index_schema_version"] == 2
    assert body["elapsed_seconds"] >= 0
    assert system.reindex_calls == [True]


def test_reindex_failure_is_sanitized():
    client, system = client_and_system()
    system.reindex_error = OSError("private disk path")
    response = client.post("/api/v1/reindex")
    assert response.status_code == 500
    assert response.json() == {"detail": "RAG reindex failed"}
    assert "private disk path" not in response.text
