import json

import pytest
from langchain_core.documents import Document

from evaluation.evaluate_retrieval import (
    aggregate,
    analyze_failures,
    describe_similarities,
    load_questions,
    query_metrics,
    unique_documents,
)


def test_unique_documents_preserves_first_document_rank():
    docs = [
        Document("a1", metadata={"document_id": "a"}),
        Document("a2", metadata={"document_id": "a"}),
        Document("b", metadata={"document_id": "b"}),
    ]
    assert [doc.metadata["document_id"] for doc in unique_documents(docs)] == ["a", "b"]


def test_document_level_recall_and_mrr():
    metrics = query_metrics(["noise", "gold-b", "gold-a"], {"gold-a", "gold-b"})
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["relevant_document_ranks"] == {"gold-a": 3, "gold-b": 2}


def test_aggregate_excludes_no_answer_rows():
    rows = [
        {"metrics": {"recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.5}, "latency_ms": 2.0},
        {"metrics": None, "latency_ms": 999.0},
    ]
    result = aggregate(rows)
    assert result == {
        "queries": 1,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "mrr": 0.5,
        "mean_latency_ms": 2.0,
    }


def test_failure_categories_keep_hybrid_regressions_and_repairs():
    def row(query_id, mode, recall, mrr):
        return {
            "query_id": query_id,
            "mode": mode,
            "metrics": {"recall_at_5": recall, "mrr": mrr},
        }

    rows = [
        row("repair", "vector", 0.0, 0.0), row("repair", "bm25", 1.0, 1.0),
        row("repair", "hybrid", 1.0, 1.0),
        row("worse", "vector", 1.0, 1.0), row("worse", "bm25", 1.0, 0.5),
        row("worse", "hybrid", 1.0, 0.5),
        row("miss", "vector", 0.0, 0.0), row("miss", "bm25", 0.0, 0.0),
        row("miss", "hybrid", 0.0, 0.0),
    ]
    result = analyze_failures(rows)
    assert result["bm25_success_vector_failure"] == ["repair"]
    assert result["hybrid_repairs_single_route_failure"] == ["repair"]
    assert result["hybrid_worse_than_best_single_route"] == ["worse"]
    assert result["all_modes_failure"] == ["miss"]


def test_similarity_distribution_summary():
    result = describe_similarities([0.2, 0.4, 0.6, 0.8])
    assert result == {
        "count": 4,
        "min": 0.2,
        "p25": pytest.approx(0.35),
        "median": pytest.approx(0.5),
        "mean": pytest.approx(0.5),
        "p75": pytest.approx(0.65),
        "max": 0.8,
    }


def test_dataset_validation_rejects_answerable_question_without_gold(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({
        "id": "q1", "query": "question", "query_type": "semantic",
        "relevant_documents": [], "filters": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="needs document-level relevance"):
        load_questions(path)
