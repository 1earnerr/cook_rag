"""Reproducible document-level evaluation of the frozen retrieval baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Iterable

QAR_DIR = Path(__file__).resolve().parent.parent
if str(QAR_DIR) not in sys.path:
    sys.path.insert(0, str(QAR_DIR))

from config import APP_DIR, RAGConfig
from main import MarkdownRAGSystem


MODES = ("vector", "bm25", "hybrid")


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        questions.append(item)
    validate_questions(questions)
    return questions


def validate_questions(questions: list[dict]) -> None:
    required = {"id", "query", "query_type", "relevant_documents", "filters"}
    ids = [item.get("id") for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    for item in questions:
        missing = required - set(item)
        if missing:
            raise ValueError(f"{item.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"{item['id']} query must be non-empty")
        if not isinstance(item["relevant_documents"], list) or not isinstance(item["filters"], dict):
            raise ValueError(f"{item['id']} has invalid relevance or filters")
        if item["query_type"] == "no_answer" and item["relevant_documents"]:
            raise ValueError(f"{item['id']} no-answer query must not have relevant documents")
        if item["query_type"] != "no_answer" and not item["relevant_documents"]:
            raise ValueError(f"{item['id']} answerable query needs document-level relevance")


def unique_documents(documents: Iterable) -> list:
    result = []
    seen = set()
    for document in documents:
        document_id = document.metadata["document_id"]
        if document_id in seen:
            continue
        seen.add(document_id)
        result.append(document)
    return result


def query_metrics(top_document_ids: list[str], relevant_ids: set[str]) -> dict[str, float | dict]:
    ranks = {
        document_id: (top_document_ids.index(document_id) + 1 if document_id in top_document_ids else None)
        for document_id in sorted(relevant_ids)
    }
    first_rank = min((rank for rank in ranks.values() if rank is not None), default=None)
    return {
        "recall_at_3": len(relevant_ids.intersection(top_document_ids[:3])) / len(relevant_ids),
        "recall_at_5": len(relevant_ids.intersection(top_document_ids[:5])) / len(relevant_ids),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "relevant_document_ranks": ranks,
    }


def aggregate(rows: list[dict]) -> dict[str, float | int]:
    scored = [row for row in rows if row["metrics"] is not None]
    if not scored:
        return {"queries": 0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0,
                "mean_latency_ms": 0.0}
    return {
        "queries": len(scored),
        "recall_at_3": mean(row["metrics"]["recall_at_3"] for row in scored),
        "recall_at_5": mean(row["metrics"]["recall_at_5"] for row in scored),
        "mrr": mean(row["metrics"]["mrr"] for row in scored),
        "mean_latency_ms": mean(row["latency_ms"] for row in scored),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe_similarities(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": median(values),
        "mean": mean(values),
        "p75": _percentile(values, 0.75),
        "max": max(values),
    }


def relevance_analysis(rows: list[dict]) -> dict:
    hybrid = [row for row in rows if row["mode"] == "hybrid"]
    answer = [row for row in hybrid if row["query_type"] != "no_answer"]
    no_answer = [row for row in hybrid if row["query_type"] == "no_answer"]
    answer_values = [row["max_semantic_similarity"] for row in answer
                     if row["max_semantic_similarity"] is not None]
    no_answer_values = [row["max_semantic_similarity"] for row in no_answer
                        if row["max_semantic_similarity"] is not None]
    thresholds = [round(0.30 + step * 0.05, 2) for step in range(11)]
    if answer_values and no_answer_values:
        midpoint = round((min(answer_values) + max(no_answer_values)) / 2, 4)
        thresholds = sorted(set(thresholds + [midpoint]))
    candidates = [{
        "threshold": threshold,
        "answer_false_reject": sum(value < threshold for value in answer_values),
        "no_answer_false_accept": sum(value >= threshold for value in no_answer_values),
    } for threshold in thresholds]
    return {
        "score": "maximum cosine similarity between the query and final Hybrid chunks",
        "answer": describe_similarities(answer_values),
        "no_answer": describe_similarities(no_answer_values),
        "answer_scores": {row["query_id"]: row["max_semantic_similarity"] for row in answer},
        "no_answer_scores": {row["query_id"]: row["max_semantic_similarity"] for row in no_answer},
        "threshold_candidates": candidates,
    }


def analyze_failures(rows: list[dict]) -> dict[str, list[str]]:
    by_query = defaultdict(dict)
    for row in rows:
        if row["metrics"] is not None:
            by_query[row["query_id"]][row["mode"]] = row

    def hit(row):
        return row["metrics"]["recall_at_5"] > 0

    categories = defaultdict(list)
    for query_id, modes in by_query.items():
        vector, bm25, hybrid = (modes[mode] for mode in MODES)
        vector_hit, bm25_hit, hybrid_hit = map(hit, (vector, bm25, hybrid))
        if vector_hit and not bm25_hit:
            categories["vector_success_bm25_failure"].append(query_id)
        if bm25_hit and not vector_hit:
            categories["bm25_success_vector_failure"].append(query_id)
        if hybrid_hit and (not vector_hit or not bm25_hit):
            categories["hybrid_repairs_single_route_failure"].append(query_id)
        if hybrid["metrics"]["mrr"] < max(vector["metrics"]["mrr"], bm25["metrics"]["mrr"]):
            categories["hybrid_worse_than_best_single_route"].append(query_id)
        if not vector_hit and not bm25_hit and not hybrid_hit:
            categories["all_modes_failure"].append(query_id)
    return {key: value for key, value in categories.items()}


def build_summary(questions: list[dict], rows: list[dict]) -> dict:
    overall = {mode: aggregate([row for row in rows if row["mode"] == mode]) for mode in MODES}
    answerable_types = sorted({item["query_type"] for item in questions if item["query_type"] != "no_answer"})
    by_type = {
        query_type: {
            mode: aggregate([row for row in rows
                             if row["mode"] == mode and row["query_type"] == query_type])
            for mode in MODES
        }
        for query_type in answerable_types
    }
    no_answer = [row for row in rows if row["query_type"] == "no_answer"]
    return {
        "dataset": {
            "total": len(questions),
            "answerable": sum(item["query_type"] != "no_answer" for item in questions),
            "no_answer": sum(item["query_type"] == "no_answer" for item in questions),
            "query_types": dict(sorted(Counter(item["query_type"] for item in questions).items())),
        },
        "parameters": {},
        "overall": overall,
        "by_query_type": by_type,
        "failure_categories": analyze_failures(rows),
        "no_answer_results": no_answer,
        "relevance_analysis": relevance_analysis(rows),
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Frozen Retrieval Baseline Results",
        "",
        "No-answer queries are excluded from Recall@K and MRR.",
        "",
        "## Overall",
        "",
        "| Method | Recall@3 | Recall@5 | MRR | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["overall"][mode]
        lines.append(
            f"| {mode} | {item['recall_at_3']:.4f} | {item['recall_at_5']:.4f} | "
            f"{item['mrr']:.4f} | {item['mean_latency_ms']:.3f} |"
        )
    lines.extend(["", "## By query type", ""])
    for query_type, modes in summary["by_query_type"].items():
        lines.extend([
            f"### {query_type}", "",
            "| Method | Queries | Recall@3 | Recall@5 | MRR |",
            "|---|---:|---:|---:|---:|",
        ])
        for mode in MODES:
            item = modes[mode]
            lines.append(
                f"| {mode} | {item['queries']} | {item['recall_at_3']:.4f} | "
                f"{item['recall_at_5']:.4f} | {item['mrr']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Failure categories", ""])
    for name, query_ids in summary["failure_categories"].items():
        lines.append(f"- **{name}**: {', '.join(query_ids) if query_ids else 'none'}")
    relevance = summary["relevance_analysis"]
    lines.extend([
        "", "## Relevance similarity", "",
        "The score is the maximum cosine similarity between the query and final Hybrid chunks.",
        "",
        "| Group | Count | Min | P25 | Median | Mean | P75 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("answer", "no_answer"):
        item = relevance[name]
        lines.append(
            f"| {name} | {item['count']} | {item['min']:.4f} | {item['p25']:.4f} | "
            f"{item['median']:.4f} | {item['mean']:.4f} | {item['p75']:.4f} | {item['max']:.4f} |"
        )
    lines.extend([
        "", "### Threshold candidates", "",
        "| Threshold | Answer false reject | No-answer false accept |",
        "|---:|---:|---:|",
    ])
    for candidate in relevance["threshold_candidates"]:
        lines.append(
            f"| {candidate['threshold']:.4f} | {candidate['answer_false_reject']} | "
            f"{candidate['no_answer_false_accept']} |"
        )
    lines.extend(["", "## No-answer behavior", ""])
    for row in summary["no_answer_results"]:
        lines.append(
            f"- {row['query_id']} / {row['mode']}: "
            + ", ".join(row["top_sources"])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(questions_path: Path, output_dir: Path) -> tuple[list[dict], dict]:
    questions = load_questions(questions_path)
    config = RAGConfig()
    system = MarkdownRAGSystem(config)
    system.initialize_system()
    system.build_knowledge_base(force=False)

    source_to_document_id = {
        document.metadata["source"]: document.metadata["document_id"]
        for document in system.data_module.documents
    }
    for item in questions:
        unknown = set(item["relevant_documents"]) - set(source_to_document_id)
        if unknown:
            raise ValueError(f"{item['id']} references missing corpus documents: {sorted(unknown)}")

    rows = []
    for item in questions:
        relevant_ids = {source_to_document_id[source] for source in item["relevant_documents"]}
        for mode in MODES:
            started = perf_counter()
            chunks = system.retrieval_module.retrieve(
                item["query"], filters=item["filters"], mode=mode, final_k=config.final_k)
            latency_ms = (perf_counter() - started) * 1000
            documents = unique_documents(chunks)
            top_document_ids = [doc.metadata["document_id"] for doc in documents]
            row = {
                "query_id": item["id"],
                "query": item["query"],
                "query_type": item["query_type"],
                "mode": mode,
                "filters": item["filters"],
                "relevant_sources": item["relevant_documents"],
                "relevant_document_ids": sorted(relevant_ids),
                "top_document_ids": top_document_ids,
                "top_sources": [doc.metadata["source"] for doc in documents],
                "latency_ms": latency_ms,
                "max_semantic_similarity": None,
                "top3_semantic_similarities": [],
                "metrics": None if item["query_type"] == "no_answer"
                else query_metrics(top_document_ids, relevant_ids),
            }
            if mode == "hybrid":
                similarities = system.index_module.vectorstore.semantic_similarities(
                    item["query"], chunks)
                ordered_similarities = sorted(similarities, reverse=True)
                row["max_semantic_similarity"] = (
                    ordered_similarities[0] if ordered_similarities else None)
                row["top3_semantic_similarities"] = ordered_similarities[:3]
            rows.append(row)

    summary = build_summary(questions, rows)
    summary["parameters"] = {
        "embedding_model": config.embedding_model,
        "vector_fetch_k": config.vector_fetch_k,
        "bm25_fetch_k": config.bm25_fetch_k,
        "final_k": config.final_k,
        "rrf_k": config.rrf_k,
        "query_rewrite_enabled": False,
        "retrieval_relevance_threshold": config.retrieval_relevance_threshold,
        "document_count": len(system.data_module.documents),
        "chunk_count": len(system.data_module.chunks),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "retrieval_results.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=APP_DIR / "evaluation" / "questions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "evaluation" / "results")
    args = parser.parse_args()
    _, summary = run_evaluation(args.questions, args.output_dir)
    print(json.dumps(summary["dataset"], ensure_ascii=False))
    for mode, metrics in summary["overall"].items():
        print(mode, json.dumps(metrics, ensure_ascii=False))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
