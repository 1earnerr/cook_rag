"""Real BGE + FAISS smoke test for the bundled public Markdown corpus.

This script builds a temporary index, saves and reloads it, and performs one
retrieval. It does not call the answer-generation model or claim retrieval
quality.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import APP_DIR, RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


def main() -> None:
    config = RAGConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        config.embedding_model, local_files_only=True)
    data = DataPreparationModule(
        config.data_path,
        config.recipe_metadata_enabled,
        config.max_section_chars,
        config.section_chunk_overlap,
        config.embedding_chunk_max_tokens,
        tokenizer,
    )
    documents = data.load_documents()
    chunks = data.chunk_documents()
    max_chunk_tokens = max(data._token_count(chunk.page_content) for chunk in chunks)
    print(
        f"corpus documents={len(documents)} chunks={len(chunks)} "
        f"max_chunk_tokens={max_chunk_tokens}"
    )

    with tempfile.TemporaryDirectory(prefix=".qar-faiss-smoke-", dir=APP_DIR) as index_dir:
        builder = IndexConstructionModule(
            model_name=config.embedding_model, index_save_path=index_dir)
        query_vector = builder.embeddings.embed_query("这是一个测试问题")
        document_vectors = builder.embeddings.embed_documents(
            ["这是第一段测试文本。", "这是第二段测试文本。"])
        print(
            f"embedding query_dimension={len(query_vector)} "
            f"document_count={len(document_vectors)}"
        )

        builder.build_vector_index(chunks)
        builder.save_index()
        saved_files = sorted(path.name for path in Path(index_dir).iterdir())
        print(f"faiss saved_files={saved_files}")

        loader = IndexConstructionModule(
            model_name=config.embedding_model,
            index_save_path=index_dir,
            embeddings=builder.embeddings,
        )
        loaded = loader.load_index()
        if loaded is None:
            raise RuntimeError("FAISS index reload failed")
        hits = loader.similarity_search("糖醋鲤鱼需要哪些原料？", k=5)
        if not hits:
            raise RuntimeError("Real FAISS retrieval returned no chunks")
        print(f"faiss reload=True hits={len(hits)}")
        for rank, hit in enumerate(hits, 1):
            print(
                f"hit[{rank}] chunk_id={hit.metadata.get('chunk_id')} "
                f"source={hit.metadata.get('source')} "
                f"section={hit.metadata.get('section')}"
            )

        retriever = RetrievalOptimizationModule(
            loaded,
            chunks,
            config.vector_fetch_k,
            config.bm25_fetch_k,
            config.final_k,
            config.rrf_k,
        )
        results = {
            mode: retriever.retrieve(
                "糖醋鲤鱼需要哪些原料？", mode=mode, final_k=config.final_k)
            for mode in ("vector", "bm25", "hybrid")
        }
        if not all(results.values()):
            raise RuntimeError("Vector/BM25/Hybrid smoke retrieval must all return chunks")
        if not all("rrf_score" in item.metadata for item in results["hybrid"]):
            raise RuntimeError("Hybrid results lost RRF metadata")
        context, sources = GenerationIntegrationModule()._build_context(results["hybrid"])
        if not sources or not all(source.render() in context for source in sources):
            raise RuntimeError("Evidence/Sources do not match the prompt context")
        print(
            "retrieval "
            + " ".join(f"{mode}_hits={len(items)}" for mode, items in results.items())
        )
        print(
            f"evidence sources={len(sources)} "
            f"all_sources_in_prompt_context={all(source.render() in context for source in sources)}"
        )


if __name__ == "__main__":
    main()
