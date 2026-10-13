from config import RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule


def test_public_corpus_has_no_templates_and_traceable_evidence():
    module = DataPreparationModule(RAGConfig().data_path)
    documents = module.load_documents()
    chunks = module.chunk_documents()
    assert documents and chunks
    assert not any('template/' in doc.metadata['source'] for doc in documents)
    assert len({c.metadata['chunk_id'] for c in chunks}) == len(chunks)
    assert all({'chunk_id', 'document_id', 'title', 'source', 'section'} <= c.metadata.keys() for c in chunks)
    long_doc_ids = {d.metadata['document_id'] for d in documents if len(d.page_content) > 2000}
    assert long_doc_ids
    generation = GenerationIntegrationModule()
    for document_id in long_doc_ids:
        hits = [c for c in chunks if c.metadata['document_id'] == document_id]
        context, sources = generation._build_context(hits)
        assert context and sources and len(context) <= 2000
        assert all(s.document_id == document_id for s in sources)
