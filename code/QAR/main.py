"""Synchronous CLI orchestration for the Markdown knowledge base."""

import logging
from pathlib import Path


from dotenv import load_dotenv
from config import APP_DIR, RAGConfig
from rag_modules import (DataPreparationModule, IndexConstructionModule,
                         RetrievalOptimizationModule, GenerationIntegrationModule)
from rag_modules.index_metadata import (
    build_index_metadata, load_index_metadata, metadata_matches, save_index_metadata,
    index_file_hashes, index_files_match,
)

load_dotenv(APP_DIR / '.env')
logger = logging.getLogger(__name__)


class MarkdownRAGSystem:
    """One in-process corpus; rebuilding is explicit, synchronous and whole-corpus."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.index_metadata = None
        if not Path(self.config.data_path).is_dir():
            raise FileNotFoundError(f'数据目录不存在: {self.config.data_path}')

    def initialize_system(self):
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model, index_save_path=self.config.index_save_path)
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model, temperature=self.config.temperature,
            max_tokens=self.config.max_tokens, context_max_chars=self.config.context_max_chars)

    def build_knowledge_base(self, force: bool = False):
        if self.index_module is None:
            raise ValueError('请先初始化索引模块')
        if self.index_module.model_name != self.config.embedding_model:
            raise ValueError('Embedding 配置已变更，请重新初始化索引模块再建库')
        # Read once: fingerprint, FAISS and BM25 all refer to these in-memory documents.
        data = DataPreparationModule(
            self.config.data_path,
            self.config.recipe_metadata_enabled,
            self.config.max_section_chars,
            self.config.section_chunk_overlap,
            self.config.embedding_chunk_max_tokens,
            self._load_embedding_tokenizer(),
        )
        data.load_documents()
        chunks = data.chunk_documents()
        if not chunks:
            raise ValueError('没有有效 Markdown 内容，未替换当前知识库')
        expected = build_index_metadata(
            records=[(doc.metadata['source'], doc.page_content) for doc in data.documents],
            embedding_model=self.config.embedding_model,
            chunking_version=f'{data.CHUNKING_VERSION}:{data.max_section_chars}:{data.section_chunk_overlap}:{data.embedding_chunk_max_tokens}',
            document_count=len(data.documents), chunk_count=len(chunks), chunks=chunks,
            recipe_metadata_enabled=self.config.recipe_metadata_enabled)
        stored = load_index_metadata(self.config.index_metadata_path)
        vectorstore = None
        if (not force and metadata_matches(stored, expected)
                and index_files_match(self.config.index_save_path, stored)):
            vectorstore = self.index_module.load_index()
        if vectorstore is None:
            logger.info('索引缺失、损坏或语料/配置变化，重建知识库')
            vectorstore = self.index_module.build_vector_index(chunks)
            # Build both retrievers before committing the success marker.
            retrieval = RetrievalOptimizationModule(
                vectorstore, chunks, self.config.vector_fetch_k, self.config.bm25_fetch_k,
                self.config.final_k, self.config.rrf_k)
            self.index_module.save_index()
            expected['index_files'] = index_file_hashes(self.config.index_save_path)
            save_index_metadata(self.config.index_metadata_path, expected)
        else:
            logger.info('索引版本及文件校验通过，复用 FAISS')
            retrieval = RetrievalOptimizationModule(
                vectorstore, chunks, self.config.vector_fetch_k, self.config.bm25_fetch_k,
                self.config.final_k, self.config.rrf_k)
            expected['index_files'] = stored['index_files']
        self.data_module = data
        self.retrieval_module = retrieval
        self.index_metadata = expected
        logger.info('知识库就绪: %s 文档 / %s 子块', len(data.documents), len(chunks))

    def _load_embedding_tokenizer(self):
        client = getattr(getattr(self.index_module, 'embeddings', None), '_client', None)
        tokenizer = getattr(client, 'tokenizer', None)
        if tokenizer is not None:
            logger.info('复用 Embedding 模型 tokenizer: max_length=%s', tokenizer.model_max_length)
            return tokenizer
        try:
            from transformers import AutoTokenizer
            # The embedding model download is managed by the embedding layer;
            # tokenizer loading must not trigger a second network request.
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model, local_files_only=True)
            logger.info('Embedding tokenizer loaded: max_length=%s', tokenizer.model_max_length)
            return tokenizer
        except Exception as exc:
            logger.warning('Embedding tokenizer 加载失败，使用保守字符长度兜底: %s', exc)
            return None

    def ask_question(self, question: str, stream: bool = False, filters: dict | None = None):
        if self.retrieval_module is None or self.generation_module is None:
            raise ValueError('请先初始化系统并构建知识库')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('问题不能为空')
        question = question.strip()
        query = question
        if self.config.query_rewrite_enabled:
            try:
                rewritten = self.generation_module.query_rewrite(question)
                if isinstance(rewritten, str) and rewritten.strip():
                    query = rewritten.strip()
            except Exception as exc:
                logger.warning('Query Rewrite 失败，回退原 Query: %s', exc)
        if filters:
            chunks = self.retrieval_module.retrieve(query, filters=filters, mode='hybrid', final_k=self.config.final_k)
        else:
            chunks = self.retrieval_module.retrieve(query, mode='hybrid', final_k=self.config.final_k)
        threshold = self.config.retrieval_relevance_threshold
        if chunks and threshold is not None:
            similarities = self.index_module.vectorstore.semantic_similarities(query, chunks)
            maximum = max(similarities, default=-1.0)
            if maximum < threshold:
                logger.info(
                    'Retrieval relevance gate rejected evidence: max_similarity=%.4f threshold=%.4f',
                    maximum, threshold)
                chunks = []
        # Preserve the actual hit and its identity, instead of replacing it with an entire parent.
        if stream:
            return self.generation_module.generate_basic_answer_stream(question, chunks)
        return self.generation_module.generate_basic_answer(question, chunks)

    @staticmethod
    def _print_sources(sources):
        for item in sources:
            print(f'[{item.evidence_id}] {item.title} / {item.section} ({item.source})')

    def run_interactive(self):
        self.initialize_system()
        self.build_knowledge_base()
        print('QAR Markdown 知识库问答（输入 exit 退出）')
        while True:
            try:
                question = input('\n问题: ').strip()
                if question.lower() in {'', 'exit', 'quit', '退出'}:
                    break
                stream = input('流式输出? [Y/n]: ').strip().lower() != 'n'
                if stream:
                    sources = []
                    for event in self.ask_question(question, stream=True):
                        if event.event == 'sources':
                            sources = event.sources
                        elif event.event == 'token':
                            print(event.text, end='', flush=True)
                    print()
                    self._print_sources(sources)
                else:
                    result = self.ask_question(question)
                    print(result.answer)
                    self._print_sources(result.sources)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception('处理问题失败')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    MarkdownRAGSystem().run_interactive()


if __name__ == '__main__':
    main()
