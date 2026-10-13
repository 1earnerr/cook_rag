"""
数据准备模块
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # Optional adapter for the bundled public recipe corpus; disabled by default.
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    
    CHUNKING_VERSION = "markdown-header-token-recursive-v2"

    def __init__(self, data_path: str, recipe_metadata_enabled: bool = False,
                 max_section_chars: int = 1200, section_chunk_overlap: int = 100,
                 embedding_chunk_max_tokens: int = 450, embedding_tokenizer=None):
        """
        初始化数据准备模块
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.recipe_metadata_enabled = recipe_metadata_enabled
        self.max_section_chars = max_section_chars
        self.section_chunk_overlap = section_chunk_overlap
        self.embedding_chunk_max_tokens = embedding_chunk_max_tokens
        self.embedding_tokenizer = embedding_tokenizer
        self.documents: List[Document] = []  # 完整 Markdown 文档
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path).resolve()
        if not data_path_obj.is_dir():
            raise FileNotFoundError(f"文档目录不存在: {data_path_obj}")

        for md_file in sorted(data_path_obj.rglob("*")):
            if not md_file.is_file() or md_file.suffix.lower() != '.md':
                continue
            try:
                parts = {part.lower() for part in md_file.relative_to(data_path_obj).parts}
                if parts & {'template', 'templates'} or md_file.stem.lower() in {'template', '示例菜'}:
                    logger.info("跳过模板文件: %s", md_file)
                    continue
                # 直接读取文件内容，保持Markdown格式
                with open(md_file, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                if not self._clean_content(content).strip():
                    continue

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": relative_path,
                        "document_id": document_id,
                        "parent_id": document_id,
                        "title": self._extract_title(self._clean_content(content), md_file.stem),
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                documents.append(doc)

            except (OSError, UnicodeError) as e:
                raise ValueError(f"读取文档失败，停止建库以避免静默遗漏: {md_file}") from e
        
        # 增强文档元数据
        if self.recipe_metadata_enabled:
            for doc in documents:
                self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    @staticmethod
    def _clean_content(content: str) -> str:
        # Keep raw parent text for fingerprints; remove non-text markup only for retrieval.
        content = re.sub(r'<!--[\s\S]*?-->', '', content)
        content = re.sub(r'!\[([^\]]*)\]\([^\n)]*\)', r'\1', content)
        return content.strip()

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return fallback
    
    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts
        
        # 提取菜品分类
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 食谱数据可选字段；通用 Markdown 文档不依赖这些字段。
        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级（使用正则精确匹配连续星号数量）
        content = doc.page_content
        star_match = re.search(r'★+', content)
        if star_match:
            star_count = len(star_match.group())
            difficulty_map = {5: '非常困难', 4: '困难', 3: '中等', 2: '简单', 1: '非常简单'}
            doc.metadata['difficulty'] = difficulty_map.get(star_count, '未知')
        else:
            doc.metadata['difficulty'] = '未知'

    
    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        self.parent_child_map.clear()

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题")
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 对每个文档进行Markdown分割
                md_chunks = markdown_splitter.split_text(self._clean_content(doc.page_content))

                logger.debug("文档 %s 分割成 %s 个块", doc.metadata['title'], len(md_chunks))

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.debug("文档保留为单块: %s", doc.metadata['title'])

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                expanded_chunks = []
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_section_chars,
                    chunk_overlap=self.section_chunk_overlap,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                    keep_separator=True,
                )
                for section_index, section_chunk in enumerate(md_chunks):
                    pieces = self._split_section(section_chunk, splitter)
                    for sub_index, piece in enumerate(pieces):
                        piece.metadata.update(section_chunk.metadata)
                        piece.metadata['section_index'] = section_index
                        piece.metadata['subchunk_index'] = sub_index
                        expanded_chunks.append(piece)

                for i, chunk in enumerate(expanded_chunks):
                    # 为子块分配唯一ID
                    child_id = self._stable_chunk_id(
                        parent_id,
                        f"{chunk.metadata.get('section_index', i)}:{chunk.metadata.get('subchunk_index', 0)}",
                        chunk.page_content,
                    )

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata['section'] = ' / '.join(
                        str(chunk.metadata[key]) for key in ('主标题', '二级标题', '三级标题')
                        if chunk.metadata.get(key)
                    ) or doc.metadata['title']
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "document_id": parent_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(expanded_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # Do not silently change the chunking strategy and reuse incompatible indexes.
                raise ValueError(f"Markdown 分块失败: {doc.metadata['source']}") from e

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def _split_section(self, section_chunk: Document, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
        """Keep a structured section unless its embedding tokenizer exceeds the safety limit."""
        if self._token_count(section_chunk.page_content) <= self.embedding_chunk_max_tokens:
            return [section_chunk]
        pieces = splitter.split_documents([section_chunk])
        # Character splitting is a first pass; recursively split any piece that
        # still exceeds the tokenizer budget. The separator list is finite and
        # the final empty separator guarantees progress.
        while any(self._token_count(piece.page_content) > self.embedding_chunk_max_tokens for piece in pieces):
            next_pieces = []
            for piece in pieces:
                if self._token_count(piece.page_content) <= self.embedding_chunk_max_tokens:
                    next_pieces.append(piece)
                    continue
                smaller_size = max(40, len(piece.page_content) // 2)
                smaller = RecursiveCharacterTextSplitter(
                    chunk_size=smaller_size,
                    chunk_overlap=min(self.section_chunk_overlap, max(0, smaller_size // 5)),
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                    keep_separator=True,
                ).split_documents([piece])
                if len(smaller) == 1 and smaller[0].page_content == piece.page_content:
                    # A tokenizer budget must not silently be violated.
                    raise ValueError("无法将 Markdown Section 切分到 embedding token 阈值以内")
                next_pieces.extend(smaller)
            pieces = next_pieces
        return pieces

    def _token_count(self, text: str) -> int:
        if self.embedding_tokenizer is not None:
            if hasattr(self.embedding_tokenizer, 'encode'):
                input_ids = self.embedding_tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )
            else:
                encoded = self.embedding_tokenizer(
                    text, add_special_tokens=True, truncation=False)
                input_ids = encoded.get('input_ids', encoded)
            return len(input_ids)
        # Tests and offline tools can inject a tokenizer. This conservative
        # fallback keeps the loader usable when only the corpus is inspected.
        return len(text) + 2

    @staticmethod
    def _stable_chunk_id(document_id: str, index, content: str) -> str:
        value = f"{document_id}:{index}:{content}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档
        
        Args:
            category: 菜品分类
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        
        Args:
            difficulty: 难度等级
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category')
            if category:
                categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty')
            if difficulty:
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                **doc.metadata,
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("document_id") or chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            title = doc.metadata['title']
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{title}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
