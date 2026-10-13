"""
索引构建模块
"""

import logging
import pickle
from typing import List
from pathlib import Path

import numpy as np
import faiss
from langchain_core.documents import Document

from .native_faiss import NativeFAISSIndex

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index", embeddings=None):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = embeddings
        self.vectorstore = None
        if self.embeddings is None:
            self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
    
    def build_vector_index(self, chunks: List[Document]) -> NativeFAISSIndex:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        self.vectorstore = NativeFAISSIndex.from_documents(chunks, self.embeddings)
        
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        raise NotImplementedError("单独添加向量会破坏 BM25/原文一致性，请统一重建知识库")

    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        # An interrupted save must never leave a valid marker for incomplete files.
        (Path(self.index_save_path) / 'index_metadata.json').unlink(missing_ok=True)
        # FAISS' native file API cannot open non-ASCII Windows paths reliably.
        # Serialize the native index in memory and let Python handle the path.
        self._save_index_files()
        logger.info(f"向量索引已保存到: {self.index_save_path}")

    def _save_index_files(self):
        index_bytes = faiss.serialize_index(self.vectorstore.index).tobytes()
        path = Path(self.index_save_path)
        (path / 'index.faiss').write_bytes(index_bytes)
        with (path / 'index.pkl').open('wb') as file:
            pickle.dump(self.vectorstore.persistence_payload(), file)
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = self._load_index_files()
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def _load_index_files(self):
        path = Path(self.index_save_path)
        serialized = np.frombuffer((path / 'index.faiss').read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(serialized)
        # Only load indexes created in this trusted local project directory.
        with (path / 'index.pkl').open('rb') as file:
            payload = pickle.load(file)
        return NativeFAISSIndex.from_persistence_payload(self.embeddings, index, payload)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        
        return self.vectorstore.similarity_search(query, k=k)
