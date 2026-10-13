"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass, field
import os
from typing import Dict, Any
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = str(APP_DIR.parent.parent / "data" / "QAR" / "cook")
    index_save_path: str = str(APP_DIR / "vector_index")
    recipe_metadata_enabled: bool = False

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))

    # 检索配置
    top_k: int = 3
    vector_fetch_k: int = 20
    bm25_fetch_k: int = 20
    final_k: int = 5
    rrf_k: int = 60
    max_section_chars: int = 1200
    section_chunk_overlap: int = 100
    embedding_chunk_max_tokens: int = 450
    query_rewrite_enabled: bool = False
    retrieval_relevance_threshold: float | None = 0.45
    context_max_chars: int = 2000

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @property
    def index_metadata_path(self) -> str:
        return str(Path(self.index_save_path) / "index_metadata.json")

    def __post_init__(self):
        if min(self.top_k, self.vector_fetch_k, self.bm25_fetch_k, self.final_k,
               self.rrf_k, self.max_section_chars, self.embedding_chunk_max_tokens) < 1 or self.section_chunk_overlap < 0:
            raise ValueError("检索、分块和上下文配置必须有效")
        if (self.retrieval_relevance_threshold is not None
                and not 0 <= self.retrieval_relevance_threshold <= 1):
            raise ValueError("retrieval_relevance_threshold 必须在 0 到 1 之间或为 None")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
