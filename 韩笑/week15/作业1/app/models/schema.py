"""数据模型"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    document_id: str
    filename: str
    status: str = "pending"


class ChunkInfo(BaseModel):
    """Chunk信息"""
    chunk_id: str
    content: str
    content_type: str  # "text" or "image"
    source_file: str
    page_number: int
    embedding_id: Optional[str] = None


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    content: str
    content_type: str
    source_file: str
    page_number: int
    score: float
    image_path: Optional[str] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: List[SearchResult]
    source_files: List[str]  # 去重后的文件列表