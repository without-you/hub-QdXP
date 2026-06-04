"""
工具函数模块
提供通用的辅助函数
"""

import uuid
import hashlib
import time
from typing import List, Dict, Any, Optional
import numpy as np


def generate_id() -> str:
    """
    生成唯一ID

    Returns:
        str: UUID字符串
    """
    return str(uuid.uuid4())


def generate_hash(text: str) -> str:
    """
    生成文本的哈希值

    Args:
        text: 输入文本

    Returns:
        str: MD5哈希值
    """
    return hashlib.md5(text.encode()).hexdigest()


def calculate_similarity(
    vector1: List[float],
    vector2: List[float],
    metric: str = "cosine"
) -> float:
    """
    计算两个向量的相似度

    Args:
        vector1: 向量1
        vector2: 向量2
        metric: 相似度度量方式 (cosine, euclidean, dot_product)

    Returns:
        float: 相似度分数
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    if metric == "cosine":
        # 余弦相似度
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    elif metric == "euclidean":
        # 欧氏距离（转换为相似度）
        distance = np.linalg.norm(v1 - v2)
        return float(1.0 / (1.0 + distance))

    elif metric == "dot_product":
        # 点积
        return float(np.dot(v1, v2))

    else:
        raise ValueError(f"不支持的度量方式: {metric}")


def batch_cosine_similarity(
    query_vector: List[float],
    vectors: List[List[float]]
) -> List[float]:
    """
    批量计算余弦相似度

    Args:
        query_vector: 查询向量
        vectors: 向量列表

    Returns:
        List[float]: 相似度列表
    """
    query = np.array(query_vector)
    matrix = np.array(vectors)

    # 计算点积
    dot_products = np.dot(matrix, query)

    # 计算范数
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # 避免除零
    norms = query_norm * matrix_norms
    norms = np.where(norms == 0, 1, norms)

    return (dot_products / norms).tolist()


def normalize_vector(vector: List[float]) -> List[float]:
    """
    归一化向量

    Args:
        vector: 输入向量

    Returns:
        List[float]: 归一化后的向量
    """
    v = np.array(vector)
    norm = np.linalg.norm(v)
    if norm == 0:
        return vector
    return (v / norm).tolist()


def get_timestamp() -> float:
    """
    获取当前时间戳

    Returns:
        float: 时间戳
    """
    return time.time()


def format_timestamp(timestamp: float) -> str:
    """
    格式化时间戳

    Args:
        timestamp: 时间戳

    Returns:
        str: 格式化的时间字符串
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    截断文本

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def merge_metadata(
    base_metadata: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    合并元数据

    Args:
        base_metadata: 基础元数据
        **kwargs: 额外的元数据

    Returns:
        Dict: 合并后的元数据
    """
    merged = base_metadata.copy()
    merged.update(kwargs)
    return merged