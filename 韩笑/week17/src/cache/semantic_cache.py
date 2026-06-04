"""
语义缓存模块
基于语义相似度的缓存，用于减少LLM调用
"""

from typing import Any, Dict, Optional, List, Callable
import json
import hashlib
import numpy as np
import redis


class SemanticCache:
    """
    语义缓存实现

    通过计算查询的语义相似度来命中缓存，避免重复的LLM调用
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "semantic_cache",
        similarity_threshold: float = 0.95,
        embedding_func: Optional[Callable] = None,
        config=None
    ):
        """
        初始化语义缓存

        Args:
            redis_client: Redis客户端
            prefix: 缓存键前缀
            similarity_threshold: 相似度阈值
            embedding_func: 嵌入向量生成函数
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.prefix = prefix
        self.similarity_threshold = similarity_threshold
        self.embedding_func = embedding_func
        self.config = config

    def _make_key(self, query: str) -> str:
        """
        生成缓存键

        Args:
            query: 查询文本

        Returns:
            str: 缓存键
        """
        # 对于精确匹配，使用哈希
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.prefix}:exact:{query_hash}"

    def _make_embedding_key(self, query: str) -> str:
        """生成嵌入向量的缓存键"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.prefix}:embedding:{query_hash}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的响应

        首先尝试精确匹配，如果未命中则进行语义搜索

        Args:
            query: 查询文本

        Returns:
            Optional[Dict]: 缓存的响应数据
        """
        # 尝试精确匹配
        exact_key = self._make_key(query)
        cached = self.redis.get(exact_key)
        if cached:
            return json.loads(cached)

        # 语义相似度匹配
        if self.embedding_func is None:
            return None

        try:
            # 获取查询的嵌入向量
            query_embedding = self.embedding_func(query)
            query_vector = np.array(query_embedding, dtype=np.float32)

            # 获取所有缓存条目的嵌入向量
            pattern = f"{self.prefix}:embedding:*"
            embedding_keys = self.redis.keys(pattern)

            if not embedding_keys:
                return None

            best_match = None
            best_similarity = 0.0

            for emb_key in embedding_keys:
                # 获取缓存的嵌入向量
                stored_embedding = self.redis.get(emb_key)
                if not stored_embedding:
                    continue

                try:
                    stored_vector = np.array(
                        json.loads(stored_embedding), dtype=np.float32
                    )
                except (json.JSONDecodeError, ValueError):
                    continue

                # 计算余弦相似度
                dot_product = np.dot(query_vector, stored_vector)
                norm1 = np.linalg.norm(query_vector)
                norm2 = np.linalg.norm(stored_vector)

                if norm1 == 0 or norm2 == 0:
                    continue

                similarity = float(dot_product / (norm1 * norm2))

                if similarity > best_similarity:
                    best_similarity = similarity
                    # 获取对应的缓存响应
                    cache_key = emb_key.decode().replace(":embedding:", ":exact:")
                    cached_response = self.redis.get(cache_key)
                    if cached_response:
                        best_match = json.loads(cached_response)

            # 如果相似度超过阈值，返回缓存的结果
            if best_match and best_similarity >= self.similarity_threshold:
                return best_match

        except Exception as e:
            # 语义匹配失败，返回None
            pass

        return None

    def set(
        self,
        query: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存

        Args:
            query: 查询文本
            response: 响应数据
            ttl: 过期时间（秒）

        Returns:
            bool: 是否设置成功
        """
        exact_key = self._make_key(query)
        cached_data = json.dumps(response)

        # 存储精确匹配缓存
        if ttl:
            result = bool(self.redis.setex(exact_key, ttl, cached_data))
        else:
            result = bool(self.redis.set(exact_key, cached_data))

        # 如果有嵌入函数，同时存储嵌入向量用于语义匹配
        if self.embedding_func and result:
            try:
                embedding = self.embedding_func(query)
                embedding_key = self._make_embedding_key(query)
                embedding_data = json.dumps(embedding)

                if ttl:
                    self.redis.setex(embedding_key, ttl, embedding_data)
                else:
                    self.redis.set(embedding_key, embedding_data)
            except Exception:
                # 嵌入存储失败不影响主缓存
                pass

        return result

    def delete(self, query: str) -> bool:
        """
        删除缓存

        Args:
            query: 查询文本

        Returns:
            bool: 是否删除成功
        """
        exact_key = self._make_key(query)
        return bool(self.redis.delete(exact_key))

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            bool: 是否清空成功
        """
        # 清除精确匹配缓存
        exact_keys = self.redis.keys(f"{self.prefix}:exact:*")
        # 清除嵌入向量缓存
        embedding_keys = self.redis.keys(f"{self.prefix}:embedding:*")

        all_keys = exact_keys + embedding_keys
        if all_keys:
            return bool(self.redis.delete(*all_keys))
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict: 统计信息
        """
        keys = self.redis.keys(f"{self.prefix}:*")
        return {
            "total_entries": len(keys),
            "prefix": self.prefix,
            "similarity_threshold": self.similarity_threshold
        }