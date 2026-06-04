"""
嵌入缓存模块
缓存文本到向量的转换结果，避免重复计算
"""

from typing import List, Optional, Dict, Any
import json
import hashlib
import redis


class EmbeddingCache:
    """
    嵌入缓存实现

    缓存文本的嵌入向量，避免对相同内容进行重复的嵌入计算
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "embedding_cache",
        ttl: int = 86400,  # 默认24小时
        config=None
    ):
        """
        初始化嵌入缓存

        Args:
            redis_client: Redis客户端
            prefix: 缓存键前缀
            ttl: 默认过期时间（秒）
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = ttl
        self.config = config

    def _make_key(self, text: str) -> str:
        """
        生成缓存键

        Args:
            text: 输入文本

        Returns:
            str: 缓存键
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.prefix}:{text_hash}"

    def get(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的嵌入向量

        Args:
            text: 输入文本

        Returns:
            Optional[List[float]]: 嵌入向量，如果未命中则返回None
        """
        key = self._make_key(text)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置嵌入向量缓存

        Args:
            text: 输入文本
            embedding: 嵌入向量
            ttl: 过期时间（秒），如果为None则使用默认值

        Returns:
            bool: 是否设置成功
        """
        key = self._make_key(text)
        cached_data = json.dumps(embedding)

        if ttl is None:
            ttl = self.default_ttl

        return bool(self.redis.setex(key, ttl, cached_data))

    def delete(self, text: str) -> bool:
        """
        删除缓存

        Args:
            text: 输入文本

        Returns:
            bool: 是否删除成功
        """
        key = self._make_key(text)
        return bool(self.redis.delete(key))

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            bool: 是否清空成功
        """
        keys = self.redis.keys(f"{self.prefix}:*")
        if keys:
            return bool(self.redis.delete(*keys))
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
            "default_ttl": self.default_ttl
        }

    def batch_get(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        批量获取嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[Optional[List[float]]]: 嵌入向量列表
        """
        if not texts:
            return []

        keys = [self._make_key(text) for text in texts]
        results = []

        # 使用pipeline批量获取
        with self.redis.pipeline() as pipe:
            for key in keys:
                pipe.get(key)
            cached_results = pipe.execute()

        for cached in cached_results:
            if cached:
                results.append(json.loads(cached))
            else:
                results.append(None)

        return results

    def batch_set(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        批量设置嵌入向量缓存

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            ttl: 过期时间（秒）

        Returns:
            bool: 是否全部设置成功
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts和embeddings长度必须相同")

        if not texts:
            return True

        if ttl is None:
            ttl = self.default_ttl

        # 使用pipeline批量设置
        with self.redis.pipeline() as pipe:
            for text, embedding in zip(texts, embeddings):
                key = self._make_key(text)
                cached_data = json.dumps(embedding)
                pipe.setex(key, ttl, cached_data)
            pipe.execute()

        return True