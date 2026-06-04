"""
向量索引模块
管理向量索引的创建、更新和删除
"""

from typing import List, Dict, Any, Optional
import redis
from redis.commands.search.field import VectorField, TagField, NumericField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType


class VectorIndex:
    """
    向量索引管理

    提供向量索引的创建、数据添加和管理功能
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        index_name: str = "vector_index",
        config=None
    ):
        """
        初始化向量索引

        Args:
            redis_client: Redis客户端
            index_name: 索引名称
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.index_name = index_name
        self.config = config
        self.client = redis_client.ft(index_name)

    def create(
        self,
        dimensions: int = 1536,
        distance_metric: str = "COSINE",
        vector_field_name: str = "embedding",
        **kwargs
    ) -> bool:
        """
        创建向量索引

        Args:
            dimensions: 向量维度
            distance_metric: 距离度量方式 (COSINE, L2, IP)
            vector_field_name: 向量字段名称
            **kwargs: 其他配置

        Returns:
            bool: 是否创建成功
        """
        try:
            # 定义索引字段
            fields = [
                VectorField(
                    vector_field_name,
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dimensions,
                        "DISTANCE_METRIC": distance_metric,
                        "INITIAL_CAP": kwargs.get("initial_cap", 1000),
                        "BLOCK_SIZE": kwargs.get("block_size", 1000)
                    }
                ),
                TagField("id"),
                TextField("text"),
                NumericField("timestamp"),
                TagField("category"),
                TagField("source")
            ]

            # 创建索引
            self.client.create_index(
                fields,
                definition=IndexDefinition(
                    prefix=[f"{self.index_name}:"],
                    index_type=IndexType.HASH
                )
            )
            return True

        except redis.exceptions.ResponseError as e:
            # 索引已存在
            if "Index already exists" in str(e):
                return True
            raise e

    def add(
        self,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        添加向量到索引

        Args:
            embedding: 嵌入向量
            metadata: 元数据
            id: 文档ID

        Returns:
            str: 文档ID
        """
        if id is None:
            import uuid
            id = str(uuid.uuid4())

        # 准备文档数据
        doc_data = {
            "id": id,
            "embedding": self._serialize_vector(embedding),
            "timestamp": metadata.get("timestamp", 0),
            "category": metadata.get("category", ""),
            "source": metadata.get("source", ""),
            "text": metadata.get("text", "")
        }

        # 存储到Redis
        key = f"{self.index_name}:{id}"
        self.redis.hset(key, mapping=doc_data)

        return id

    def add_batch(
        self,
        embeddings: List[List[float]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        批量添加向量到索引

        Args:
            embeddings: 嵌入向量列表
            metadata_list: 元数据列表
            ids: 文档ID列表

        Returns:
            List[str]: 文档ID列表
        """
        if metadata_list is None:
            metadata_list = [{}] * len(embeddings)

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # 使用pipeline批量添加
        with self.redis.pipeline() as pipe:
            for embedding, metadata, doc_id in zip(embeddings, metadata_list, ids):
                doc_data = {
                    "id": doc_id,
                    "embedding": self._serialize_vector(embedding),
                    "timestamp": metadata.get("timestamp", 0),
                    "category": metadata.get("category", ""),
                    "source": metadata.get("source", ""),
                    "text": metadata.get("text", "")
                }
                key = f"{self.index_name}:{doc_id}"
                pipe.hset(key, mapping=doc_data)
            pipe.execute()

        return ids

    def delete(self, id: str) -> bool:
        """
        删除文档

        Args:
            id: 文档ID

        Returns:
            bool: 是否删除成功
        """
        key = f"{self.index_name}:{id}"
        return bool(self.redis.delete(key))

    def delete_batch(self, ids: List[str]) -> int:
        """
        批量删除文档

        Args:
            ids: 文档ID列表

        Returns:
            int: 成功删除的数量
        """
        if not ids:
            return 0

        keys = [f"{self.index_name}:{id}" for id in ids]
        return self.redis.delete(*keys)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档

        Args:
            id: 文档ID

        Returns:
            Optional[Dict]: 文档数据
        """
        key = f"{self.index_name}:{id}"
        doc = self.redis.hgetall(key)
        if doc:
            return {
                "id": doc.get(b"id", b"").decode(),
                "text": doc.get(b"text", b"").decode(),
                "category": doc.get(b"category", b"").decode(),
                "source": doc.get(b"source", b"").decode(),
                "timestamp": float(doc.get(b"timestamp", 0)),
                "embedding": self._deserialize_vector(doc.get(b"embedding", b""))
            }
        return None

    def info(self) -> Dict[str, Any]:
        """
        获取索引信息

        Returns:
            Dict: 索引信息
        """
        try:
            info = self.client.info()
            return {
                "index_name": self.index_name,
                "num_docs": info.get("num_docs", 0),
                "max_doc_id": info.get("max_doc_id", 0),
                "num_terms": info.get("num_terms", 0),
                "num_records": info.get("num_records", 0)
            }
        except redis.exceptions.ResponseError:
            return {"index_name": self.index_name, "error": "索引不存在"}

    def drop(self) -> bool:
        """
        删除索引

        Returns:
            bool: 是否删除成功
        """
        try:
            self.client.dropindex()
            return True
        except redis.exceptions.ResponseError:
            return False

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """序列化向量为字节"""
        import numpy as np
        return np.array(vector, dtype=np.float32).tobytes()

    def _deserialize_vector(self, data: bytes) -> List[float]:
        """反序列化字节为向量"""
        import numpy as np
        if not data:
            return []
        return np.frombuffer(data, dtype=np.float32).tolist()