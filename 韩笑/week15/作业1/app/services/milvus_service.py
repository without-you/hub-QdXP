"""Milvus向量数据库服务"""
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from 作业1.app.core.config import settings


class MilvusService:
    def __init__(self):
        self._client = None
        self.collection_name = "multimodal_rag"
        self.embedding_dim = settings.EMBEDDING_DIMENSION

    @property
    def client(self):
        if self._client is None:
            self._client = MilvusClient(
                uri=settings.MILVUS_URI,
                token=settings.MILVUS_TOKEN
            )
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        """确保collection存在"""
        if not self._client.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="page_number", dtype=DataType.INT32),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields=fields, description="Multimodal RAG collection")
            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                dimension=self.embedding_dim
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="IP"
            )
            self._client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            self._client.load_collection(self.collection_name)

    def insert(self, id: str, content: str, content_type: str, source_file: str,
               page_number: int, embedding: list[float], image_path: str = None):
        """插入一条记录"""
        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "id": id,
                "content": content,
                "content_type": content_type,
                "source_file": source_file,
                "page_number": page_number,
                "image_path": image_path,
                "embedding": embedding
            }]
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """向量检索"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "content", "content_type", "source_file", "page_number", "image_path"]
        )
        if not results or not results[0]:
            return []
        return [
            {
                "chunk_id": r["entity"]["id"],
                "content": r["entity"]["content"],
                "content_type": r["entity"]["content_type"],
                "source_file": r["entity"]["source_file"],
                "page_number": r["entity"]["page_number"],
                "image_path": r["entity"].get("image_path"),
                "score": r["distance"]
            }
            for r in results[0]
        ]

    def delete_collection(self):
        """删除collection（测试用）"""
        if self._client and self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)


milvus_service = MilvusService()