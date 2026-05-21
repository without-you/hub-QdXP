"""文本嵌入服务（DashScope text-embedding-v4 API）"""
from openai import OpenAI
from 作业1.app.core.config import settings


class EmbeddingService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.QWEN_API_KEY,
                base_url=settings.QWEN_BASE_URL,
                timeout=60.0
            )
        return self._client

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL_NAME,
            input=text,
            dimensions=settings.EMBEDDING_DIMENSION,
            encoding_format="float"
        )
        return response.data[0].embedding


embedding_service = EmbeddingService()
