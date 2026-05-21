"""项目配置"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "多模态RAG系统"
    DEBUG: bool = True

    # Milvus配置（Zilliz Cloud）
    MILVUS_URI: str = "https://in03-64b3b51a29e4049.serverless.gcp-us-west1.cloud.zilliz.com:443"
    MILVUS_TOKEN: str = "5c13e46bf54ed870dbb413ecae5c527d37ee42ff8840bc3e97c989a1724f598a86c1bd62c6a3858ce85ba87460a358f0e084c6f0"

    # Kafka配置
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_DOCUMENT: str = "document_parse"

    # Mineru配置（Agent轻量API）
    MINERU_BASE_URL: str = "https://mineru.net/api/v1/agent"
    MINERU_TOKEN: str = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIxNDUwMDc1MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTM1NTUxOCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTUxNTUxODA2ODIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJmNjk3ZTNiNy0zNTU1LTQzZjQtOWNjNi0zOWJjN2U0Mjk2ZTIiLCJlbWFpbCI6IiIsImV4cCI6MTc4NzEzMTUxOH0.poKRG8XnPZacuWc9lwsEykqLv4jvs4BxntDUBTaqJCop8zEe1UMIxXV6WgRsWK8lI1H6_CsHLqRINPDCeO_Ohg"

    # Qwen配置
    QWEN_API_KEY: str = "sk-777ae59d8b3e451db4dd91fe6961dbe5"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    QWEN_MODEL_NAME: str = "qwen3.6-plus"

    # Embedding配置
    EMBEDDING_MODEL_NAME: str = "text-embedding-v4"
    EMBEDDING_DIMENSION: int = 1024

    # 存储路径
    DATA_DIR: str = "data"
    PDF_DIR: str = "data/pdfs"
    IMAGE_DIR: str = "data/images"
    CHUNK_DIR: str = "data/chunks"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()