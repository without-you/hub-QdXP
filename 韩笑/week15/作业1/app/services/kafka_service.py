"""Kafka消息队列服务"""
import json
from kafka import KafkaProducer, KafkaConsumer
from 作业1.app.core.config import settings


class KafkaService:
    def __init__(self):
        self._producer = None
        self._consumer = None

    @property
    def producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                acks='all',
                retries=3
            )
        return self._producer

    def get_consumer(self, group_id: str = "document_parser") -> KafkaConsumer:
        return KafkaConsumer(
            settings.KAFKA_TOPIC_DOCUMENT,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )

    def send_parse_task(self, document_id: str, file_path: str, filename: str):
        """发送文档解析任务到Kafka"""
        self.producer.send(
            settings.KAFKA_TOPIC_DOCUMENT,
            value={
                "document_id": document_id,
                "file_path": file_path,
                "filename": filename,
                "action": "parse"
            }
        )
        self.producer.flush()

    def close(self):
        if self._producer:
            self._producer.close()


kafka_service = KafkaService()