"""测试配置和基本功能"""
from 作业1.app.core.config import settings


def test_config():
    assert settings.APP_NAME == "多模态RAG系统"
    assert settings.PDF_DIR == "data/pdfs"


def test_settings_defaults():
    """测试默认配置"""
    assert settings.DEBUG is True
    assert settings.KAFKA_TOPIC_DOCUMENT == "document_parse"