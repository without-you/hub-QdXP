"""
缓存模块测试
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from src.cache.semantic_cache import SemanticCache
from src.cache.embedding_cache import EmbeddingCache


class TestSemanticCache:
    """测试SemanticCache类"""

    def setup_method(self):
        """测试前准备"""
        self.redis_client = Mock()
        self.cache = SemanticCache(
            redis_client=self.redis_client,
            prefix="test_semantic",
            similarity_threshold=0.95
        )

    def test_make_key(self):
        """测试生成缓存键"""
        key = self.cache._make_key("测试文本")
        assert "test_semantic:exact:" in key

    def test_get_exact_match(self):
        """测试精确匹配获取"""
        expected_response = {"answer": "北京"}
        self.redis_client.get.return_value = json.dumps(expected_response).encode()

        result = self.cache.get("中国的首都是哪里？")

        assert result == expected_response
        self.redis_client.get.assert_called_once()

    def test_get_no_match(self):
        """测试未命中获取"""
        self.redis_client.get.return_value = None

        result = self.cache.get("不存在的问题")

        assert result is None

    def test_set_with_ttl(self):
        """测试设置带TTL的缓存"""
        self.redis_client.setex.return_value = True

        result = self.cache.set(
            "测试问题",
            {"answer": "测试答案"},
            ttl=3600
        )

        assert result is True
        self.redis_client.setex.assert_called_once()

    def test_set_without_ttl(self):
        """测试设置不带TTL的缓存"""
        self.redis_client.set.return_value = True

        result = self.cache.set(
            "测试问题",
            {"answer": "测试答案"}
        )

        assert result is True
        self.redis_client.set.assert_called_once()

    def test_delete(self):
        """测试删除缓存"""
        self.redis_client.delete.return_value = 1

        result = self.cache.delete("测试问题")

        assert result is True

    def test_clear(self):
        """测试清空缓存"""
        self.redis_client.keys.return_value = [b"key1", b"key2"]
        self.redis_client.delete.return_value = 2

        result = self.cache.clear()

        assert result is True

    def test_get_stats(self):
        """测试获取统计信息"""
        self.redis_client.keys.return_value = [b"key1", b"key2"]

        stats = self.cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["prefix"] == "test_semantic"


class TestEmbeddingCache:
    """测试EmbeddingCache类"""

    def setup_method(self):
        """测试前准备"""
        self.redis_client = Mock()
        self.cache = EmbeddingCache(
            redis_client=self.redis_client,
            prefix="test_embedding",
            ttl=86400
        )

    def test_make_key(self):
        """测试生成缓存键"""
        key = self.cache._make_key("测试文本")
        assert "test_embedding:" in key

    def test_get(self):
        """测试获取嵌入向量"""
        expected_embedding = [0.1] * 1536
        self.redis_client.get.return_value = json.dumps(expected_embedding).encode()

        result = self.cache.get("测试文本")

        assert result == expected_embedding

    def test_get_no_match(self):
        """测试未命中获取"""
        self.redis_client.get.return_value = None

        result = self.cache.get("不存在的文本")

        assert result is None

    def test_set(self):
        """测试设置嵌入向量"""
        self.redis_client.setex.return_value = True
        embedding = [0.1] * 1536

        result = self.cache.set("测试文本", embedding, ttl=3600)

        assert result is True

    def test_batch_get(self):
        """测试批量获取"""
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        self.redis_client.pipeline.return_value.__enter__ = Mock(
            return_value=Mock(execute=Mock(return_value=[
                json.dumps(embeddings[0]).encode(),
                json.dumps(embeddings[1]).encode()
            ]))
        )
        self.redis_client.pipeline.return_value.__exit__ = Mock()

        results = self.cache.batch_get(["文本1", "文本2"])

        assert len(results) == 2
        assert results[0] == embeddings[0]
        assert results[1] == embeddings[1]

    def test_batch_set(self):
        """测试批量设置"""
        self.redis_client.pipeline.return_value.__enter__ = Mock(
            return_value=Mock(execute=Mock())
        )
        self.redis_client.pipeline.return_value.__exit__ = Mock()

        texts = ["文本1", "文本2"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        result = self.cache.batch_set(texts, embeddings)

        assert result is True

    def test_batch_set_length_mismatch(self):
        """测试批量设置长度不匹配"""
        texts = ["文本1", "文本2"]
        embeddings = [[0.1] * 1536]

        with pytest.raises(ValueError):
            self.cache.batch_set(texts, embeddings)


if __name__ == "__main__":
    pytest.main([__file__])