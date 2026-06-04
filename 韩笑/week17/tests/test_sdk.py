"""
SDK客户端测试
"""

import pytest
from unittest.mock import Mock, patch
from src.sdk.client import VectorCacheClient


class TestVectorCacheClient:
    """测试VectorCacheClient类"""

    def setup_method(self):
        """测试前准备"""
        self.client = VectorCacheClient(
            redis_url="redis://localhost:6379",
            index_name="test_index"
        )

    def test_init(self):
        """测试初始化"""
        assert self.client.index_name == "test_index"
        assert self.client.vector_index is not None
        assert self.client.vector_search is not None
        assert self.client.semantic_cache is not None
        assert self.client.embedding_cache is not None

    @patch('src.sdk.client.VectorCacheClient._get_embedding')
    def test_add_text(self, mock_get_embedding):
        """测试添加文本"""
        mock_get_embedding.return_value = [0.1] * 1536

        # 模拟缓存未命中
        self.client.embedding_cache.get = Mock(return_value=None)
        self.client.embedding_cache.set = Mock(return_value=True)

        # 模拟向量索引添加
        self.client.vector_index.add = Mock(return_value="test_id")

        result = self.client.add_text("测试文本", {"category": "test"})

        assert result == "test_id"
        mock_get_embedding.assert_called_once_with("测试文本")

    @patch('src.sdk.client.VectorCacheClient._get_embedding')
    def test_search(self, mock_get_embedding):
        """测试搜索"""
        mock_get_embedding.return_value = [0.1] * 1536

        # 模拟缓存未命中
        self.client.embedding_cache.get = Mock(return_value=None)
        self.client.embedding_cache.set = Mock(return_value=True)

        # 模拟搜索结果
        expected_results = [
            {"id": "1", "score": 0.95, "text": "测试文本1"},
            {"id": "2", "score": 0.85, "text": "测试文本2"}
        ]
        self.client.vector_search.search = Mock(return_value=expected_results)

        results = self.client.search("测试查询", top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "1"

    def test_get_cached_response(self):
        """测试获取缓存响应"""
        expected_response = {"answer": "北京"}
        self.client.semantic_cache.get = Mock(return_value=expected_response)

        result = self.client.get_cached_response("中国的首都是哪里？")

        assert result == expected_response

    def test_set_cached_response(self):
        """测试设置缓存响应"""
        self.client.semantic_cache.set = Mock(return_value=True)

        result = self.client.set_cached_response(
            "中国的首都是哪里？",
            {"answer": "北京"},
            ttl=3600
        )

        assert result is True

    def test_close(self):
        """测试关闭连接"""
        self.client.redis_client.close = Mock()
        self.client.close()
        self.client.redis_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])