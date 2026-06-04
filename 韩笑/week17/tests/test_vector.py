"""
向量模块测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.vector.index import VectorIndex
from src.vector.search import VectorSearch


class TestVectorIndex:
    """测试VectorIndex类"""

    def setup_method(self):
        """测试前准备"""
        self.redis_client = Mock()
        self.index = VectorIndex(
            redis_client=self.redis_client,
            index_name="test_index"
        )

    def test_create(self):
        """测试创建索引"""
        self.index.client.create_index = Mock()

        result = self.index.create(dimensions=1536, distance_metric="COSINE")

        assert result is True
        self.index.client.create_index.assert_called_once()

    def test_create_already_exists(self):
        """测试创建已存在的索引"""
        from redis.exceptions import ResponseError
        self.index.client.create_index = Mock(
            side_effect=ResponseError("Index already exists")
        )

        result = self.index.create(dimensions=1536)

        assert result is True

    def test_add(self):
        """测试添加向量"""
        self.redis_client.hset.return_value = True

        embedding = [0.1] * 1536
        metadata = {"text": "测试文本", "category": "test"}

        result = self.index.add(embedding, metadata)

        assert result is not None
        self.redis_client.hset.assert_called_once()

    def test_add_with_id(self):
        """测试添加向量（指定ID）"""
        self.redis_client.hset.return_value = True

        embedding = [0.1] * 1536
        metadata = {"text": "测试文本"}

        result = self.index.add(embedding, metadata, id="custom_id")

        assert result == "custom_id"

    def test_add_batch(self):
        """测试批量添加向量"""
        self.redis_client.pipeline.return_value.__enter__ = Mock(
            return_value=Mock(execute=Mock())
        )
        self.redis_client.pipeline.return_value.__exit__ = Mock()

        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata_list = [
            {"text": "文本1"},
            {"text": "文本2"}
        ]

        results = self.index.add_batch(embeddings, metadata_list)

        assert len(results) == 2

    def test_delete(self):
        """测试删除文档"""
        self.redis_client.delete.return_value = 1

        result = self.index.delete("test_id")

        assert result is True

    def test_get(self):
        """测试获取文档"""
        doc_data = {
            b"id": b"test_id",
            b"text": b"test text",
            b"category": b"test",
            b"source": b"test",
            b"timestamp": b"1234567890",
            b"embedding": np.array([0.1] * 1536, dtype=np.float32).tobytes()
        }
        self.redis_client.hgetall.return_value = doc_data

        result = self.index.get("test_id")

        assert result is not None
        assert result["id"] == "test_id"
        assert result["text"] == "test text"

    def test_get_not_found(self):
        """测试获取不存在的文档"""
        self.redis_client.hgetall.return_value = {}

        result = self.index.get("nonexistent_id")

        assert result is None

    def test_info(self):
        """测试获取索引信息"""
        self.index.client.info = Mock(return_value={
            "num_docs": 100,
            "max_doc_id": 100,
            "num_terms": 50,
            "num_records": 100
        })

        info = self.index.info()

        assert info["index_name"] == "test_index"
        assert info["num_docs"] == 100

    def test_drop(self):
        """测试删除索引"""
        self.index.client.dropindex = Mock()

        result = self.index.drop()

        assert result is True


class TestVectorSearch:
    """测试VectorSearch类"""

    def setup_method(self):
        """测试前准备"""
        self.redis_client = Mock()
        self.search = VectorSearch(
            redis_client=self.redis_client,
            index_name="test_index"
        )

    def test_search(self):
        """测试向量搜索"""
        # 模拟搜索结果（新实现返回原始字典格式）
        mock_results = {
            b'results': [
                {
                    b'id': b'test_index:test_id',
                    b'extra_attributes': {
                        b'text': b'test text',
                        b'score': b'0.95',
                        b'category': b'test'
                    }
                }
            ]
        }
        self.redis_client.execute_command = Mock(return_value=mock_results)

        query_embedding = [0.1] * 1536
        results = self.search.search(query_embedding, top_k=10)

        assert len(results) == 1
        assert results[0]["id"] == "test_index:test_id"
        assert results[0]["score"] == 0.95

    def test_search_with_filters(self):
        """测试带过滤条件的搜索"""
        mock_results = {
            b'results': [
                {
                    b'id': b'test_index:test_id',
                    b'extra_attributes': {
                        b'text': b'test text',
                        b'score': b'0.95',
                        b'category': b'test'
                    }
                }
            ]
        }
        self.redis_client.execute_command = Mock(return_value=mock_results)

        query_embedding = [0.1] * 1536
        filters = {"category": "test"}
        results = self.search.search(query_embedding, filters=filters)

        assert len(results) == 1

    def test_search_by_category(self):
        """测试按类别搜索"""
        mock_results = {
            b'results': [
                {
                    b'id': b'test_index:test_id',
                    b'extra_attributes': {
                        b'text': b'test text',
                        b'score': b'0.95',
                        b'category': b'test_category'
                    }
                }
            ]
        }
        self.redis_client.execute_command = Mock(return_value=mock_results)

        query_embedding = [0.1] * 1536
        results = self.search.search_by_category(query_embedding, "test_category")

        assert len(results) == 1

    def test_search_error_handling(self):
        """测试搜索错误处理"""
        from redis.exceptions import ResponseError
        self.search.client.search = Mock(
            side_effect=ResponseError("Search error")
        )

        query_embedding = [0.1] * 1536
        results = self.search.search(query_embedding)

        assert results == []

    def test_build_filter_str(self):
        """测试构建过滤条件字符串"""
        filters = {
            "category": "test",
            "timestamp": {"min": 100, "max": 200}
        }

        filter_str = self.search._build_filter_str(filters)

        assert "@category:{test}" in filter_str
        assert "@timestamp:[100 200]" in filter_str


if __name__ == "__main__":
    pytest.main([__file__])