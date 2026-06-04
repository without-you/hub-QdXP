"""
向量搜索模块
提供向量相似性搜索和混合查询功能
"""

from typing import List, Dict, Any, Optional
import redis


class VectorSearch:
    """
    向量搜索实现

    提供向量相似性搜索、元数据过滤和混合查询功能
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        index_name: str = "vector_index",
        config=None
    ):
        """
        初始化向量搜索

        Args:
            redis_client: Redis客户端
            index_name: 索引名称
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.index_name = index_name
        self.config = config
        self.client = redis_client.ft(index_name)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行向量相似性搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件
            return_fields: 需要返回的字段

        Returns:
            List[Dict]: 搜索结果列表
        """
        import numpy as np

        # 序列化查询向量
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        # 构建查询字符串
        if filters:
            filter_str = self._build_filter_str(filters)
            query_str = f"{filter_str}=>[KNN {top_k} @embedding $BLOB AS score]"
        else:
            query_str = f"*=>[KNN {top_k} @embedding $BLOB AS score]"

        # 执行搜索
        try:
            results = self.redis.execute_command(
                'FT.SEARCH', self.index_name,
                query_str,
                'PARAMS', '2', 'BLOB', query_vector,
                'SORTBY', 'score',
                'RETURN', '3', 'text', 'score', 'category',
                'DIALECT', '2'
            )

            # 处理结果
            return self._process_raw_results(results)

        except redis.exceptions.ResponseError as e:
            print(f"搜索错误: {e}")
            return []

    def hybrid_search(
        self,
        query_embedding: List[float],
        text_query: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行混合搜索（向量 + 文本）

        Args:
            query_embedding: 查询向量
            text_query: 文本查询
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            List[Dict]: 搜索结果列表
        """
        import numpy as np

        # 序列化查询向量
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        # 构建查询字符串
        query_parts = []

        # 添加文本查询
        if text_query:
            query_parts.append(f"@text:{text_query}")

        # 添加过滤条件
        if filters:
            filter_str = self._build_filter_str(filters)
            query_parts.append(filter_str)

        # 组合查询
        if query_parts:
            filter_str = " ".join(query_parts)
            query_str = f"({filter_str})=>[KNN {top_k} @embedding $BLOB AS score]"
        else:
            query_str = f"*=>[KNN {top_k} @embedding $BLOB AS score]"

        # 执行搜索
        try:
            results = self.redis.execute_command(
                'FT.SEARCH', self.index_name,
                query_str,
                'PARAMS', '2', 'BLOB', query_vector,
                'SORTBY', 'score',
                'RETURN', '3', 'text', 'score', 'category',
                'DIALECT', '2'
            )

            return self._process_raw_results(results)

        except redis.exceptions.ResponseError as e:
            print(f"混合搜索错误: {e}")
            return []

    def search_by_category(
        self,
        query_embedding: List[float],
        category: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        按类别搜索

        Args:
            query_embedding: 查询向量
            category: 类别
            top_k: 返回结果数量

        Returns:
            List[Dict]: 搜索结果列表
        """
        filters = {"category": category}
        return self.search(query_embedding, top_k=top_k, filters=filters)

    def _build_filter_str(self, filters: Dict[str, Any]) -> str:
        """
        构建过滤条件字符串

        Args:
            filters: 过滤条件

        Returns:
            str: 过滤条件字符串
        """
        filter_parts = []

        for field, value in filters.items():
            if isinstance(value, dict):
                # 范围查询
                if "min" in value and "max" in value:
                    filter_parts.append(f"@{field}:[{value['min']} {value['max']}]")
                elif "min" in value:
                    filter_parts.append(f"@{field}:[{value['min']} +inf]")
                elif "max" in value:
                    filter_parts.append(f"@{field}:[-inf {value['max']}]")
            elif isinstance(value, list):
                # 多值查询
                values = " | ".join(str(v) for v in value)
                filter_parts.append(f"@{field}:{{{values}}}")
            else:
                # 精确匹配（TAG字段）
                filter_parts.append(f"@{field}:{{{value}}}")

        if filter_parts:
            return " ".join(filter_parts)
        return "*"

    def _process_raw_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        处理原始搜索结果

        Args:
            results: 原始搜索结果

        Returns:
            List[Dict]: 处理后的结果
        """
        processed = []

        if not results or not isinstance(results, dict):
            return processed

        raw_results = results.get(b'results', [])

        for doc in raw_results:
            if isinstance(doc, dict):
                doc_id = doc.get(b'id', b'').decode('utf-8', errors='ignore')
                extra = doc.get(b'extra_attributes', {})

                result = {
                    "id": doc_id,
                    "text": extra.get(b'text', b'').decode('utf-8', errors='ignore'),
                    "score": float(extra.get(b'score', 0)),
                    "category": extra.get(b'category', b'').decode('utf-8', errors='ignore')
                }
                processed.append(result)

        return processed