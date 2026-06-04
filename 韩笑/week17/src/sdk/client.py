"""
向量检索与智能缓存服务的Python SDK客户端
"""

from typing import List, Dict, Any, Optional
import redis
from ..cache import SemanticCache, EmbeddingCache, ConversationHistory, IntentClassifier
from ..vector import VectorIndex, VectorSearch
from ..config import get_settings, Settings
from ..llm import LLMAdapter


class VectorCacheClient:
    """
    向量检索与智能缓存服务的主客户端

    提供统一的接口来管理向量索引、执行语义搜索和使用缓存功能
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        redis_url: Optional[str] = None,
        index_name: Optional[str] = None,
        **kwargs
    ):
        """
        初始化客户端

        Args:
            config: 配置对象，如果为None则使用全局配置
            redis_url: Redis连接URL（优先级高于配置文件）
            index_name: 索引名称（优先级高于配置文件）
            **kwargs: 其他配置参数
        """
        # 加载配置
        self.config = config or get_settings()

        # Redis连接配置
        if redis_url:
            self.redis_client = redis.from_url(
                redis_url,
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                socket_connect_timeout=self.config.redis.socket_connect_timeout,
                decode_responses=self.config.redis.decode_responses
            )
        else:
            self.redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                password=self.config.redis.password,
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                socket_connect_timeout=self.config.redis.socket_connect_timeout,
                decode_responses=self.config.redis.decode_responses
            )

        # 索引名称
        self.index_name = index_name or self.config.vector.default_index_name

        # 初始化LLM适配器
        self.llm_adapter = LLMAdapter(self.config)

        # 初始化各个模块
        self.vector_index = VectorIndex(
            self.redis_client,
            self.index_name,
            config=self.config
        )
        self.vector_search = VectorSearch(
            self.redis_client,
            self.index_name,
            config=self.config
        )
        self.semantic_cache = SemanticCache(
            self.redis_client,
            prefix=self.config.cache.semantic.prefix,
            similarity_threshold=self.config.cache.semantic.similarity_threshold,
            embedding_func=self._get_embedding,
            config=self.config
        )
        self.embedding_cache = EmbeddingCache(
            self.redis_client,
            prefix=self.config.cache.embedding.prefix,
            ttl=self.config.cache.embedding.default_ttl,
            config=self.config
        )
        self.conversation_history = ConversationHistory(
            self.redis_client,
            prefix="conversation",
            config=self.config
        )
        self.intent_classifier = IntentClassifier(
            self.redis_client,
            prefix="intent",
            embedding_func=self._get_embedding,
            config=self.config
        )

    def create_index(
        self,
        dimensions: Optional[int] = None,
        distance_metric: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        创建向量索引

        Args:
            dimensions: 向量维度（默认使用配置值）
            distance_metric: 距离度量方式 (COSINE, L2, IP)
            **kwargs: 其他索引配置

        Returns:
            bool: 是否创建成功
        """
        return self.vector_index.create(
            dimensions=dimensions or self.config.vector.default_dimensions,
            distance_metric=distance_metric or self.config.vector.default_distance_metric,
            **kwargs
        )

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        添加文本到索引

        Args:
            text: 文本内容
            metadata: 元数据
            id: 可选的文档ID

        Returns:
            str: 文档ID
        """
        # 先检查嵌入缓存
        embedding = self.embedding_cache.get(text)
        if embedding is None:
            # 如果缓存未命中，需要调用嵌入模型
            embedding = self._get_embedding(text)
            self.embedding_cache.set(text, embedding)

        return self.vector_index.add(
            embedding=embedding,
            metadata={"text": text, **(metadata or {})},
            id=id
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            List[Dict]: 搜索结果列表
        """
        # 获取查询向量
        query_embedding = self.embedding_cache.get(query)
        if query_embedding is None:
            query_embedding = self._get_embedding(query)
            self.embedding_cache.set(query, query_embedding)

        return self.vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的响应

        Args:
            query: 查询文本

        Returns:
            Optional[Dict]: 缓存的响应，如果未命中则返回None
        """
        return self.semantic_cache.get(query)

    def set_cached_response(
        self,
        query: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存的响应

        Args:
            query: 查询文本
            response: 响应数据
            ttl: 过期时间（秒）

        Returns:
            bool: 是否设置成功
        """
        return self.semantic_cache.set(query, response, ttl=ttl)

    def chat(
        self,
        query: str,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        与LLM对话，支持缓存

        Args:
            query: 查询文本
            use_cache: 是否使用缓存
            **kwargs: 其他参数传递给LLM

        Returns:
            Dict[str, Any]: LLM响应
        """
        # 检查缓存
        if use_cache:
            cached_response = self.get_cached_response(query)
            if cached_response:
                return {
                    "response": cached_response,
                    "from_cache": True
                }

        # 调用LLM
        response = self.llm_adapter.chat(query, **kwargs)

        # 缓存响应
        if use_cache:
            self.set_cached_response(
                query,
                response,
                ttl=self.config.cache.semantic.default_ttl
            )

        return {
            "response": response,
            "from_cache": False
        }

    def _get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        return self.llm_adapter.get_embedding(text)

    def chat_with_history(
        self,
        query: str,
        session_id: str,
        use_cache: bool = True,
        max_context_turns: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带对话历史的LLM对话

        Args:
            query: 用户查询
            session_id: 会话ID
            use_cache: 是否使用缓存
            max_context_turns: 上下文最大轮数
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: LLM响应
        """
        # 保存用户消息
        self.conversation_history.add_message(session_id, "user", query)

        # 获取历史上下文
        history = self.conversation_history.get_context_messages(
            session_id, max_turns=max_context_turns
        )

        # 检查缓存
        if use_cache:
            cached_response = self.get_cached_response(query)
            if cached_response:
                # 保存助手回复
                content = cached_response.get("content", str(cached_response))
                self.conversation_history.add_message(
                    session_id, "assistant", content,
                    metadata={"from_cache": True}
                )
                return {
                    "response": cached_response,
                    "from_cache": True,
                    "session_id": session_id
                }

        # 调用LLM（带历史上下文）
        response = self.llm_adapter.chat(
            query,
            history=history[:-1] if len(history) > 1 else None,
            **kwargs
        )

        # 缓存响应
        if use_cache:
            self.set_cached_response(query, response, ttl=self.config.cache.semantic.default_ttl)

        # 保存助手回复
        content = response.get("content", str(response))
        self.conversation_history.add_message(
            session_id, "assistant", content,
            metadata={"from_cache": False}
        )

        return {
            "response": response,
            "from_cache": False,
            "session_id": session_id
        }

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            limit: 返回的最大条数

        Returns:
            List[Dict]: 消息列表
        """
        return self.conversation_history.get_history(session_id, limit=limit)

    def clear_conversation(self, session_id: str) -> bool:
        """
        清除指定会话的历史

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否清除成功
        """
        return self.conversation_history.clear(session_id)

    def register_intent(
        self,
        intent_name: str,
        description: str,
        examples: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册一个意图

        Args:
            intent_name: 意图名称
            description: 意图描述
            examples: 示例文本列表
            metadata: 额外元数据

        Returns:
            bool: 是否注册成功
        """
        return self.intent_classifier.register_intent(
            intent_name, description, examples, metadata
        )

    def classify_intent(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        对文本进行意图分类

        Args:
            text: 输入文本
            top_k: 返回的最大意图数
            threshold: 最低相似度阈值

        Returns:
            List[Dict]: 分类结果列表
        """
        return self.intent_classifier.classify(text, top_k=top_k, threshold=threshold)

    def close(self):
        """关闭连接"""
        self.redis_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()