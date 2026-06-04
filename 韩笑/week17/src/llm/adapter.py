"""
LLM适配器
提供统一的LLM调用接口，支持多种提供商
"""

from typing import List, Dict, Any, Optional
from ..config import Settings
from .providers import (
    BaseLLMProvider,
    DeepSeekProvider,
    OpenAIProvider,
    QwenProvider,
    LiteLLMProvider
)


class LLMAdapter:
    """
    LLM适配器

    根据配置自动选择合适的LLM提供商，提供统一的调用接口
    支持LLM和Embedding使用不同的提供商
    """

    def __init__(self, config: Settings):
        """
        初始化LLM适配器

        Args:
            config: 配置对象
        """
        self.config = config
        self.provider = self._create_provider()
        self.embedding_provider = self._create_embedding_provider()

    def _create_provider(self) -> BaseLLMProvider:
        """
        根据配置创建LLM提供商

        Returns:
            BaseLLMProvider: LLM提供商实例
        """
        provider_name = self.config.llm.provider.lower()

        if provider_name == "deepseek":
            return DeepSeekProvider(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "qwen":
            return QwenProvider(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "litellm":
            return LiteLLMProvider(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {provider_name}")

    def _create_embedding_provider(self) -> BaseLLMProvider:
        """
        根据配置创建Embedding提供商

        Returns:
            BaseLLMProvider: Embedding提供商实例
        """
        provider_name = self.config.embedding.provider.lower()

        # 使用embedding专用配置
        api_key = self.config.embedding.api_key or self.config.llm.api_key
        base_url = self.config.embedding.base_url or self.config.llm.base_url

        if provider_name == "deepseek":
            return DeepSeekProvider(
                api_key=api_key,
                base_url=base_url,
                model=self.config.embedding.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                model=self.config.embedding.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "qwen":
            return QwenProvider(
                api_key=api_key,
                base_url=base_url,
                model=self.config.embedding.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        elif provider_name == "litellm":
            return LiteLLMProvider(
                api_key=api_key,
                base_url=base_url,
                model=self.config.embedding.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            )
        else:
            raise ValueError(f"不支持的Embedding提供商: {provider_name}")

    def chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        与LLM对话

        Args:
            query: 用户查询
            system_prompt: 系统提示词
            history: 对话历史
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: LLM响应
        """
        # 构建消息列表
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加对话历史
        if history:
            messages.extend(history)

        # 添加用户查询
        messages.append({"role": "user", "content": query})

        # 调用LLM
        return self.provider.chat(messages, **kwargs)

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用消息列表与LLM对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: LLM响应
        """
        return self.provider.chat(messages, **kwargs)

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        return self.embedding_provider.get_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return self.embedding_provider.get_embeddings(texts)

    @property
    def provider_name(self) -> str:
        """获取当前LLM提供商名称"""
        return self.provider.name

    @property
    def model_name(self) -> str:
        """获取当前LLM模型名称"""
        return self.provider.model

    @property
    def embedding_provider_name(self) -> str:
        """获取当前Embedding提供商名称"""
        return self.embedding_provider.name

    @property
    def embedding_model_name(self) -> str:
        """获取当前Embedding模型名称"""
        return self.embedding_provider.model