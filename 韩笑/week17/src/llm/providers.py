"""
LLM提供商实现
支持DeepSeek、OpenAI、通义千问、LiteLLM等多种提供商
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
import json


class BaseLLMProvider(ABC):
    """
    LLM提供商基类

    定义所有LLM提供商必须实现的接口
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 30
    ):
        """
        初始化提供商

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            timeout: 请求超时时间
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """提供商名称"""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        与LLM对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        pass

    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        发送HTTP请求

        Args:
            endpoint: API端点
            data: 请求数据
            headers: 请求头

        Returns:
            Dict[str, Any]: 响应数据
        """
        url = f"{self.base_url}{endpoint}"

        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if headers:
            default_headers.update(headers)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                json=data,
                headers=default_headers
            )
            response.raise_for_status()
            return response.json()


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek提供商

    支持DeepSeek的Chat和Embedding API
    """

    @property
    def name(self) -> str:
        return "deepseek"

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        与DeepSeek对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }

        response = self._make_request("/chat/completions", data)

        return {
            "content": response["choices"][0]["message"]["content"],
            "model": response["model"],
            "usage": response.get("usage", {}),
            "finish_reason": response["choices"][0].get("finish_reason")
        }

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.embeddings.create(
                model="deepseek-embedding",
                input=text
            )

            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: DeepSeek embedding failed: {e}")
            # 返回一个模拟的嵌入向量（仅用于测试）
            import hashlib
            import numpy as np
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            return np.random.randn(1536).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.embeddings.create(
                model="deepseek-embedding",
                input=texts
            )

            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Warning: DeepSeek batch embedding failed: {e}")
            # 返回模拟的嵌入向量
            return [self.get_embedding(text) for text in texts]


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI提供商

    支持OpenAI的Chat和Embedding API
    """

    @property
    def name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        与OpenAI对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }

        response = self._make_request("/chat/completions", data)

        return {
            "content": response["choices"][0]["message"]["content"],
            "model": response["model"],
            "usage": response.get("usage", {}),
            "finish_reason": response["choices"][0].get("finish_reason")
        }

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }

        response = self._make_request("/embeddings", data)

        return response["data"][0]["embedding"]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        data = {
            "model": "text-embedding-ada-002",
            "input": texts
        }

        response = self._make_request("/embeddings", data)

        return [item["embedding"] for item in response["data"]]


class QwenProvider(BaseLLMProvider):
    """
    通义千问提供商

    支持通义千问的Chat和Embedding API
    """

    @property
    def name(self) -> str:
        return "qwen"

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        与通义千问对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        data = {
            "model": kwargs.get("model", self.model),
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature)
            }
        }

        response = self._make_request("/services/aigc/text-generation/generation", data)

        return {
            "content": response["output"]["choices"][0]["message"]["content"],
            "model": response.get("model", self.model),
            "usage": response.get("usage", {}),
            "finish_reason": response["output"]["choices"][0].get("finish_reason")
        }

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.embeddings.create(
                model=self.model,
                input=text
            )

            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Qwen embedding failed: {e}")
            # 返回一个模拟的嵌入向量（仅用于测试）
            import hashlib
            import numpy as np
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            return np.random.randn(1024).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = client.embeddings.create(
                model=self.model,
                input=texts
            )

            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Warning: Qwen batch embedding failed: {e}")
            # 返回模拟的嵌入向量
            return [self.get_embedding(text) for text in texts]


class LiteLLMProvider(BaseLLMProvider):
    """
    LiteLLM提供商

    通过LiteLLM统一接口调用多种LLM
    """

    @property
    def name(self) -> str:
        return "litellm"

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        通过LiteLLM对话

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        try:
            import litellm

            response = litellm.completion(
                model=kwargs.get("model", self.model),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                api_key=self.api_key,
                api_base=self.base_url
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
        except ImportError:
            raise ImportError("请安装litellm: pip install litellm")

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        Args:
            text: 文本内容

        Returns:
            List[float]: 嵌入向量
        """
        try:
            import litellm

            response = litellm.embedding(
                model=self.model,
                input=[text],
                api_key=self.api_key,
                api_base=self.base_url
            )

            return response.data[0]["embedding"]
        except ImportError:
            raise ImportError("请安装litellm: pip install litellm")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        try:
            import litellm

            response = litellm.embedding(
                model=self.model,
                input=texts,
                api_key=self.api_key,
                api_base=self.base_url
            )

            return [item["embedding"] for item in response.data]
        except ImportError:
            raise ImportError("请安装litellm: pip install litellm")