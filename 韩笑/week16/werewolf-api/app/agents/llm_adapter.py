"""LLMAdapter — 多模型适配层

支持两种调用路径:
  1. DeepSeek API  — 通过 OpenAI SDK 直连（reasoning_effort + thinking）
  2. LiteLLM       — 通用路径，兼容 OpenAI / Qwen / Claude 等

职责:
  - 统一模型调用接口
  - 结构化 JSON 输出 → AgentDecision 解析
  - 解析失败重试 + 兜底
  - asyncio.Semaphore 限流
  - 超时控制
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from app.schemas.actions import AgentDecision

logger = logging.getLogger(__name__)


# ============================================================
# 适配器后端枚举
# ============================================================

class AdapterBackend(Enum):
    DEEPSEEK = auto()   # OpenAI SDK → api.deepseek.com
    LITELLM = auto()    # LiteLLM 通用路径


# ============================================================
# 配置
# ============================================================

@dataclass
class LLMConfig:
    """模型调用配置"""
    model: str = "deepseek-v4-pro"
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str = "high"          # DeepSeek: low | medium | high
    enable_thinking: bool = True            # DeepSeek thinking.type = "enabled"
    request_timeout: float = 60.0
    max_concurrent: int = 2                 # asyncio.Semaphore


# ============================================================
# LLMAdapter
# ============================================================

class LLMAdapter:
    """多模型适配器

    用法:
        # DeepSeek 模式
        adapter = LLMAdapter(
            model="deepseek-v4-pro",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
        )

        # LiteLLM 模式
        adapter = LLMAdapter(backend=AdapterBackend.LITELLM, model="qwen3-8b")

        # 决策
        decision = await adapter.decide(system_prompt, user_prompt)
    """

    # JSON 提取正则
    _JSON_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```|(\{[\s\S]*\})")

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str = "high",
        enable_thinking: bool = True,
        request_timeout: float = 60.0,
        max_concurrent: int = 2,
        backend: AdapterBackend = AdapterBackend.DEEPSEEK,
    ):
        self.config = LLMConfig(
            model=model,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            request_timeout=request_timeout,
            max_concurrent=max_concurrent,
        )
        self.backend = backend
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # 懒加载的客户端实例
        self._openai_client = None

    # ================================================================
    # 核心调用
    # ================================================================

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
    ) -> str:
        """发送 chat completion，返回原始文本"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._call(messages, temperature or self.config.temperature)

    async def decide(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        retry_on_parse_error: bool = True,
    ) -> AgentDecision:
        """调用 LLM 并解析为 AgentDecision

        解析失败时重试一次，仍失败则抛异常，由 BaseAgent 层走兜底。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = await self._call(messages, temperature or self.config.temperature)
        decision = self._parse_decision(raw)

        if decision is None and retry_on_parse_error:
            logger.warning("首次 JSON 解析失败，重试一次...")
            retry_prompt = (
                "你上一次回复格式不符合 JSON 规范。请严格按照以下格式回复，不要添加任何额外文字：\n"
                '```json\n{"thought": "...", "action": "...", "target": <int|null>, "content": "..."}\n```'
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": retry_prompt})
            raw2 = await self._call(messages, temperature or self.config.temperature)
            decision = self._parse_decision(raw2)

        if decision is None:
            raise ValueError(f"LLM 输出无法解析为 AgentDecision: {raw[:300]}")

        return decision

    # ================================================================
    # 内部调用分发
    # ================================================================

    async def _call(self, messages: list[dict], temperature: float) -> str:
        """带限流的模型调用，按后端分发"""
        async with self._semaphore:
            try:
                if self.backend == AdapterBackend.DEEPSEEK:
                    return await self._call_deepseek(messages, temperature)
                else:
                    return await self._call_litellm(messages, temperature)
            except asyncio.TimeoutError:
                logger.error("LLM 调用超时 (%.1fs)", self.config.request_timeout)
                raise
            except Exception:
                logger.exception("LLM 调用失败 model=%s backend=%s", self.config.model, self.backend)
                raise

    # ================================================================
    # DeepSeek API（OpenAI SDK 直连）
    # ================================================================

    async def _call_deepseek(self, messages: list[dict], temperature: float) -> str:
        """通过 OpenAI SDK 调用 DeepSeek API"""
        cfg = self.config

        client = self._get_openai_client()

        start = time.monotonic()

        # 构建 extra_body：启用 thinking
        extra_body = {}
        if cfg.enable_thinking:
            extra_body["thinking"] = {"type": "enabled"}

        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=cfg.max_tokens,
                    stream=False,
                    reasoning_effort=cfg.reasoning_effort,
                    extra_body=extra_body if extra_body else None,
                )
            ),
            timeout=cfg.request_timeout,
        )

        elapsed = time.monotonic() - start

        choice = response.choices[0]
        content = choice.message.content or ""

        # 记录 thinking（DeepSeek 返回的 reasoning_content）
        if cfg.enable_thinking and hasattr(choice.message, "reasoning_content"):
            rc = choice.message.reasoning_content
            if rc:
                logger.debug("DeepSeek reasoning (%.1fs, %d chars): %s...", elapsed, len(rc), rc[:200])

        logger.debug(
            "DeepSeek 调用完成 model=%s elapsed=%.1fs tokens_in=%d tokens_out=%d",
            cfg.model, elapsed,
            getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0,
            getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0,
        )
        return content

    def _get_openai_client(self):
        """懒加载 OpenAI 客户端"""
        if self._openai_client is None:
            from openai import OpenAI

            cfg = self.config
            if not cfg.api_key:
                raise ValueError(
                    "DeepSeek API key 未设置。请在项目根目录 config.yaml 的 llm.api_key 中填写"
                )
            self._openai_client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )
        return self._openai_client

    # ================================================================
    # LiteLLM 通用路径（向后兼容）
    # ================================================================

    async def _call_litellm(self, messages: list[dict], temperature: float) -> str:
        """通过 LiteLLM 调用模型"""
        from litellm import acompletion

        cfg = self.config
        start = time.monotonic()

        response = await asyncio.wait_for(
            acompletion(
                model=cfg.model,
                messages=messages,
                temperature=temperature,
                max_tokens=cfg.max_tokens,
            ),
            timeout=cfg.request_timeout,
        )

        elapsed = time.monotonic() - start
        content = response.choices[0].message.content
        logger.debug("LiteLLM 调用完成 model=%s elapsed=%.1fs", cfg.model, elapsed)
        return content or ""

    # ================================================================
    # JSON 解析
    # ================================================================

    def _parse_decision(self, raw: str) -> Optional[AgentDecision]:
        """从 LLM 原始输出中提取 JSON 并解析为 AgentDecision"""
        if not raw:
            return None

        # 尝试直接解析
        try:
            data = json.loads(raw.strip())
            return AgentDecision(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试从 ```json ... ``` 或 {...} 中提取
        matches = self._JSON_PATTERN.findall(raw)
        for match in matches:
            json_str = match[0] or match[1]
            if not json_str or not json_str.strip():
                continue
            try:
                data = json.loads(json_str.strip())
                return AgentDecision(**data)
            except (json.JSONDecodeError, ValueError):
                continue

        return None


# ================================================================
# 工厂函数
# ================================================================

def create_deepseek_adapter(
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> LLMAdapter:
    """创建 DeepSeek 适配器（从 config.yaml 读取默认值）"""
    from app.config import get_llm_config
    cfg = get_llm_config()
    return LLMAdapter(
        model=model or cfg.get("model", "deepseek-v4-pro"),
        api_key=api_key or cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "https://api.deepseek.com"),
        temperature=kwargs.pop("temperature", cfg.get("temperature", 0.7)),
        max_tokens=kwargs.pop("max_tokens", cfg.get("max_tokens", 4096)),
        reasoning_effort=kwargs.pop("reasoning_effort", cfg.get("reasoning_effort", "high")),
        enable_thinking=kwargs.pop("enable_thinking", cfg.get("enable_thinking", True)),
        request_timeout=kwargs.pop("request_timeout", cfg.get("request_timeout", 60.0)),
        max_concurrent=kwargs.pop("max_concurrent", cfg.get("max_concurrent", 2)),
        backend=AdapterBackend.DEEPSEEK,
        **kwargs,
    )


def create_litellm_adapter(model: str | None = None, **kwargs) -> LLMAdapter:
    """创建 LiteLLM 适配器"""
    from app.config import get_llm_config
    cfg = get_llm_config()
    return LLMAdapter(
        model=model or cfg.get("model", "qwen3-8b"),
        backend=AdapterBackend.LITELLM,
        **kwargs,
    )
