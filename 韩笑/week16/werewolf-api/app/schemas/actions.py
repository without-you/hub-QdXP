"""Agent 决策输出模型 — 结构化 JSON，服务端二次解析校验"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PlayerStyle(str, Enum):
    """玩家决策风格"""
    BALANCED = "balanced"    # 平衡
    BOLD = "bold"            # 大胆/激进
    CAUTIOUS = "cautious"    # 谨慎/保守
    RANDOM = "random"        # 随机（测试用）


class SuspectLevel(str, Enum):
    """嫌疑等级"""
    TRUST = "trust"          # 铁好人
    LIKELY_GOOD = "likely_good"
    NEUTRAL = "neutral"
    LIKELY_WOLF = "likely_wolf"
    CONFIRMED_WOLF = "confirmed_wolf"  # 查杀


# ============================================================
# 记忆条目
# ============================================================

class MemoryEntry(BaseModel):
    """Agent 记忆单元，持久化到 SQLite"""
    round: int
    phase: str
    event_type: str
    content: dict = Field(default_factory=dict)
    importance: int = Field(default=1, ge=1, le=5, description="1-5 重要性评分")


# ============================================================
# 各阶段决策模型
# ============================================================

class NightKillDecision(BaseModel):
    """狼人夜间击杀决策"""
    player_id: int
    thought: str = Field(default="", description="CoT 推理过程")
    target: int | None = Field(default=None, description="击杀目标，None=空刀")
    reason: str = ""


class NightVerifyDecision(BaseModel):
    """预言家夜间查验决策"""
    player_id: int
    thought: str = ""
    target: int
    reason: str = ""


class NightWitchDecision(BaseModel):
    """女巫夜间用药决策"""
    player_id: int
    thought: str = ""
    use_antidote: bool = False   # 是否使用解药
    use_poison: bool = False     # 是否使用毒药
    poison_target: int | None = None


class DaySpeechDecision(BaseModel):
    """白天发言决策"""
    player_id: int
    thought: str = ""
    content: str = Field(min_length=1, max_length=2000)
    strategy: str = ""  # "claim_seer" | "water_proof" | "accuse" | "defend" | "analysis"


class VoteDecision(BaseModel):
    """投票放逐决策"""
    player_id: int
    thought: str = ""
    target: int | None = None  # None = 弃票
    reason: str = ""


# ============================================================
# AgentDecision — 供 LLM 结构化输出的目标模型
# ============================================================

class AgentDecision(BaseModel):
    """Agent 每次决策的统一输出格式

    LLM System Prompt 要求严格按此 JSON schema 输出。
    服务端收到后做二次解析，失败则重试一次，仍失败走兜底。
    """
    thought: str = Field(
        default="",
        description="Chain-of-Thought 内心推理过程，完整记录在 ActionLog 中",
    )
    action: str = ""       # "kill" | "verify" | "save" | "poison" | "vote" | "speak" | "skip" | "self_destruct"
    target: int | None = Field(default=None, description="目标玩家编号，skip 时为 null")
    content: str = Field(default="", description="发言内容，仅 speak action 时填写")

    @field_validator("content", mode="before")
    @classmethod
    def coerce_content(cls, v):
        """LLM 可能返回 null，转为空字符串"""
        if v is None:
            return ""
        return v

    @field_validator("thought", mode="before")
    @classmethod
    def coerce_thought(cls, v):
        if v is None:
            return ""
        return v


# ============================================================
# 兜底决策
# ============================================================

class FallbackDecisions:
    """超时或解析失败时的默认决策"""

    @staticmethod
    def for_wolf(player_id: int) -> AgentDecision:
        return AgentDecision(
            thought="[兜底] 超时未响应，自动空刀",
            action="kill",
            target=None,
        )

    @staticmethod
    def for_seer(player_id: int) -> AgentDecision:
        return AgentDecision(
            thought="[兜底] 超时未响应，自动跳过查验",
            action="skip",
            target=None,
        )

    @staticmethod
    def for_witch(player_id: int) -> AgentDecision:
        return AgentDecision(
            thought="[兜底] 超时未响应，不使用任何药",
            action="skip",
            target=None,
        )

    @staticmethod
    def for_villager(player_id: int) -> AgentDecision:
        return AgentDecision(
            thought="[兜底] 超时未响应，自动弃票",
            action="vote",
            target=None,
        )

    @classmethod
    def for_role(cls, role: str, player_id: int) -> AgentDecision:
        mapping = {
            "werewolf": cls.for_wolf,
            "seer": cls.for_seer,
            "witch": cls.for_witch,
            "villager": cls.for_villager,
        }
        return mapping.get(role, cls.for_villager)(player_id)
