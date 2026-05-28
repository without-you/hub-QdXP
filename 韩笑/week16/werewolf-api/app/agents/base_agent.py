"""BaseAgent — 所有角色 Agent 的抽象基类

定义统一接口 + 模板方法模式：
  - 相同行为：消息路由、记忆管理、Prompt 渲染、LLM 调用、兜底
  - 差异化行为：_think()、_decide()、_handle_private_info() 由子类实现
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import BaseLoader, Environment

from app.schemas.actions import AgentDecision, FallbackDecisions, MemoryEntry
from app.schemas.messages import (
    ActionRequestMessage,
    ActionType,
    GameOverMessage,
    GameStartMessage,
    Phase,
    PhaseChangeMessage,
    PrivateInfoMessage,
    PublicBroadcastMessage,
    Role,
    ServerMessage,
)

logger = logging.getLogger(__name__)

# Prompt 模板根目录
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# ============================================================
# 记忆系统
# ============================================================

class MemoryStore:
    """Agent 记忆存储，按重要性和轮次管理"""

    def __init__(self, max_entries: int = 200):
        self._entries: list[MemoryEntry] = []
        self.max_entries = max_entries

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._prune()

    def get_recent(self, n: int = 20) -> list[MemoryEntry]:
        return self._entries[-n:]

    def get_by_type(self, event_type: str) -> list[MemoryEntry]:
        return [e for e in self._entries if e.event_type == event_type]

    def get_important(self, min_importance: int = 3) -> list[MemoryEntry]:
        return [e for e in self._entries if e.importance >= min_importance]

    def get_since_round(self, round_num: int) -> list[MemoryEntry]:
        return [e for e in self._entries if e.round >= round_num]

    def clear(self) -> None:
        self._entries.clear()

    def _prune(self) -> None:
        """超出容量时按重要性保留"""
        if len(self._entries) <= self.max_entries:
            return
        self._entries.sort(key=lambda e: (e.importance, e.round), reverse=True)
        self._entries = self._entries[: self.max_entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)


# ============================================================
# BaseAgent
# ============================================================

class BaseAgent(ABC):
    """所有角色 Agent 的抽象基类

    子类必须实现:
      - _think()               : 产生 CoT 推理文本
      - _decide()              : 从 valid_actions 中选择并产出 AgentDecision
      - _handle_private_info() : 处理角色私有信道消息
      - _get_night_context()   : 构建夜间决策的 prompt context
      - _get_day_context()     : 构建白天发言/投票的 prompt context
    """

    # ---- 角色标识 ----

    ROLE: Role  # 子类必须覆盖

    def __init__(
        self,
        player_id: int,
        name: str = "",
        style: str = "balanced",
        llm_adapter=None,  # LLMAdapter 实例，由工厂注入
    ):
        self.player_id = player_id
        self.name = name or f"Player_{player_id}"
        self.style = style
        self._llm = llm_adapter
        self._alive = True
        self._sheriff = False

        # 记忆
        self.memory = MemoryStore()

        # Prompt 模板引擎
        self._jinja = Environment(loader=BaseLoader())

        # 游戏上下文（由 Game Master 通过 receive_message 注入）
        self._game_context: dict = {}

    # ================================================================
    # 公开属性
    # ================================================================

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_sheriff(self) -> bool:
        return self._sheriff

    @property
    def llm(self):
        """获取 LLM 适配器"""
        return self._llm

    # ================================================================
    # 消息路由（统一入口）
    # ================================================================

    async def receive_message(self, msg: ServerMessage) -> Optional[AgentDecision]:
        """服务端推送消息的统一处理入口

        根据消息 type 路由到对应 handler。
        仅 action_request 需要返回 AgentDecision。
        """
        if isinstance(msg, GameStartMessage):
            await self._on_game_start(msg)
        elif isinstance(msg, PhaseChangeMessage):
            await self._on_phase_change(msg)
        elif isinstance(msg, PrivateInfoMessage):
            await self._handle_private_info(msg)
        elif isinstance(msg, PublicBroadcastMessage):
            await self._on_public_broadcast(msg)
        elif isinstance(msg, ActionRequestMessage):
            return await self._handle_action_request(msg)
        elif isinstance(msg, GameOverMessage):
            await self._on_game_over(msg)
        return None

    # ================================================================
    # 内部消息处理器（可被子类覆盖）
    # ================================================================

    async def _on_game_start(self, msg: GameStartMessage) -> None:
        self._alive = True
        self._game_context["my_role"] = msg.role.value
        self._game_context["player_names"] = msg.player_names
        self._game_context["num_players"] = len(msg.player_names)
        logger.info("[Agent %d] 游戏开始，角色=%s", self.player_id, msg.role.value)

    async def _on_phase_change(self, msg: PhaseChangeMessage) -> None:
        self._game_context["phase"] = msg.phase.value
        self._game_context["round"] = msg.round
        logger.debug("[Agent %d] 阶段变更: %s (第%d轮)", self.player_id, msg.phase.value, msg.round)

    async def _on_public_broadcast(self, msg: PublicBroadcastMessage) -> None:
        entry = MemoryEntry(
            round=self._game_context.get("round", 0),
            phase=self._game_context.get("phase", ""),
            event_type=msg.event,
            content=msg.content,
            importance=self._rate_public_event(msg.event),
        )
        self.memory.add(entry)

    async def _on_game_over(self, msg: GameOverMessage) -> None:
        logger.info("[Agent %d] 游戏结束，胜方=%s，原因=%s", self.player_id, msg.winner.value, msg.reason)

    # ================================================================
    # action_request 处理（模板方法）
    # ================================================================

    async def _handle_action_request(self, msg: ActionRequestMessage) -> AgentDecision:
        """处理行动请求 — 模板方法"""
        deadline = msg.deadline
        try:
            context = self._build_context(msg)
            thought = await self._think(context, msg.phase)
            decision = await self._decide(context, msg.phase, msg.valid_actions)
            decision.thought = thought

            # 记忆
            self.memory.add(MemoryEntry(
                round=msg.round,
                phase=msg.phase.value,
                event_type="self_decision",
                content={"action": decision.action, "target": decision.target, "thought": thought},
                importance=4,
            ))
            return decision

        except Exception:
            logger.exception("[Agent %d] 决策异常，使用兜底策略", self.player_id)
            return FallbackDecisions.for_role(self.ROLE.value, self.player_id)

    # ================================================================
    # 抽象方法 — 子类必须实现
    # ================================================================

    @abstractmethod
    async def _think(self, context: dict, phase: Phase) -> str:
        """产生 Chain-of-Thought 推理文本"""
        ...

    @abstractmethod
    async def _decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        """基于推理结果做出决策"""
        ...

    @abstractmethod
    async def _handle_private_info(self, msg: PrivateInfoMessage) -> None:
        """处理角色私有信道消息（信息隔离核心）"""
        ...

    @abstractmethod
    def _build_context(self, msg: ActionRequestMessage) -> dict:
        """构建当前决策所需的完整上下文，用于 Prompt 渲染"""
        ...

    # ================================================================
    # 共享工具方法
    # ================================================================

    def _load_prompt_template(self, role: str, section: str) -> str:
        """从 YAML 加载 Prompt 模板"""
        path = PROMPTS_DIR / f"{role}.yaml"
        if not path.exists():
            logger.warning("Prompt 模板不存在: %s，使用内置默认模板", path)
            return self._default_prompt(section)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get(role, {}).get(section, self._default_prompt(section))

    def _render_prompt(self, template_str: str, context: dict) -> str:
        """用 Jinja2 渲染 Prompt 模板"""
        template = self._jinja.from_string(template_str)
        return template.render(**context)

    def _rate_public_event(self, event: str) -> int:
        """评估公开事件的重要性（用于记忆管理）"""
        high = {"player_death", "vote_result", "sheriff_elected", "seer_claim"}
        medium = {"player_speak"}
        if event in high:
            return 5
        if event in medium:
            return 3
        return 1

    @staticmethod
    def _default_prompt(section: str) -> str:
        defaults = {
            "system": "你是一名狼人杀玩家，身份是{{ my_role }}。请根据场上信息做出理性决策。",
            "night_prompt": "请做出夜间决策。",
            "day_prompt": "请参与白天讨论和投票。",
        }
        return defaults.get(section, "")

    def __repr__(self) -> str:
        return f"<{self.ROLE.value.capitalize()}Agent id={self.player_id} alive={self._alive}>"
