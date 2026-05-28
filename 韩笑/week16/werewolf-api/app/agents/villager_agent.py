"""VillagerAgent — 平民角色

无夜间技能，仅通过白天发言和投票找出狼人。
最考验推理能力的角色。
Prompt 强调"逻辑分析"而非"信息陈述"；禁止瞎编信息。
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.base_agent import BaseAgent, MemoryEntry
from app.schemas.actions import AgentDecision, SuspectLevel
from app.schemas.messages import (
    ActionRequestMessage,
    Phase,
    PrivateInfoMessage,
    Role,
)

logger = logging.getLogger(__name__)


class VillagerAgent(BaseAgent):
    ROLE = Role.VILLAGER

    def __init__(self, player_id: int, name: str = "", style: str = "balanced", llm_adapter=None):
        super().__init__(player_id, name, style, llm_adapter)
        # 平民推理状态
        self._suspicions: dict[int, SuspectLevel] = {}      # 对每个玩家的怀疑等级
        self._observed_claims: dict[int, str] = {}           # 记录声称的身份
        self._vote_patterns: list[dict] = []                 # 投票行为观察

    # ================================================================
    # 私有信息处理（平民无私有信息）
    # ================================================================

    async def _handle_private_info(self, msg: PrivateInfoMessage) -> None:
        # 平民不接收私有信息
        pass

    # ================================================================
    # 上下文构建
    # ================================================================

    def _build_context(self, msg: ActionRequestMessage) -> dict:
        alive = msg.context.get("alive_players", [])
        return {
            "my_role": "villager",
            "my_id": self.player_id,
            "my_name": self.name,
            "my_style": self.style,
            "phase": msg.phase.value,
            "round": msg.round,
            "valid_actions": msg.valid_actions,
            "alive_players": alive,
            "suspicions": {str(k): v.value for k, v in self._suspicions.items()},
            "observed_claims": self._observed_claims,
            "recent_memory": [
                {"type": e.event_type, "content": e.content}
                for e in self.memory.get_recent(20)
            ],
        }

    # ================================================================
    # CoT 推理
    # ================================================================

    async def _think(self, context: dict, phase: Phase) -> str:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("villager", "system"),
                context,
            )
            phase_key = "day_prompt"  # 平民只有白天
            user = self._render_prompt(
                self._load_prompt_template("villager", phase_key),
                context,
            )
            try:
                return await self._llm.chat(prompt, user)
            except Exception:
                logger.exception("平民 LLM 推理失败，使用规则推理")
        return self._rule_based_think(context, phase)

    # ================================================================
    # 决策
    # ================================================================

    async def _decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("villager", "system"),
                context,
            )
            user = self._render_prompt(
                self._load_prompt_template("villager", "decision_prompt"),
                context,
            )
            try:
                return await self._llm.decide(prompt, user)
            except Exception:
                logger.exception("平民 LLM 决策失败，使用规则决策")
        return self._rule_based_decide(context, phase, valid_actions)

    # ================================================================
    # 公开信息处理（覆盖基类以追踪发言和投票模式）
    # ================================================================

    async def _on_public_broadcast(self, msg) -> None:
        await super()._on_public_broadcast(msg)

        content = msg.content
        event = msg.event

        if event == "player_speak":
            speaker = content.get("player_id")
            text = content.get("content", "")
            # 检测声称身份
            if "预言家" in text:
                self._observed_claims[speaker] = "seer"
            elif "女巫" in text:
                self._observed_claims[speaker] = "witch"
            elif "猎人" in text:
                self._observed_claims[speaker] = "hunter"

        if event == "vote_result":
            self._vote_patterns.append(content)

    # ================================================================
    # 规则兜底
    # ================================================================

    def _rule_based_think(self, context: dict, phase: Phase) -> str:
        alive = context.get("alive_players", [])
        suspicions = context.get("suspicions", {})
        claims = context.get("observed_claims", {})

        parts = [f"[规则推理] 平民{self.player_id}号。存活: {alive}。"]
        if claims:
            parts.append(f"身份声称: {claims}。")
        if suspicions:
            parts.append(f"怀疑度: {suspicions}。")
        parts.append("我是平民，没有额外信息，只能通过发言逻辑推断狼人。")
        return " ".join(parts)

    def _rule_based_decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        alive = context.get("alive_players", [])

        if phase == Phase.SPEECH:
            return AgentDecision(
                action="speak",
                content=self._generate_analysis_speech(context),
                thought="规则决策: 平民分析发言",
            )

        if phase == Phase.VOTE:
            # 投给最高怀疑对象，否则弃票
            high_suspect = self._get_highest_suspect(alive)
            if high_suspect is not None:
                return AgentDecision(
                    action="vote", target=high_suspect,
                    thought=f"规则决策: 投给最高嫌疑 {high_suspect}",
                )
            return AgentDecision(action="abstain", target=None, thought="规则决策: 无明确目标，弃票")

        return AgentDecision(action="skip", target=None, thought="规则决策: 跳过")

    def _generate_analysis_speech(self, context: dict) -> str:
        """生成逻辑分析型发言"""
        alive = context.get("alive_players", [])
        claims = context.get("observed_claims", {})

        parts = [f"我是{self.player_id}号玩家，平民身份。"]
        parts.append(f"目前场上存活{len(alive)}人。")

        if claims:
            for pid, role in claims.items():
                if pid != self.player_id:
                    parts.append(f"{pid}号声称是{role}，大家注意分辨真假。")

        high_suspect = self._get_highest_suspect(alive)
        if high_suspect is not None:
            parts.append(f"我目前最怀疑{high_suspect}号，发言中有矛盾。")

        parts.append("建议大家统一投票，不要分票给狼人可乘之机。")
        return " ".join(parts)

    def _get_highest_suspect(self, alive: list[int]) -> Optional[int]:
        """根据怀疑度找出存活玩家中最可疑的"""
        if not self._suspicions:
            return None
        for pid in alive:
            if self._suspicions.get(pid) == SuspectLevel.CONFIRMED_WOLF:
                return pid
        for pid in alive:
            if self._suspicions.get(pid) == SuspectLevel.LIKELY_WOLF:
                return pid
        return None

    # ================================================================
    # 公开方法: 供外部注入推理线索
    # ================================================================

    def set_suspicion(self, player_id: int, level: SuspectLevel) -> None:
        """设置对某玩家的怀疑等级"""
        self._suspicions[player_id] = level
