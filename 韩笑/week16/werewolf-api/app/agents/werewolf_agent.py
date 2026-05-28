"""WerewolfAgent — 狼人角色

夜间: 与队友共识击杀（意见不合 → 空刀兜底）
白天: 伪装好人 / 悍跳神职 / 团队冲票或倒钩
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.base_agent import BaseAgent, MemoryEntry
from app.schemas.actions import AgentDecision
from app.schemas.messages import (
    ActionRequestMessage,
    Phase,
    PrivateInfoMessage,
    Role,
)

logger = logging.getLogger(__name__)


class WerewolfAgent(BaseAgent):
    ROLE = Role.WEREWOLF

    def __init__(self, player_id: int, name: str = "", style: str = "balanced", llm_adapter=None):
        super().__init__(player_id, name, style, llm_adapter)
        # 狼人特有状态
        self._teammates: list[int] = []
        self._kill_history: list[dict] = []       # [{round, target, result}]
        self._last_team_chat: str = ""

    # ================================================================
    # 私有信息处理
    # ================================================================

    async def _handle_private_info(self, msg: PrivateInfoMessage) -> None:
        payload = msg.payload

        if "teammates" in payload:
            self._teammates = payload["teammates"]
            self.memory.add(MemoryEntry(
                round=self._game_context.get("round", 0),
                phase="init",
                event_type="teammate_info",
                content={"teammates": self._teammates},
                importance=5,
            ))

        if "team_chat" in payload:
            self._last_team_chat = payload["team_chat"]
            self.memory.add(MemoryEntry(
                round=self._game_context.get("round", 0),
                phase="night_wolf",
                event_type="team_chat",
                content={"chat": payload["team_chat"]},
                importance=4,
            ))

    # ================================================================
    # 上下文构建
    # ================================================================

    def _build_context(self, msg: ActionRequestMessage) -> dict:
        alive = msg.context.get("alive_players", [])
        return {
            "my_role": "werewolf",
            "my_id": self.player_id,
            "my_name": self.name,
            "my_style": self.style,
            "teammates": self._teammates,
            "phase": msg.phase.value,
            "round": msg.round,
            "valid_actions": msg.valid_actions,
            "alive_players": alive,
            "recent_memory": [
                {"type": e.event_type, "content": e.content}
                for e in self.memory.get_recent(15)
            ],
            "kill_history": self._kill_history[-5:],
        }

    # ================================================================
    # CoT 推理
    # ================================================================

    async def _think(self, context: dict, phase: Phase) -> str:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("werewolf", "system"),
                context,
            )
            phase_key = "night_prompt" if phase in (Phase.NIGHT_WOLF,) else "day_prompt"
            user = self._render_prompt(
                self._load_prompt_template("werewolf", phase_key),
                context,
            )
            try:
                return await self._llm.chat(prompt, user)
            except Exception:
                logger.exception("狼人 LLM 推理失败，使用规则推理")
        return self._rule_based_think(context, phase)

    # ================================================================
    # 决策
    # ================================================================

    async def _decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("werewolf", "system"),
                context,
            )
            user = self._render_prompt(
                self._load_prompt_template("werewolf", "decision_prompt"),
                context,
            )
            try:
                return await self._llm.decide(prompt, user)
            except Exception:
                logger.exception("狼人 LLM 决策失败，使用规则决策")
        return self._rule_based_decide(context, phase, valid_actions)

    # ================================================================
    # 规则兜底（无 LLM 时使用）
    # ================================================================

    def _rule_based_think(self, context: dict, phase: Phase) -> str:
        phase_val = phase.value
        if phase == Phase.NIGHT_WOLF:
            alive = context.get("alive_players", [])
            non_wolf_targets = [p for p in alive if p not in self._teammates and p != self.player_id]
            preferred = non_wolf_targets[0] if non_wolf_targets else None
            return (
                f"[规则推理] 我是{self.player_id}号狼人。队友: {self._teammates}。"
                f"存活玩家: {alive}。优先击杀非队友: {preferred}。"
            )
        else:
            return (
                f"[规则推理] 白天发言。我是狼人伪装成平民。"
                f"存活: {context.get('alive_players', [])}。"
                f"应避免暴露，跟随多数人投票。"
            )

    def _rule_based_decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        alive = context.get("alive_players", [])

        if phase == Phase.NIGHT_WOLF:
            non_wolf = [p for p in alive if p not in self._teammates and p != self.player_id]
            if non_wolf:
                target = non_wolf[0]
                return AgentDecision(
                    action="kill", target=target,
                    thought=f"规则决策: 击杀 {target} 号（非队友优先）",
                )
            return AgentDecision(action="skip", target=None, thought="规则决策: 无可击杀目标，空刀")

        if phase == Phase.SPEECH:
            return AgentDecision(
                action="speak",
                content=self._generate_safe_speech(context),
                thought="规则决策: 伪装平民发言",
            )

        if phase == Phase.VOTE:
            if "self_destruct" in valid_actions:
                # 极端劣势自爆（只剩2个存活且1狼）
                wolf_alive = sum(1 for p in self._teammates if p in alive)
                if wolf_alive <= 1 and len(alive) <= 3:
                    return AgentDecision(action="self_destruct", target=None,
                                         thought="规则决策: 自爆进入黑夜")

            non_wolf = [p for p in alive if p not in self._teammates and p != self.player_id]
            target = non_wolf[0] if non_wolf else None
            return AgentDecision(
                action="vote", target=target,
                thought=f"规则决策: 投票放逐 {target}",
            )

        return AgentDecision(action="skip", target=None, thought="规则决策: 跳过")

    def _generate_safe_speech(self, context: dict) -> str:
        """生成安全的平民式发言"""
        alive = context.get("alive_players", [])
        return (
            f"我是{self.player_id}号玩家，平民身份。"
            f"目前场上存活{len(alive)}人，"
            f"我建议大家仔细听发言，找狼人逻辑漏洞。"
            f"我暂时没有明确目标，先听听后面的人怎么说。"
        )
