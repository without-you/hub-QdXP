"""WitchAgent — 女巫角色

夜间: 决定是否使用解药（救被刀者）/ 毒药（毒杀一人）
白天: 隐藏神职身份，利用救人所知信息辅助推理
核心: 资源状态锁（has_antidote / has_poison），各限一次
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


class WitchAgent(BaseAgent):
    ROLE = Role.WITCH

    def __init__(self, player_id: int, name: str = "", style: str = "balanced", llm_adapter=None):
        super().__init__(player_id, name, style, llm_adapter)
        # 药水状态
        self._has_antidote: bool = True   # 解药
        self._has_poison: bool = False    # 毒药（6人板子默认无）
        self._saved_players: list[int] = []
        self._poisoned_players: list[int] = []
        self._night_kill_target: Optional[int] = None  # 每晚狼刀目标
        self._can_self_save_first_night: bool = True

    # ================================================================
    # 私有信息处理
    # ================================================================

    async def _handle_private_info(self, msg: PrivateInfoMessage) -> None:
        payload = msg.payload

        if "antidote_available" in payload:
            self._has_antidote = payload["antidote_available"]

        if "poison_available" in payload:
            self._has_poison = payload["poison_available"]

        if "night_kill_target" in payload:
            self._night_kill_target = payload["night_kill_target"]
            if self._night_kill_target is not None:
                self.memory.add(MemoryEntry(
                    round=self._game_context.get("round", 0),
                    phase="night_witch",
                    event_type="wolf_kill_info",
                    content={"target": self._night_kill_target},
                    importance=5,
                ))

    # ================================================================
    # 上下文构建
    # ================================================================

    def _build_context(self, msg: ActionRequestMessage) -> dict:
        alive = msg.context.get("alive_players", [])
        return {
            "my_role": "witch",
            "my_id": self.player_id,
            "my_name": self.name,
            "my_style": self.style,
            "phase": msg.phase.value,
            "round": msg.round,
            "valid_actions": msg.valid_actions,
            "alive_players": alive,
            "has_antidote": self._has_antidote,
            "has_poison": self._has_poison,
            "night_kill_target": self._night_kill_target,
            "saved_players": self._saved_players,
            "is_first_night": (msg.round == 1),
            "recent_memory": [
                {"type": e.event_type, "content": e.content}
                for e in self.memory.get_recent(10)
            ],
        }

    # ================================================================
    # CoT 推理
    # ================================================================

    async def _think(self, context: dict, phase: Phase) -> str:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("witch", "system"),
                context,
            )
            phase_key = "night_prompt" if phase == Phase.NIGHT_WITCH else "day_prompt"
            user = self._render_prompt(
                self._load_prompt_template("witch", phase_key),
                context,
            )
            try:
                return await self._llm.chat(prompt, user)
            except Exception:
                logger.exception("女巫 LLM 推理失败，使用规则推理")
        return self._rule_based_think(context, phase)

    # ================================================================
    # 决策
    # ================================================================

    async def _decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("witch", "system"),
                context,
            )
            user = self._render_prompt(
                self._load_prompt_template("witch", "decision_prompt"),
                context,
            )
            try:
                return await self._llm.decide(prompt, user)
            except Exception:
                logger.exception("女巫 LLM 决策失败，使用规则决策")
        return self._rule_based_decide(context, phase, valid_actions)

    # ================================================================
    # 规则兜底
    # ================================================================

    def _rule_based_think(self, context: dict, phase: Phase) -> str:
        if phase == Phase.NIGHT_WITCH:
            target = context.get("night_kill_target")
            parts = [f"[规则推理] 女巫{self.player_id}号。"]
            if self._has_antidote:
                parts.append(f"解药可用。被刀者: {target}。")
                if target == self.player_id and context.get("is_first_night"):
                    parts.append("首夜自救通常明智。")
                else:
                    parts.append(f"考虑是否救 {target}。")
            else:
                parts.append("解药已用。")
            if self._has_poison:
                parts.append("毒药可用。")
            return " ".join(parts)
        else:
            return (
                f"[规则推理] 女巫{self.player_id}号，隐藏身份发言。"
                f"我知道昨晚的被刀信息，但不能暴露。"
            )

    def _rule_based_decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        if phase == Phase.NIGHT_WITCH:
            target = context.get("night_kill_target")
            is_first = context.get("is_first_night", False)

            # 解药逻辑
            use_save = False
            if self._has_antidote and target is not None:
                if target == self.player_id and is_first and self._can_self_save_first_night:
                    # 首夜自救
                    use_save = True
                elif target != self.player_id:
                    # 救他人（首夜必救，后续慎重）
                    if is_first:
                        use_save = True
                    else:
                        use_save = False  # 非首夜不自动救

            if use_save:
                self._has_antidote = False
                self._saved_players.append(target)
                return AgentDecision(
                    action="save",
                    target=target,
                    thought=f"规则决策: 使用解药救 {target} 号",
                )

            # 毒药逻辑（默认不用）
            return AgentDecision(
                action="nosave",
                target=None,
                thought="规则决策: 不使用解药",
            )

        if phase == Phase.SPEECH:
            return AgentDecision(
                action="speak",
                content=self._generate_safe_speech(context),
                thought="规则决策: 女巫伪装平民发言",
            )

        if phase == Phase.VOTE:
            alive = context.get("alive_players", [])
            target = alive[0] if alive and alive[0] != self.player_id else (alive[1] if len(alive) > 1 else None)
            return AgentDecision(
                action="vote", target=target,
                thought=f"规则决策: 投票放逐 {target}",
            )

        return AgentDecision(action="skip", target=None, thought="规则决策: 跳过")

    def _generate_safe_speech(self, context: dict) -> str:
        """生成不暴露女巫身份的发言"""
        alive = context.get("alive_players", [])
        return (
            f"我是{self.player_id}号玩家，平民身份。"
            f"目前存活{len(alive)}人。"
            f"我认真听了前面的发言，建议大家理性分析投票，不要分票。"
            f"我会跟大多数好人一起投票。"
        )
