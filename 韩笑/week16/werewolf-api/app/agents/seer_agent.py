"""SeerAgent — 预言家角色

夜间: 查验一名玩家身份（金水/查杀）
白天: 报验人信息，争取警徽，带领好人阵营
核心: VerifiedList 记忆模块，防止遗忘或篡改历史验人结果
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


class VerifiedList:
    """预言家查验记录 — 不可变的核心记忆

    每次查验结果以 MemoryEntry 形式持久化。
    """

    def __init__(self):
        self._records: dict[int, bool] = {}  # {player_id: is_wolf}

    def record(self, player_id: int, is_wolf: bool) -> None:
        self._records[player_id] = is_wolf

    def is_wolf(self, player_id: int) -> Optional[bool]:
        return self._records.get(player_id)

    @property
    def gold_water(self) -> list[int]:
        """金水列表（验出的好人）"""
        return [pid for pid, is_w in self._records.items() if not is_w]

    @property
    def wolf_check(self) -> list[int]:
        """查杀列表（验出的狼人）"""
        return [pid for pid, is_w in self._records.items() if is_w]

    @property
    def checked_count(self) -> int:
        return len(self._records)

    def to_context(self) -> dict:
        return {
            "verified": self._records,
            "gold_water": self.gold_water,
            "wolf_check": self.wolf_check,
        }

    def __repr__(self) -> str:
        return f"VerifiedList(gold={self.gold_water}, wolf={self.wolf_check})"


class SeerAgent(BaseAgent):
    ROLE = Role.SEER

    def __init__(self, player_id: int, name: str = "", style: str = "balanced", llm_adapter=None):
        super().__init__(player_id, name, style, llm_adapter)
        self._verified = VerifiedList()
        self._last_check_round: int = 0  # 上次查验的轮次
        self._revealed_info: list[int] = []  # 已经报过的验人信息（避免重复报）

    # ================================================================
    # 私有信息处理
    # ================================================================

    async def _handle_private_info(self, msg: PrivateInfoMessage) -> None:
        payload = msg.payload

        if "verified" in payload:
            verified = payload["verified"]
            for pid_str, is_wolf in verified.items():
                pid = int(pid_str)
                self._verified.record(pid, is_wolf)
                label = "狼人" if is_wolf else "好人"
                self.memory.add(MemoryEntry(
                    round=self._game_context.get("round", 0),
                    phase="night_seer",
                    event_type="verify_result",
                    content={"target": pid, "is_wolf": is_wolf, "label": label},
                    importance=5,  # 最高重要性，永不被剪枝
                ))

        if "verify_result" in payload:
            target = payload.get("target")
            is_wolf = payload.get("is_wolf", False)
            if target is not None:
                self._verified.record(target, is_wolf)
                self._last_check_round = self._game_context.get("round", 0)
                logger.info("[预言家 %d] 查验 %d → %s", self.player_id, target, "狼人" if is_wolf else "好人")

    # ================================================================
    # 上下文构建
    # ================================================================

    def _build_context(self, msg: ActionRequestMessage) -> dict:
        alive = msg.context.get("alive_players", [])
        verif = self._verified.to_context()
        return {
            "my_role": "seer",
            "my_id": self.player_id,
            "my_name": self.name,
            "my_style": self.style,
            "phase": msg.phase.value,
            "round": msg.round,
            "valid_actions": msg.valid_actions,
            "alive_players": alive,
            "verified": verif["verified"],
            "gold_water": verif["gold_water"],
            "wolf_check": verif["wolf_check"],
            "checked_count": self._verified.checked_count,
            "unchecked": [p for p in alive if p not in verif["verified"] and p != self.player_id],
            "revealed_info": self._revealed_info,
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
                self._load_prompt_template("seer", "system"),
                context,
            )
            phase_key = "night_prompt" if phase == Phase.NIGHT_SEER else "day_prompt"
            user = self._render_prompt(
                self._load_prompt_template("seer", phase_key),
                context,
            )
            try:
                return await self._llm.chat(prompt, user)
            except Exception:
                logger.exception("预言家 LLM 推理失败，使用规则推理")
        return self._rule_based_think(context, phase)

    # ================================================================
    # 决策
    # ================================================================

    async def _decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        if self._llm is not None:
            prompt = self._render_prompt(
                self._load_prompt_template("seer", "system"),
                context,
            )
            user = self._render_prompt(
                self._load_prompt_template("seer", "decision_prompt"),
                context,
            )
            try:
                return await self._llm.decide(prompt, user)
            except Exception:
                logger.exception("预言家 LLM 决策失败，使用规则决策")
        return self._rule_based_decide(context, phase, valid_actions)

    # ================================================================
    # 规则兜底
    # ================================================================

    def _rule_based_think(self, context: dict, phase: Phase) -> str:
        if phase == Phase.NIGHT_SEER:
            unchecked = context.get("unchecked", [])
            gold = context.get("gold_water", [])
            wolf = context.get("wolf_check", [])
            return (
                f"[规则推理] 预言家{self.player_id}号。"
                f"已查验{self._verified.checked_count}人: 金水{gold}, 查杀{wolf}。"
                f"未查验: {unchecked}。优先查验存活非金水玩家。"
            )
        else:
            wolf = context.get("wolf_check", [])
            gold = context.get("gold_water", [])
            return (
                f"[规则推理] 白天报验人。查杀={wolf}, 金水={gold}。"
                f"若有查杀应优先引导放逐查杀对象。"
            )

    def _rule_based_decide(
        self, context: dict, phase: Phase, valid_actions: list[str]
    ) -> AgentDecision:
        alive = context.get("alive_players", [])

        if phase == Phase.NIGHT_SEER:
            unchecked = context.get("unchecked", [])
            # 优先查验存活非自己的玩家
            targets = [p for p in unchecked if p in alive and p != self.player_id]
            if targets:
                target = targets[0]
                return AgentDecision(
                    action="verify", target=target,
                    thought=f"规则决策: 查验 {target} 号",
                )
            return AgentDecision(action="skip", target=None, thought="规则决策: 无有效查验目标")

        if phase == Phase.SPEECH:
            wolf = context.get("wolf_check", [])
            gold = context.get("gold_water", [])
            alive_wolf = [w for w in wolf if w in alive]
            alive_gold = [g for g in gold if g in alive]

            content_parts = [f"我是{self.player_id}号玩家，预言家身份。"]
            if alive_wolf:
                content_parts.append(f"昨晚查验{alive_wolf[0]}号，查杀！请跟我投票放逐。")
            elif alive_gold:
                content_parts.append(f"昨晚查验{alive_gold[0]}号，金水。")
            else:
                content_parts.append("暂无有效验人信息。")

            content_parts.append(f"已查验{self._verified.checked_count}人，请大家相信我的验人结果。")
            return AgentDecision(
                action="speak",
                content=" ".join(content_parts),
                thought=f"规则决策: 报验人信息",
            )

        if phase == Phase.VOTE:
            wolf = context.get("wolf_check", [])
            alive_wolf = [w for w in wolf if w in alive]
            if alive_wolf:
                return AgentDecision(
                    action="vote", target=alive_wolf[0],
                    thought=f"规则决策: 投票放逐查杀 {alive_wolf[0]}",
                )
            # 无查杀时投最可疑的
            target = alive[0] if alive and alive[0] != self.player_id else (alive[1] if len(alive) > 1 else None)
            return AgentDecision(action="vote", target=target, thought="规则决策: 跟随直觉投票")

        return AgentDecision(action="skip", target=None, thought="规则决策: 跳过")
