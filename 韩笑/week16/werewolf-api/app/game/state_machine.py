"""StateMachine — 狼人杀阶段流转引擎

纯逻辑层，不依赖 FastAPI。职责:
  - 阶段顺序定义与推进
  - 各阶段合法行动集合生成
  - 胜利条件判定（屠边规则）
  - 警长竞选 / 猎人开枪等特殊流程

用法:
    sm = StateMachine(board_config)
    sm.advance()           # → next phase
    sm.get_active_roles()  # → 当前阶段可以行动的 role 列表
    sm.check_win(...)      # → Winner | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from app.schemas.messages import Phase, Role, Winner

logger = logging.getLogger(__name__)


# ============================================================
# 内部子状态（用于管理 phase 内部的微步骤）
# ============================================================

class SubPhase(Enum):
    """phase 内部的子状态标记"""
    # speech 子状态
    SHERIFF_CANDIDATES = auto()   # 等待玩家"上警"参选
    SHERIFF_SPEECH = auto()       # 警上玩家轮流发言
    SHERIFF_VOTE = auto()         # 警下玩家投票选警长
    NORMAL_SPEECH = auto()        # 正常轮流发言


# ============================================================
# Phase 定义
# ============================================================

# 阶段推进顺序
PHASE_ORDER: tuple[Phase, ...] = (
    Phase.NIGHT_WOLF,
    Phase.NIGHT_SEER,
    Phase.NIGHT_WITCH,
    Phase.NIGHT_RESULT,
    Phase.DAY_START,
    Phase.SPEECH,
    Phase.VOTE,
    Phase.DAY_END,
)

# 各阶段可以行动的玩家（返回 Role 列表的工厂函数）
PHASE_ACTIVE_ROLES: dict[Phase, tuple[Role, ...]] = {
    Phase.NIGHT_WOLF:   (Role.WEREWOLF,),
    Phase.NIGHT_SEER:   (Role.SEER,),
    Phase.NIGHT_WITCH:  (Role.WITCH,),
    Phase.NIGHT_RESULT: (),   # 系统处理
    Phase.DAY_START:    (),   # 系统处理
    Phase.SPEECH:       (Role.WEREWOLF, Role.SEER, Role.WITCH, Role.VILLAGER),  # 全体存活玩家
    Phase.VOTE:         (Role.WEREWOLF, Role.SEER, Role.WITCH, Role.VILLAGER),
    Phase.DAY_END:      (),   # 系统处理
}


# ============================================================
# StateMachine
# ============================================================

@dataclass
class StateMachine:
    """游戏阶段状态机

    维护当前阶段 / 轮次 / 天数，提供阶段推进、合法行动查询、胜负判定。
    """

    # 板子规则
    sheriff_election: bool = True
    sheriff_vote_power: float = 1.5
    allow_wolf_self_destruct: bool = True
    witch_can_self_save_first_night: bool = True
    witch_has_poison: bool = False
    dead_can_last_words: bool = True

    # 运行时状态
    phase: Phase = Phase.NIGHT_WOLF
    round: int = 1                        # 游戏轮次（每轮 = 黑夜 + 白天）
    day_number: int = 0                   # 当前天数（从 0 开始，night_wolf 不变，day_start +1）

    # 警长竞选
    sheriff_election_done: bool = False
    sheriff_election_round: int = 1       # 警长竞选仅在第 1 天
    sub_phase: Optional[SubPhase] = None
    sheriff_candidates: list[int] = field(default_factory=list)
    sheriff_votes: dict[int, int] = field(default_factory=dict)

    # 当前阶段已行动的玩家集合（用于判断阶段是否完成）
    _acted_players: set[int] = field(default_factory=set)

    # ================================================================
    # 阶段查询
    # ================================================================

    @property
    def is_night(self) -> bool:
        return self.phase in (Phase.NIGHT_WOLF, Phase.NIGHT_SEER, Phase.NIGHT_WITCH)

    @property
    def is_day(self) -> bool:
        return self.phase in (Phase.DAY_START, Phase.SPEECH, Phase.VOTE, Phase.DAY_END)

    def get_active_roles(self) -> tuple[Role, ...]:
        """当前阶段可行动的 Role 列表"""
        return PHASE_ACTIVE_ROLES.get(self.phase, ())

    def is_role_active(self, role: Role) -> bool:
        return role in self.get_active_roles()

    def get_valid_actions(
        self,
        role: Role,
        player_id: int,
        alive_players: list[int],
        *,
        can_self_save: bool = True,
    ) -> list[str]:
        """生成当前阶段该角色的合法行动列表

        Returns:
            如 ["kill_1","kill_2","skip"] 或 ["verify_1","verify_2","verify_3"]
        """
        phase = self.phase

        if phase == Phase.NIGHT_WOLF:
            targets = [p for p in alive_players if role != Role.WEREWOLF or True]  # 狼人可刀任意存活者
            return [f"kill_{t}" for t in alive_players] + ["skip"]

        if phase == Phase.NIGHT_SEER:
            return [f"verify_{t}" for t in alive_players]

        if phase == Phase.NIGHT_WITCH:
            actions: list[str] = []
            # 解药
            actions.append("save")    # 使用解药
            actions.append("nosave")  # 不使用解药
            # 毒药
            if self.witch_has_poison:
                actions.extend(f"poison_{t}" for t in alive_players)
                actions.append("nopoison")
            return actions

        if phase == Phase.VOTE:
            # 可投任意存活玩家 + 弃票
            return [f"vote_{t}" for t in alive_players] + ["abstain"]

        if phase == Phase.SPEECH:
            return ["speak"]

        return ["skip"]

    # ================================================================
    # 阶段推进
    # ================================================================

    def advance(self) -> Phase:
        """推进到下一个阶段，返回新阶段"""
        idx = PHASE_ORDER.index(self.phase)
        next_idx = (idx + 1) % len(PHASE_ORDER)
        prev_phase = self.phase
        self.phase = PHASE_ORDER[next_idx]

        # 天数推进
        if prev_phase == Phase.NIGHT_WOLF and self.phase == Phase.NIGHT_SEER:
            pass  # 同一天黑夜
        elif self.phase == Phase.DAY_START:
            self.day_number += 1
        elif self.phase == Phase.NIGHT_WOLF:
            self.round += 1

        # 清理当前阶段状态
        self._acted_players.clear()

        # speech 阶段首日进入警长竞选
        if self.phase == Phase.SPEECH:
            if self._should_hold_election():
                self.sub_phase = SubPhase.SHERIFF_CANDIDATES
                self.sheriff_candidates.clear()
                self.sheriff_votes.clear()
            else:
                self.sub_phase = SubPhase.NORMAL_SPEECH

        logger.info(
            "阶段推进: %s → %s (round=%d day=%d)",
            prev_phase.value, self.phase.value, self.round, self.day_number,
        )
        return self.phase

    def mark_player_acted(self, player_id: int) -> None:
        self._acted_players.add(player_id)

    def all_required_acted(self, active_player_count: int) -> bool:
        return len(self._acted_players) >= active_player_count

    def reset_acted(self) -> None:
        self._acted_players.clear()

    # ================================================================
    # 警长竞选
    # ================================================================

    def _should_hold_election(self) -> bool:
        return (
            self.sheriff_election
            and not self.sheriff_election_done
            and self.round == self.sheriff_election_round
        )

    def enter_sheriff_candidate(self, player_id: int) -> None:
        """玩家上警参选"""
        if self.sub_phase != SubPhase.SHERIFF_CANDIDATES:
            return
        if player_id not in self.sheriff_candidates:
            self.sheriff_candidates.append(player_id)

    def start_sheriff_speeches(self) -> None:
        self.sub_phase = SubPhase.SHERIFF_SPEECH

    def start_sheriff_vote(self) -> None:
        self.sub_phase = SubPhase.SHERIFF_VOTE

    def record_sheriff_vote(self, voter_id: int, candidate_id: int) -> None:
        self.sheriff_votes[voter_id] = candidate_id

    def finish_sheriff_election(self) -> int | None:
        """结算警长竞选，返回当选者 ID 或 None（平票无警长）"""
        if not self.sheriff_votes:
            return None

        tally: dict[int, int] = {}
        for candidate_id in self.sheriff_votes.values():
            tally[candidate_id] = tally.get(candidate_id, 0) + 1

        max_votes = max(tally.values())
        winners = [c for c, v in tally.items() if v == max_votes]

        if len(winners) == 1:
            self.sheriff_election_done = True
            self.sub_phase = SubPhase.NORMAL_SPEECH
            return winners[0]
        return None  # 平票

    # ================================================================
    # 胜负判定 — 屠边规则
    # ================================================================

    @staticmethod
    def check_win(players: list[dict]) -> Optional[Winner]:
        """检查是否有一方达成胜利条件

        Args:
            players: [{"role": Role, "is_alive": bool}, ...]

        Returns:
            Winner.GOOD / Winner.EVIL / None（游戏继续）
        """
        alive_wolves = sum(
            1 for p in players
            if p["role"] == Role.WEREWOLF and p["is_alive"]
        )
        alive_gods = sum(
            1 for p in players
            if p["role"] in (Role.SEER, Role.WITCH) and p["is_alive"]
        )
        alive_villagers = sum(
            1 for p in players
            if p["role"] == Role.VILLAGER and p["is_alive"]
        )

        # 好人胜利：所有狼人阵亡
        if alive_wolves == 0:
            return Winner.GOOD

        # 狼人屠边成功：所有神职死亡 OR 所有平民死亡
        if alive_gods == 0 or alive_villagers == 0:
            return Winner.EVIL

        return None

    @staticmethod
    def get_win_reason(winner: Winner, players: list[dict]) -> str:
        """生成胜利原因描述"""
        alive_wolves = sum(1 for p in players if p["role"] == Role.WEREWOLF and p["is_alive"])
        alive_gods = sum(1 for p in players if p["role"] in (Role.SEER, Role.WITCH) and p["is_alive"])
        alive_villagers = sum(1 for p in players if p["role"] == Role.VILLAGER and p["is_alive"])

        if winner == Winner.GOOD:
            return "all_wolves_dead"
        if alive_gods == 0:
            return "all_gods_dead"
        if alive_villagers == 0:
            return "all_villagers_dead"
        return "unknown"

    # ================================================================
    # 阶段超时配置
    # ================================================================

    @staticmethod
    def get_timeout(phase: Phase, timeouts: Optional[dict] = None) -> int:
        """获取阶段的超时秒数"""
        defaults = {
            Phase.NIGHT_WOLF: 60,
            Phase.NIGHT_SEER: 30,
            Phase.NIGHT_WITCH: 30,
            Phase.NIGHT_RESULT: 5,
            Phase.DAY_START: 5,
            Phase.SPEECH: 120,
            Phase.VOTE: 60,
            Phase.DAY_END: 5,
        }
        if timeouts:
            defaults.update(timeouts)
        return defaults.get(phase, 30)

    # ================================================================
    # 重置
    # ================================================================

    def reset(self) -> None:
        self.phase = Phase.NIGHT_WOLF
        self.round = 1
        self.day_number = 0
        self.sheriff_election_done = False
        self.sheriff_candidates.clear()
        self.sheriff_votes.clear()
        self.sub_phase = None
        self._acted_players.clear()
