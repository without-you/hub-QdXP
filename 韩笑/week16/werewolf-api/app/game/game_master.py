"""GameMaster — 裁判核心

纯逻辑层。职责:
  - 游戏创建与玩家管理
  - 阶段驱动：调用 StateMachine + 等待 Agent 行动 + 推进
  - 信息隔离：按角色过滤消息，私有信道单播
  - 投票统计：含警长 1.5 票加权
  - 超时兜底调度
  - 结构化日志（CoT / 决策 / 游戏事件）

用法（单步）:
    gm = GameMaster(board_config)
    gm.init_game(player_names)
    result = await gm.step(agent_decisions)   # agent_decisions: {player_id: AgentDecision}
    # result 包含 phase / deaths / public_log / private_info / is_game_over

用法（全自动）:
    gm = GameMaster(board_config)
    gm.init_game(player_names)
    gm.set_message_sender(ws_send_callback)
    winner = await gm.run_game()   # 驱动整局游戏直到结束
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.schemas.actions import AgentDecision, FallbackDecisions
from app.schemas.messages import (
    ActionRequestMessage,
    ActionType,
    DeathRecord,
    GameOverMessage,
    GameStartMessage,
    Phase,
    PhaseChangeMessage,
    PlayerState,
    PrivateInfoMessage,
    PublicBroadcastMessage,
    Role,
    VoteRecord,
    Winner,
)
from app.game.state_machine import PHASE_ORDER, SubPhase, StateMachine

logger = logging.getLogger(__name__)

BOARDS_DIR = Path(__file__).resolve().parent / "boards"


# ============================================================
# 内部数据结构
# ============================================================

@dataclass
class Player:
    player_id: int
    name: str
    role: Role
    is_alive: bool = True
    is_sheriff: bool = False

    @property
    def state(self) -> PlayerState:
        return PlayerState(player_id=self.player_id, name=self.name, is_alive=self.is_alive, is_sheriff=self.is_sheriff)

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "role": self.role, "is_alive": self.is_alive}


@dataclass
class GameState:
    game_id: str
    board_type: str
    players: list[Player]
    sm: StateMachine
    public_log: list[str] = field(default_factory=list)
    deaths_this_round: list[DeathRecord] = field(default_factory=list)
    night_kill_target: int | None = None      # 狼人击杀目标
    witch_saved: int | None = None             # 女巫解救目标
    witch_poisoned: int | None = None          # 女巫毒杀目标
    seer_checked: dict[int, bool] = field(default_factory=dict)  # {target_id: is_wolf}
    vote_records: list[VoteRecord] = field(default_factory=list)
    speech_queue: list[int] = field(default_factory=list)
    speech_index: int = 0
    is_game_over: bool = False
    winner: Optional[Winner] = None
    win_reason: str = ""

    # 对局完整日志
    action_log: list[dict] = field(default_factory=list)


# ============================================================
# GameMaster
# ============================================================

class GameMaster:
    """裁判核心引擎"""

    def __init__(self, board_config: dict | str):
        """
        Args:
            board_config: dict 配置或 board 文件名（如 "standard_6"）
        """
        if isinstance(board_config, str):
            board_config = self._load_board(board_config)
        self.board = board_config
        self._state: Optional[GameState] = None

        # 消息发送回调（由 WS 层注入）
        self._sender: Optional[Callable] = None

    # ================================================================
    # 配置加载
    # ================================================================

    @staticmethod
    def _load_board(name: str) -> dict:
        path = BOARDS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"板子配置不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ================================================================
    # 游戏初始化
    # ================================================================

    def init_game(
        self,
        player_names: list[str],
        game_id: str | None = None,
        shuffle_roles: bool = True,
    ) -> str:
        """初始化对局：分配角色、创建状态机

        Returns:
            game_id
        """
        board_roles = self.board["roles"]
        total = self.board["total_players"]
        rules = self.board.get("rules", {})

        if len(player_names) != total:
            raise ValueError(f"玩家数需为 {total}，实际 {len(player_names)}")

        # 构建角色列表
        role_list: list[Role] = []
        for role_name, count in board_roles.items():
            role_list.extend([Role(role_name)] * count)

        if shuffle_roles:
            random.shuffle(role_list)

        # 创建玩家
        players = [
            Player(player_id=i, name=name, role=role_list[i])
            for i, name in enumerate(player_names)
        ]

        # 创建状态机
        sm = StateMachine(
            sheriff_election=rules.get("sheriff_election", True),
            sheriff_vote_power=rules.get("sheriff_vote_power", 1.5),
            allow_wolf_self_destruct=rules.get("allow_wolf_self_destruct", True),
            witch_can_self_save_first_night=rules.get("witch_can_self_save_first_night", True),
            witch_has_poison=rules.get("witch_has_poison", False),
            dead_can_last_words=rules.get("dead_can_last_words", True),
        )

        gid = game_id or f"g_{uuid.uuid4().hex[:8]}"
        self._state = GameState(
            game_id=gid,
            board_type=self.board["board_type"],
            players=players,
            sm=sm,
        )

        self._log_public(f"[系统] 对局 {gid} 创建成功，{total}名玩家就位")
        for p in players:
            logger.info("  seat=%d role=%s name=%s", p.player_id, p.role.value, p.name)

        return gid

    # ================================================================
    # 消息发送注入
    # ================================================================

    def set_message_sender(self, sender: Callable):
        """注入消息发送回调

        Args:
            sender: async def send(player_id: int, msg: ServerMessage) -> None
        """
        self._sender = sender

    async def _send(self, player_id: int, msg) -> None:
        if self._sender:
            await self._sender(player_id, msg)

    async def _broadcast(self, msg) -> None:
        for p in self._state.players:
            if p.is_alive or isinstance(msg, GameOverMessage):
                await self._send(p.player_id, msg)

    # ================================================================
    # 游戏启动
    # ================================================================

    async def send_game_start(self) -> None:
        """向所有玩家发送 game_start 消息（含角色分配）"""
        s = self._state
        for p in s.players:
            teammates = [
                other.player_id
                for other in s.players
                if other.role == Role.WEREWOLF and other.player_id != p.player_id
            ]
            await self._send(p.player_id, GameStartMessage(
                player_id=p.player_id,
                role=p.role,
                teammates=teammates if p.role == Role.WEREWOLF else [],
                player_names={pl.player_id: pl.name for pl in s.players},
            ))

    # ================================================================
    # step() — 单步推进（核心）
    # ================================================================

    async def step(self, agent_decisions: dict[int, AgentDecision] | None = None) -> dict:
        """执行一次 step，推进一个阶段

        Args:
            agent_decisions: {player_id: AgentDecision}，外部 Agent 的决策结果。
                             若为 None 表示系统阶段（night_result/day_start/day_end）。

        Returns:
            {
                "phase": Phase,
                "day_number": int,
                "step_data": dict,
                "players": list[PlayerState],
                "dialogues": list[dict],
                "deaths": list[DeathRecord],
                "is_game_over": bool,
                "winner": Winner | None,
                "action_requests": list[dict],   # 下一阶段需要 Agent 行动的请求
            }
        """
        s = self._state
        sm = s.sm
        phase = sm.phase
        decisions = agent_decisions or {}

        result = {
            "phase": phase.value,
            "day_number": s.sm.day_number,
            "step_data": {},
            "players": [p.state for p in s.players],
            "dialogues": [],
            "deaths": [],
            "is_game_over": False,
            "winner": None,
            "action_requests": [],
        }

        # ——— 分发到各阶段处理器 ———
        if phase == Phase.NIGHT_WOLF:
            result["step_data"] = self._handle_night_wolf(decisions)
        elif phase == Phase.NIGHT_SEER:
            result["step_data"] = self._handle_night_seer(decisions)
        elif phase == Phase.NIGHT_WITCH:
            result["step_data"] = self._handle_night_witch(decisions)
        elif phase == Phase.NIGHT_RESULT:
            result.update(self._handle_night_result())
        elif phase == Phase.DAY_START:
            result["step_data"] = self._handle_day_start()
        elif phase == Phase.SPEECH:
            result["step_data"] = self._handle_speech(decisions)
        elif phase == Phase.VOTE:
            result.update(self._handle_vote(decisions))
        elif phase == Phase.DAY_END:
            result.update(self._handle_day_end())

        # 检查游戏结束
        player_dicts = [p.to_dict() for p in s.players]
        winner = sm.check_win(player_dicts)
        if winner:
            s.is_game_over = True
            s.winner = winner
            s.win_reason = sm.get_win_reason(winner, player_dicts)
            result["is_game_over"] = True
            result["winner"] = winner.value
            result["deaths"] = s.deaths_this_round
            return result

        # 推进到下一阶段，并生成 action_requests
        # 警长竞选子阶段未完成时，不离开 SPEECH 阶段
        if not (sm.phase == Phase.SPEECH and sm.sub_phase not in (None, SubPhase.NORMAL_SPEECH)):
            sm.advance()
        result["action_requests"] = self._build_action_requests()
        return result

    # ================================================================
    # 构建下一阶段的 action_requests
    # ================================================================

    def _build_action_requests(self) -> list[dict]:
        """根据新阶段生成需要行动的 Agent 列表和合法行动"""
        s = self._state
        sm = s.sm
        phase = sm.phase
        alive = [p.player_id for p in s.players if p.is_alive]
        timeouts = self.board.get("timeouts", {})
        timeout = StateMachine.get_timeout(phase, timeouts)

        active_roles = sm.get_active_roles()
        if not active_roles:
            return []

        requests = []
        for p in s.players:
            if not p.is_alive:
                continue
            if p.role not in active_roles:
                continue

            valid_actions = sm.get_valid_actions(p.role, p.player_id, alive)
            requests.append({
                "player_id": p.player_id,
                "role": p.role.value,
                "phase": phase.value,
                "round": sm.round,
                "valid_actions": valid_actions,
                "deadline": time.time() + timeout,
            })
        return requests

    # ================================================================
    # 夜间阶段处理器
    # ================================================================

    def _handle_night_wolf(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        alive = [p.player_id for p in s.players if p.is_alive]

        # 收集狼人投票
        wolf_votes: list[dict] = []
        for pid, decision in decisions.items():
            player = self._get_player(pid)
            if not player or player.role != Role.WEREWOLF or not player.is_alive:
                continue
            target = decision.target if decision.action == "kill" else None
            wolf_votes.append({"player_id": pid, "target": target})
            self._log_action(pid, s.sm.round, "night_wolf", decision)

        # 共识机制：多数票 → 击杀目标；平票或无人投票 → 空刀
        tally: dict[int, int] = {}
        for v in wolf_votes:
            if v["target"] is not None and v["target"] in alive:
                tally[v["target"]] = tally.get(v["target"], 0) + 1

        target = None
        if tally:
            max_votes = max(tally.values())
            top_targets = [t for t, c in tally.items() if c == max_votes]
            if len(top_targets) == 1:
                target = top_targets[0]

        s.night_kill_target = target
        if target is not None:
            self._log_public(f"[狼人] 击杀目标: {target}号玩家")
        else:
            self._log_public("[狼人] 今晚空刀")

        return {"wolf_votes": wolf_votes, "final_target": target}

    def _handle_night_seer(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        alive = [p.player_id for p in s.players if p.is_alive]

        seer = self._get_players_by_role(Role.SEER)[0] if self._get_players_by_role(Role.SEER) else None
        if not seer or not seer.is_alive:
            return {"verify_target": None, "result": None}

        decision = decisions.get(seer.player_id)
        if not decision or decision.action not in ("verify",):
            return {"verify_target": None, "result": None}

        target = decision.target
        if target is None or target not in alive:
            return {"verify_target": None, "result": None}

        target_player = self._get_player(target)
        is_wolf = target_player.role == Role.WEREWOLF if target_player else False
        s.seer_checked[target] = is_wolf

        self._log_private(seer.player_id, f"[预言家] 查验 {target}号 → {'狼人' if is_wolf else '好人'}")
        self._log_action(seer.player_id, s.sm.round, "night_seer", decision)

        return {"verify_target": target, "is_wolf": is_wolf}

    def _handle_night_witch(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state

        witch = self._get_players_by_role(Role.WITCH)[0] if self._get_players_by_role(Role.WITCH) else None
        if not witch or not witch.is_alive:
            return {"used_antidote": False, "used_poison": False}

        decision = decisions.get(witch.player_id)
        if not decision:
            return {"used_antidote": False, "used_poison": False}

        # 解药
        use_antidote = (decision.action == "save" and s.night_kill_target is not None)
        if use_antidote:
            s.witch_saved = s.night_kill_target
        else:
            s.witch_saved = None

        # 毒药（仅当板子允许且女巫使用）
        use_poison = False
        if self.board["rules"].get("witch_has_poison", False):
            use_poison = decision.action == "poison" and decision.target is not None
            if use_poison:
                s.witch_poisoned = decision.target

        self._log_action(witch.player_id, s.sm.round, "night_witch", decision)

        return {"used_antidote": use_antidote, "used_poison": use_poison, "saved_player": s.witch_saved, "poisoned_player": s.witch_poisoned}

    # ================================================================
    # 夜晚结算
    # ================================================================

    def _handle_night_result(self) -> dict:
        s = self._state
        deaths: list[DeathRecord] = []
        dialogues: list[dict] = []

        # 狼刀结算
        if s.night_kill_target is not None and s.night_kill_target != s.witch_saved:
            victim = self._kill_player(s.night_kill_target, "killed_by_wolves")
            if victim:
                deaths.append(victim)
        elif s.witch_saved is not None:
            dialogues.append({"speaker": "system", "content": f"女巫使用解药救活了 {s.witch_saved} 号玩家"})

        # 毒药结算
        if s.witch_poisoned is not None:
            victim = self._kill_player(s.witch_poisoned, "poisoned")
            if victim:
                deaths.append(victim)

        s.deaths_this_round = deaths

        # 重置本轮夜间状态（为下一轮做准备）
        s.night_kill_target = None
        s.witch_saved = None
        s.witch_poisoned = None

        return {
            "step_data": {"night_result": "resolved"},
            "deaths": deaths,
            "dialogues": dialogues,
            "is_game_over": s.is_game_over,
            "winner": s.winner.value if s.winner else None,
        }

    # ================================================================
    # 白天阶段处理器
    # ================================================================

    def _handle_day_start(self) -> dict:
        s = self._state
        deaths = s.deaths_this_round

        if deaths:
            for d in deaths:
                self._log_public(f"[死讯] {d.player_id}号玩家({d.role.value}) 死亡，原因: {d.cause}")
                # 死者遗言
                if self.board["rules"].get("dead_can_last_words", True):
                    self._log_public(f"[遗言] {d.player_id}号玩家留下遗言")
        else:
            self._log_public("[系统] 昨晚是平安夜")

        return {"deaths": [{"player_id": d.player_id, "role": d.role.value, "cause": d.cause} for d in deaths]}

    def _handle_speech(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        sm = s.sm

        # 首日警长竞选流程
        if sm.sub_phase == SubPhase.SHERIFF_CANDIDATES:
            return self._handle_sheriff_candidates(decisions)
        elif sm.sub_phase == SubPhase.SHERIFF_SPEECH:
            return self._handle_sheriff_speeches(decisions)
        elif sm.sub_phase == SubPhase.SHERIFF_VOTE:
            return self._handle_sheriff_vote(decisions)

        # 普通发言
        dialogues: list[dict] = []
        for pid, decision in decisions.items():
            if decision.action == "speak" and decision.content:
                player = self._get_player(pid)
                dialogues.append({"player_id": pid, "name": player.name if player else "", "content": decision.content, "thought": decision.thought})
                self._log_public(f"[发言] {pid}号({player.name if player else ''}): {decision.content[:100]}...")
                self._log_action(pid, s.sm.round, "speech", decision)

        return {"dialogues": dialogues}

    def _handle_sheriff_candidates(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        sm = s.sm
        candidates: list[int] = []

        for pid, decision in decisions.items():
            if decision.action == "run_sheriff":
                sm.enter_sheriff_candidate(pid)
                candidates.append(pid)
                self._log_public(f"[警长竞选] {pid}号玩家上警参选")

        if not candidates:
            # 无人上警 → 跳过警长竞选，直接进入正常发言
            self._log_public("[警长] 无人上警，本局无警长")
            sm.sub_phase = SubPhase.NORMAL_SPEECH
            return {"sheriff_phase": "candidates", "candidates": [], "skipped": True}

        sm.start_sheriff_speeches()
        return {"sheriff_phase": "candidates", "candidates": candidates}

    def _handle_sheriff_speeches(self, decisions: dict[int, AgentDecision]) -> dict:
        dialogues: list[dict] = []
        for pid, decision in decisions.items():
            if decision.action == "speak" and decision.content:
                player = self._get_player(pid)
                dialogues.append({"player_id": pid, "name": player.name if player else "", "content": decision.content, "thought": decision.thought})
                self._log_public(f"[警上发言] {pid}号: {decision.content[:100]}...")

        self._state.sm.start_sheriff_vote()
        return {"sheriff_phase": "speeches", "dialogues": dialogues}

    def _handle_sheriff_vote(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        sm = s.sm

        # 警下玩家投票（非参选者）
        for pid, decision in decisions.items():
            if pid in sm.sheriff_candidates:
                continue
            if decision.action == "vote_sheriff" and decision.target in sm.sheriff_candidates:
                sm.record_sheriff_vote(pid, decision.target)
                self._log_public(f"[警长投票] {pid}号 → {decision.target}号")

        winner = sm.finish_sheriff_election()
        if winner is not None:
            self._set_sheriff(winner)
            self._log_public(f"[警长] {winner}号玩家当选警长")
        else:
            self._log_public("[警长] 平票，本局无警长")

        return {"sheriff_phase": "vote", "elected_sheriff": winner}

    # ================================================================
    # 投票放逐
    # ================================================================

    def _handle_vote(self, decisions: dict[int, AgentDecision]) -> dict:
        s = self._state
        alive = [p.player_id for p in s.players if p.is_alive]
        votes: dict[int, int | None] = {}  # {voter_id: target_id | None}

        for pid, decision in decisions.items():
            player = self._get_player(pid)
            if not player or not player.is_alive:
                continue

            target = None
            if decision.action == "vote":
                target = decision.target if decision.target in alive else None
            elif decision.action == "self_destruct":
                # 狼人自爆 → 立即进入黑夜
                if player.role == Role.WEREWOLF and self.board["rules"].get("allow_wolf_self_destruct", True):
                    self._kill_player(pid, "self_destruct")
                    self._log_public(f"[自爆] {pid}号狼人自爆！直接进入黑夜")
                    s.sm.advance()  # 跳过投票 → 进入 day_end → 下一轮黑夜
                    return {
                        "step_data": {"wolf_self_destruct": pid},
                        "deaths": s.deaths_this_round,
                        "dialogues": [{"speaker": "system", "content": f"{pid}号狼人自爆，直接进入黑夜"}],
                        "is_game_over": s.is_game_over,
                        "winner": s.winner.value if s.winner else None,
                    }
            else:
                target = None

            votes[pid] = target
            s.vote_records.append(VoteRecord(voter_id=pid, target_id=target))
            self._log_action(pid, s.sm.round, "vote", decision)

        # 计票
        sheriff = self._get_sheriff()
        tally: dict[int, float] = defaultdict(float)
        for voter_id, target_id in votes.items():
            if target_id is None:
                continue
            weight = s.sm.sheriff_vote_power if voter_id == sheriff else 1.0
            tally[target_id] += weight

        if not tally:
            self._log_public("[投票] 无人得票，平安日")
            return {
                "step_data": {"vote_result": "tie", "tally": {}},
                "dialogues": [{"speaker": "system", "content": "无人得票，平安日"}],
                "deaths": [],
                "is_game_over": s.is_game_over,
                "winner": s.winner.value if s.winner else None,
            }

        max_votes = max(tally.values())
        top = [pid for pid, v in tally.items() if v == max_votes]

        if len(top) == 1:
            eliminated = top[0]
            victim = self._kill_player(eliminated, "voted_out")
            self._log_public(f"[放逐] {eliminated}号玩家被放逐（{tally[eliminated]:.1f}票）")
            deaths = [victim] if victim else []
        else:
            self._log_public(f"[投票] 平票({max_votes}票)，无人被放逐")
            deaths = []

        s.vote_records.clear()
        s.deaths_this_round = deaths

        return {
            "step_data": {"tally": {str(k): v for k, v in tally.items()}, "eliminated": top[0] if len(top) == 1 else None},
            "deaths": deaths,
            "dialogues": [],
            "is_game_over": s.is_game_over,
            "winner": s.winner.value if s.winner else None,
        }

    # ================================================================
    # 阶段结束
    # ================================================================

    def _handle_day_end(self) -> dict:
        s = self._state
        player_dicts = [p.to_dict() for p in s.players]
        winner = s.sm.check_win(player_dicts)

        if winner:
            s.is_game_over = True
            s.winner = winner
            s.win_reason = s.sm.get_win_reason(winner, player_dicts)
            return {
                "step_data": {"game_over": True},
                "is_game_over": True,
                "winner": winner.value,
                "deaths": s.deaths_this_round,
                "dialogues": [],
            }

        return {
            "step_data": {"day_ended": True},
            "is_game_over": False,
            "winner": None,
            "deaths": [],
            "dialogues": [],
        }

    # ================================================================
    # 全自动运行
    # ================================================================

    async def run_game(self, agent_factory: Callable) -> dict:
        """全自动运行整局游戏

        Args:
            agent_factory: async def get_decision(player_id, action_request) -> AgentDecision

        Returns:
            {"winner": str, "reason": str, "rounds": int, "log": list}
        """
        await self.send_game_start()

        while not self._state.is_game_over:
            s = self._state
            phase = s.sm.phase

            # 收集 Agent 决策
            requests = self._build_action_requests()
            decisions: dict[int, AgentDecision] = {}

            if requests:
                # 并行等待所有 Agent 决策（各自有超时）
                tasks = []
                for req in requests:
                    tasks.append(self._collect_decision_with_timeout(agent_factory, req))
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for req, result in zip(requests, results):
                    if isinstance(result, Exception):
                        logger.warning("Agent %d 决策异常: %s，使用兜底", req["player_id"], result)
                        decisions[req["player_id"]] = FallbackDecisions.for_role(
                            req["role"], req["player_id"]
                        )
                    else:
                        decisions[req["player_id"]] = result

            # 推进阶段
            await self.step(decisions)

        return {
            "winner": self._state.winner.value,
            "reason": self._state.win_reason,
            "rounds": self._state.sm.round,
            "log": self._state.action_log,
        }

    async def _collect_decision_with_timeout(self, agent_factory: Callable, req: dict) -> AgentDecision:
        """带超时的 Agent 决策收集"""
        try:
            decision = await asyncio.wait_for(
                agent_factory(req["player_id"], req),
                timeout=req.get("deadline", 30) - time.time(),
            )
            return decision
        except asyncio.TimeoutError:
            return FallbackDecisions.for_role(req["role"], req["player_id"])

    # ================================================================
    # 信息隔离 — 公开/私有消息构建
    # ================================================================

    def get_public_state(self, request_player_id: int) -> dict:
        """获取某玩家可见的公开游戏状态（REST API 用）"""
        s = self._state
        return {
            "game_id": s.game_id,
            "phase": s.sm.phase.value,
            "round": s.sm.round,
            "day_number": s.sm.day_number,
            "alive_players": [p.player_id for p in s.players if p.is_alive],
            "public_log": s.public_log[-50:],  # 最近 50 条
        }

    def build_private_info(self, player_id: int) -> Optional[PrivateInfoMessage]:
        """构建指定玩家的私有信息消息（WebSocket 用）"""
        s = self._state
        player = self._get_player(player_id)
        if not player:
            return None

        payload: dict = {}

        if player.role == Role.WEREWOLF:
            teammates = [p.player_id for p in s.players if p.role == Role.WEREWOLF and p.player_id != player_id]
            payload["teammates"] = teammates
            payload["kill_target"] = s.night_kill_target

        elif player.role == Role.SEER:
            payload["verified"] = {str(k): v for k, v in s.seer_checked.items()}

        elif player.role == Role.WITCH:
            payload["antidote_available"] = (s.witch_saved is None)
            payload["poison_available"] = self.board["rules"].get("witch_has_poison", False) and s.witch_poisoned is None
            payload["night_kill_target"] = s.night_kill_target

        return PrivateInfoMessage(info_type=f"{player.role.value}_info", payload=payload)

    # ================================================================
    # 工具方法
    # ================================================================

    def _get_player(self, player_id: int) -> Optional[Player]:
        for p in self._state.players:
            if p.player_id == player_id:
                return p
        return None

    def _get_players_by_role(self, role: Role) -> list[Player]:
        return [p for p in self._state.players if p.role == role]

    def _get_sheriff(self) -> int | None:
        for p in self._state.players:
            if p.is_sheriff and p.is_alive:
                return p.player_id
        return None

    def _set_sheriff(self, player_id: int) -> None:
        for p in self._state.players:
            p.is_sheriff = (p.player_id == player_id)

    def _kill_player(self, player_id: int, cause: str) -> Optional[DeathRecord]:
        player = self._get_player(player_id)
        if not player or not player.is_alive:
            return None
        player.is_alive = False
        return DeathRecord(player_id=player_id, role=player.role, cause=cause)

    def _log_public(self, message: str) -> None:
        self._state.public_log.append(message)
        logger.info("[PUBLIC] %s", message)

    def _log_private(self, player_id: int, message: str) -> None:
        logger.info("[PRIVATE→%d] %s", player_id, message)

    def _log_action(self, player_id: int, round_num: int, phase: str, decision: AgentDecision) -> None:
        entry = {
            "player_id": player_id,
            "round": round_num,
            "phase": phase,
            "action": decision.action,
            "target": decision.target,
            "thought": decision.thought,
            "content": decision.content,
        }
        self._state.action_log.append(entry)

    # ================================================================
    # 公开属性
    # ================================================================

    @property
    def state(self) -> Optional[GameState]:
        return self._state

    @property
    def phase(self) -> Optional[Phase]:
        return self._state.sm.phase if self._state else None

    @property
    def is_game_over(self) -> bool:
        return self._state.is_game_over if self._state else False
