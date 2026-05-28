"""GameOrchestrator — 游戏实时流程调度

连接 WebSocket + GameMaster + Agent，驱动整局游戏:
  1. 等待所有 Agent 连接并就绪
  2. 按阶段发送 action_request
  3. 等待 Agent 决策（带超时兜底）
  4. 推进 step + 广播结果
  5. 终局通知
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.game.game_master import GameMaster
from app.schemas.actions import AgentDecision, FallbackDecisions
from app.schemas.messages import (
    ActionRequestMessage,
    GameOverMessage,
    GameStartMessage,
    Phase,
    PhaseChangeMessage,
    PlayerState,
    PrivateInfoMessage,
    PublicBroadcastMessage,
    Role,
    Winner,
)
from app.ws.connection import ConnectionPool, pool

logger = logging.getLogger(__name__)


class GameOrchestrator:
    """一局游戏的实时流程调度器"""

    def __init__(self, gm: GameMaster, timeout: float = 30.0):
        self.gm = gm
        self.timeout = timeout
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def game_id(self) -> str:
        return self.gm.state.game_id

    # ================================================================
    # 启动游戏循环
    # ================================================================

    async def run(self) -> dict:
        """启动游戏循环，阻塞直到游戏结束"""
        self._running = True
        s = self.gm.state

        try:
            # 1. 发送 game_start
            await self._broadcast_game_start()

            # 2. 等待所有玩家 ready
            total = len(s.players)
            await self._wait_all_ready(total, timeout=self.timeout * 2)

            # 3. 主循环
            while not s.is_game_over and self._running:
                await self._tick()

            # 4. 游戏结束
            return await self._finish_game()

        except Exception:
            logger.exception("[Orchestrator] 游戏异常 game=%s", self.game_id)
            raise
        finally:
            self._running = False

    # ================================================================
    # Tick — 单步
    # ================================================================

    async def _tick(self) -> None:
        s = self.gm.state
        sm = s.sm
        phase = sm.phase

        # 构建 action_requests
        requests = self.gm._build_action_requests()

        if not requests:
            # 系统阶段 → 直接推进
            result = await self.gm.step({})
            await self._broadcast_phase_result(result)
            return

        # 发送 action_request 给对应 Agent
        deadline = asyncio.get_event_loop().time() + self.timeout
        for req in requests:
            pid = req["player_id"]
            await pool.send(self.game_id, pid, ActionRequestMessage(
                phase=Phase(req["phase"]),
                round=req["round"],
                valid_actions=req["valid_actions"],
                deadline=req.get("deadline", deadline),
                context=req.get("context", {}),
            ))

        # 等待决策（并行）
        decisions = await self._collect_decisions(
            [r["player_id"] for r in requests],
            deadline,
        )

        # 推进
        result = await self.gm.step(decisions)
        await self._broadcast_phase_result(result)

    # ================================================================
    # 决策收集
    # ================================================================

    async def _collect_decisions(
        self, player_ids: list[int], deadline: float
    ) -> dict[int, AgentDecision]:
        """并行等待所有 Agent 决策，超时走兜底"""
        decisions: dict[int, AgentDecision] = {}

        async def get_decision(pid: int) -> AgentDecision:
            try:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                msg = await pool.receive(self.game_id, pid, timeout=remaining)
                if msg is None:
                    raise asyncio.TimeoutError()
                if hasattr(msg, "action"):
                    return AgentDecision(
                        action=getattr(msg, "action", "skip"),
                        target=getattr(msg, "target", None),
                        thought=getattr(msg, "thought", ""),
                        content=getattr(msg, "content", ""),
                    )
                return FallbackDecisions.for_role("villager", pid)
            except (asyncio.TimeoutError, Exception):
                player = self.gm._get_player(pid)
                role = player.role.value if player else "villager"
                logger.warning("[Orchestrator] Agent %d 超时，使用兜底", pid)
                return FallbackDecisions.for_role(role, pid)

        tasks = [get_decision(pid) for pid in player_ids]
        results = await asyncio.gather(*tasks)
        for pid, decision in zip(player_ids, results):
            decisions[pid] = decision

        return decisions

    # ================================================================
    # 广播
    # ================================================================

    async def _broadcast_game_start(self) -> None:
        s = self.gm.state
        for p in s.players:
            teammates = [
                other.player_id
                for other in s.players
                if other.role == Role.WEREWOLF and other.player_id != p.player_id
            ]
            await pool.send(p.player_id, self.game_id, GameStartMessage(
                player_id=p.player_id,
                role=p.role,
                teammates=teammates if p.role == Role.WEREWOLF else [],
                player_names={pl.player_id: pl.name for pl in s.players},
            ))

    async def _broadcast_phase_result(self, result: dict) -> None:
        s = self.gm.state

        # 阶段变更
        await pool.broadcast(self.game_id, PhaseChangeMessage(
            phase=s.sm.phase,
            round=s.sm.round,
            timeout_sec=self.timeout,
        ))

        # 死讯
        for death in result.get("deaths", []):
            await pool.broadcast(self.game_id, PublicBroadcastMessage(
                event="player_death",
                round=s.sm.round,
                content={"player_id": death.player_id if hasattr(death, "player_id") else death.get("player_id", 0),
                         "role": death.role.value if hasattr(death, "role") else death.get("role", ""),
                         "cause": death.cause if hasattr(death, "cause") else death.get("cause", "")},
            ))

        # 发言
        for dialogue in result.get("dialogues", []):
            await pool.broadcast(self.game_id, PublicBroadcastMessage(
                event="player_speak",
                round=s.sm.round,
                content=dialogue,
            ))

        # 私有信息（按角色推送）
        for p in s.players:
            if p.is_alive:
                private = self.gm.build_private_info(p.player_id)
                if private:
                    await pool.send(p.player_id, self.game_id, private)

    async def _wait_all_ready(self, total: int, timeout: float) -> None:
        """等待所有玩家发送 ready"""
        ready_set: set[int] = set()
        deadline = asyncio.get_event_loop().time() + timeout

        async def wait_one(pid: int):
            while pid not in ready_set:
                msg = await pool.receive(self.game_id, pid, timeout=2.0)
                if msg is not None and hasattr(msg, "type") and msg.type == "ready":
                    ready_set.add(pid)
                    logger.info("[Orchestrator] 玩家 %d ready (%d/%d)", pid, len(ready_set), total)
                    return

        # 并行等待所有玩家
        s = self.gm.state
        tasks = [wait_one(p.player_id) for p in s.players]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("等待 ready 超时，已就绪: %d/%d", len(ready_set), total)

    async def _finish_game(self) -> dict:
        s = self.gm.state
        winner = s.winner or Winner.GOOD

        await pool.broadcast(self.game_id, GameOverMessage(
            winner=winner,
            reason=s.win_reason or "unknown",
            players=[p.state for p in s.players],
        ))

        # 持久化到数据库
        try:
            from app.db.repository import UnitOfWork
            uow = UnitOfWork()
            uow.persist_game_end(self.game_id, winner, s.win_reason or "unknown", s.sm.round)
            logger.info("游戏结果已持久化: %s winner=%s", self.game_id, winner.value)
        except Exception:
            logger.exception("持久化游戏结果失败")

        # 清理连接
        await pool.cleanup_game(self.game_id)

        return {"game_id": self.game_id, "winner": winner.value, "reason": s.win_reason}


# ============================================================
# 全局调度器注册表
# ============================================================

_orchestrators: dict[str, GameOrchestrator] = {}


def get_orchestrator(game_id: str) -> Optional[GameOrchestrator]:
    return _orchestrators.get(game_id)


def register_orchestrator(game_id: str, orch: GameOrchestrator) -> None:
    _orchestrators[game_id] = orch


def unregister_orchestrator(game_id: str) -> None:
    _orchestrators.pop(game_id, None)
