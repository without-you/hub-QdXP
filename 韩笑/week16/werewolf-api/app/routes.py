"""REST API 路由 — 游戏生命周期管理

仅用于游戏的创建、配置和元数据查询，不参与回合内实时交互。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.game.game_master import GameMaster
from app.schemas.messages import Phase, Role, Winner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["game"])


# ============================================================
# 请求/响应模型
# ============================================================

class CreateGameRequest(BaseModel):
    board_type: str = "standard_6"
    roles: Optional[dict[str, int]] = None      # e.g. {"werewolf":2, "seer":1, "witch":1, "villager":2}
    player_names: list[str] = Field(min_length=4, max_length=12)
    shuffle: bool = True
    llm_model: str = "deepseek-v4-pro"
    player_styles: Optional[dict[str, str]] = None  # {"0": "bold", "1": "cautious"}


class CreateGameResponse(BaseModel):
    game_id: str
    status: str = "created"


class GameStateResponse(BaseModel):
    game_id: str
    phase: str
    round: int
    day_number: int
    alive_players: list[int]
    is_game_over: bool
    winner: Optional[str] = None
    public_log: list[str] = Field(default_factory=list)


class StepResponse(BaseModel):
    phase: str
    day_number: int
    step_data: dict = Field(default_factory=dict)
    players: list[dict] = Field(default_factory=list)
    dialogues: list[dict] = Field(default_factory=list)
    deaths: list[dict] = Field(default_factory=list)
    is_game_over: bool = False
    winner: Optional[str] = None
    action_requests: list[dict] = Field(default_factory=list)


# ============================================================
# 内存中的游戏实例存储
# ============================================================

_games: dict[str, GameMaster] = {}


def _get_gm(game_id: str) -> GameMaster:
    gm = _games.get(game_id)
    if gm is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return gm


# ============================================================
# REST 端点
# ============================================================

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "werewolf-team-api", "version": "0.1.0"}


@router.post("/games", response_model=CreateGameResponse, status_code=201)
async def create_game(req: CreateGameRequest):
    """创建新对局"""
    board = req.board_type
    gm = GameMaster(board)
    gid = gm.init_game(req.player_names, shuffle_roles=req.shuffle)

    # 存储实例
    _games[gid] = gm

    # 记录到数据库
    try:
        from app.db.repository import UnitOfWork
        from app.schemas.messages import PlayerState
        uow = UnitOfWork()
        player_states = [PlayerState(player_id=p.player_id, name=p.name) for p in gm.state.players]
        uow.persist_game_start(gid, board, req.llm_model, player_states)
        roles_map = {p.player_id: p.role for p in gm.state.players}
        uow.persist_roles(gid, roles_map)
    except Exception:
        logger.exception("数据库记录失败（不影响游戏）")

    logger.info("[API] 创建对局: %s, 玩家: %s", gid, req.player_names)
    return CreateGameResponse(game_id=gid)


@router.get("/games/{game_id}/state", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """获取游戏状态（公开信息）"""
    gm = _get_gm(game_id)
    s = gm.state
    return GameStateResponse(
        game_id=s.game_id,
        phase=s.sm.phase.value,
        round=s.sm.round,
        day_number=s.sm.day_number,
        alive_players=[p.player_id for p in s.players if p.is_alive],
        is_game_over=s.is_game_over,
        winner=s.winner.value if s.winner else None,
        public_log=s.public_log[-30:],
    )


@router.post("/games/{game_id}/step", response_model=StepResponse)
async def step_game(game_id: str):
    """推进游戏一个阶段（调试/手动模式用）"""
    gm = _get_gm(game_id)
    result = await gm.step({})
    # PlayerState → dict
    if "players" in result:
        result["players"] = [
            p.model_dump() if hasattr(p, "model_dump") else p
            for p in result["players"]
        ]
    return StepResponse(**result)


@router.get("/games/{game_id}/replay")
async def get_replay(game_id: str):
    """获取对局回放日志"""
    gm = _get_gm(game_id)

    # 尝试从 DB 补充数据
    db_data = {"actions": [], "events": [], "memories": []}
    try:
        from app.db.repository import UnitOfWork
        uow = UnitOfWork()
        db_data["actions"] = uow.actions.get_replay_log(game_id)
        db_data["events"] = uow.events.get_full_timeline(game_id)
        db_data["game"] = uow.games.get_game(game_id)
    except Exception:
        pass

    return {
        "game_id": game_id,
        "public_log": gm.state.public_log,
        "action_log": gm.state.action_log,
        "winner": gm.state.winner.value if gm.state.winner else None,
        "db_actions": db_data["actions"][-20:],
        "db_events": db_data["events"][-20:],
        "db_game": db_data["game"],
    }


@router.get("/games")
async def list_games():
    """列出活跃对局"""
    return {
        "active_games": len(_games),
        "games": [
            {
                "game_id": gid,
                "phase": gm.state.sm.phase.value if gm.state else "?",
                "round": gm.state.sm.round if gm.state else 0,
                "players": len(gm.state.players) if gm.state else 0,
            }
            for gid, gm in _games.items()
        ],
    }


@router.post("/games/{game_id}/start")
async def start_game(game_id: str):
    """启动游戏循环（WebSocket 模式）

    所有 Agent 必须已在 /ws/{game_id}/{player_id} 连接并就绪。
    调用此端点后，GameOrchestrator 将驱动整局游戏。
    """
    gm = _get_gm(game_id)
    from app.ws.orchestrator import GameOrchestrator, register_orchestrator
    from app.ws.connection import pool

    connected = pool.get_online_count(game_id)
    total = len(gm.state.players)
    if connected < total:
        return {
            "status": "waiting",
            "connected": connected,
            "total": total,
            "message": f"等待 {total - connected} 个 Agent 连接",
        }

    orch = GameOrchestrator(gm, timeout=30.0)
    register_orchestrator(game_id, orch)

    # 后台运行游戏循环
    import asyncio
    task = asyncio.create_task(orch.run())

    return {
        "status": "started",
        "game_id": game_id,
        "connected": connected,
    }


# 后台运行中的游戏状态
_running_games: dict[str, dict] = {}


@router.post("/games/{game_id}/auto-play-async")
async def auto_play_async(game_id: str, llm: bool = False):
    """异步启动自动演示 — 立即返回，前端轮询 state 获取进度"""
    import asyncio

    gm = _get_gm(game_id)
    s = gm.state
    if s.is_game_over:
        return {"status": "already_finished"}

    _running_games[game_id] = {
        "running": True,
        "llm": llm,
        "started_at": __import__("time").time(),
    }

    # 后台运行
    asyncio.create_task(_run_auto_play(game_id, llm))

    return {
        "status": "started",
        "game_id": game_id,
        "mode": "llm" if llm else "random",
        "message": f"游戏已在后台启动，请轮询 /state 获取进度",
    }


async def _run_auto_play(game_id: str, llm: bool):
    """后台执行自动演示"""
    try:
        result = await _auto_play_game(game_id, llm)
        _running_games[game_id] = {"running": False, "result": result}
    except Exception as e:
        logger.exception("后台自动演示失败: %s", e)
        _running_games[game_id] = {"running": False, "error": str(e)}


@router.get("/games/{game_id}/auto-play-status")
async def auto_play_status(game_id: str):
    """查询后台自动演示状态"""
    info = _running_games.get(game_id, {})
    gm = _get_gm(game_id)
    s = gm.state

    # 返回当前进度 + 已有的 action_log（增量）
    return {
        "running": info.get("running", False),
        "mode": info.get("llm", False),
        "phase": s.sm.phase.value,
        "round": s.sm.round,
        "alive": [p.player_id for p in s.players if p.is_alive],
        "player_roles": [{"player_id": p.player_id, "name": p.name, "role": p.role.value} for p in s.players],
        "is_game_over": s.is_game_over,
        "winner": s.winner.value if s.winner else None,
        "reason": s.win_reason,
        "action_log": s.action_log,        # 增量返回已有行动
        "public_log": s.public_log[-20:],  # 最近20条
        "result": info.get("result"),       # 游戏结束后才非空
    }


# 实际执行逻辑
async def _auto_play_game(game_id: str, llm: bool) -> dict:
    """执行自动演示并返回完整结果"""
    import random
    from app.schemas.actions import AgentDecision
    from app.schemas.messages import Phase, Role

    gm = _get_gm(game_id)
    s = gm.state

    # —— LLM 模式 ——
    agents = {}
    if llm:
        from app.config import get_llm_config
        llm_cfg = get_llm_config()
        api_key = llm_cfg.get("api_key", "")
        if not api_key:
            return {"status": "error", "message": "请在 config.yaml 的 llm.api_key 中填写 API Key"}

        from app.agents.llm_adapter import create_deepseek_adapter
        from app.agents.werewolf_agent import WerewolfAgent
        from app.agents.seer_agent import SeerAgent
        from app.agents.witch_agent import WitchAgent
        from app.agents.villager_agent import VillagerAgent

        adapter = create_deepseek_adapter()
        role_cls = {
            Role.WEREWOLF: WerewolfAgent,
            Role.SEER: SeerAgent,
            Role.WITCH: WitchAgent,
            Role.VILLAGER: VillagerAgent,
        }
        for p in s.players:
            cls = role_cls[p.role]
            agent = cls(player_id=p.player_id, name=p.name, llm_adapter=adapter)
            if p.role == Role.WEREWOLF:
                teammates = [o.player_id for o in s.players if o.role == Role.WEREWOLF and o.player_id != p.player_id]
                agent._teammates = teammates
            agents[p.player_id] = agent

    def _random_target(exclude_self=True, exclude_wolves=False):
        candidates = [p for p in s.players if p.is_alive]
        if exclude_self:
            candidates = [p for p in candidates if p.player_id != pid]
        if exclude_wolves:
            candidates = [p for p in candidates if p.role != Role.WEREWOLF]
        return random.choice(candidates).player_id if candidates else None

    max_steps = 200
    step_count = 0
    while not s.is_game_over and step_count < max_steps:
        phase = s.sm.phase
        round_num = s.sm.round
        alive = [p.player_id for p in s.players if p.is_alive]
        decisions = {}

        for p in s.players:
            if not p.is_alive or p.role not in s.sm.get_active_roles():
                continue
            pid = p.player_id

            if llm and pid in agents:
                agent = agents[pid]
                valid_actions = s.sm.get_valid_actions(p.role, pid, alive)
                context = {
                    "my_role": p.role.value, "my_id": pid, "my_name": p.name, "my_style": "balanced",
                    "phase": phase.value, "round": round_num, "valid_actions": valid_actions,
                    "alive_players": alive,
                    "teammates": getattr(agent, '_teammates', []),
                    "unchecked": alive, "gold_water": [], "wolf_check": [], "checked_count": 0,
                    "revealed_info": [], "recent_memory": [], "kill_history": [],
                    "has_antidote": getattr(agent, '_has_antidote', True),
                    "has_poison": getattr(agent, '_has_poison', False),
                    "night_kill_target": s.night_kill_target, "saved_players": [],
                    "is_first_night": round_num == 1, "suspicions": {}, "observed_claims": {},
                }
                try:
                    decision = await agent._decide(context, phase, valid_actions)
                    decisions[pid] = decision
                except Exception as e:
                    logger.warning(f"Agent {pid} LLM调用失败: {e}")
                    decisions[pid] = AgentDecision(action="skip", target=None, thought=f"[LLM异常] {str(e)[:100]}")
            else:
                if phase == Phase.NIGHT_WOLF and p.role == Role.WEREWOLF:
                    target = _random_target(exclude_self=True, exclude_wolves=True)
                    decisions[pid] = AgentDecision(action="kill", target=target, thought=f"[自动] 狼人{pid}号击杀{target}号")
                elif phase == Phase.NIGHT_SEER and p.role == Role.SEER:
                    target = _random_target(exclude_self=True)
                    decisions[pid] = AgentDecision(action="verify", target=target, thought=f"[自动] 预言家{pid}号查验{target}号")
                elif phase == Phase.NIGHT_WITCH and p.role == Role.WITCH:
                    decisions[pid] = AgentDecision(action="nosave", target=None, thought=f"[自动] 女巫{pid}号不使用解药")
                elif phase == Phase.VOTE:
                    target = _random_target(exclude_self=True)
                    decisions[pid] = AgentDecision(action="vote", target=target, thought=f"[自动] {pid}号投票放逐{target}号")
                elif phase == Phase.SPEECH:
                    decisions[pid] = AgentDecision(action="speak", content=f"我是{pid}号，我认为我们应该仔细分析每个人的发言逻辑。", thought=f"[自动] {pid}号发言")

        await gm.step(decisions)
        step_count += 1

    # 持久化
    try:
        from app.db.repository import UnitOfWork
        from app.schemas.actions import MemoryEntry
        uow = UnitOfWork()
        uow.persist_game_end(game_id, s.winner, s.win_reason or "unknown", s.sm.round)
        for p in s.players:
            uow.players.set_role(game_id, p.player_id, p.role)
            uow.players.set_alive(game_id, p.player_id, p.is_alive)
        for entry in s.action_log:
            uow.actions.log_action(game_id, entry.get("round", 0), entry.get("phase", ""),
                                    entry.get("player_id", 0),
                                    AgentDecision(action=entry.get("action","skip"), target=entry.get("target"),
                                                  thought=entry.get("thought",""), content=entry.get("content","")))
        for log_text in s.public_log:
            event_type = "system"
            if "发言" in log_text: event_type = "player_speak"
            elif "死讯" in log_text: event_type = "player_death"
            elif "投票" in log_text or "放逐" in log_text: event_type = "vote_result"
            elif "警长" in log_text: event_type = "sheriff_elected"
            uow.events.log_event(game_id, s.sm.round, "", event_type, {"message": log_text})
        if llm and agents:
            for pid, agent in agents.items():
                for mem in agent.memory:
                    uow.memories.save_memory(game_id, pid, mem)
        logger.info("对局 %s 数据已全部持久化", game_id)
    except Exception as e:
        logger.exception("持久化失败: %s", e)

    player_roles = [{"player_id": p.player_id, "name": p.name, "role": p.role.value} for p in s.players]
    return {
        "status": "finished", "game_id": game_id,
        "winner": s.winner.value if s.winner else None, "reason": s.win_reason,
        "rounds": s.sm.round, "steps": step_count,
        "mode": "llm" if llm else "random", "player_roles": player_roles,
        "public_log": s.public_log[-30:], "action_log": s.action_log[-30:],
    }


@router.post("/games/{game_id}/auto-play")
async def auto_play_game(game_id: str, llm: bool = False):
    """同步自动演示（保留向后兼容）"""
    result = await _auto_play_game(game_id, llm)
    return result
    """自动演示模式

    参数:
        llm: false=随机决策(快速演示), true=DeepSeek在线模型驱动(需DEEPSEEK_API_KEY)
    """
    import asyncio, random
    from app.schemas.actions import AgentDecision
    from app.schemas.messages import Phase, Role, ActionRequestMessage

    gm = _get_gm(game_id)
    s = gm.state

    if s.is_game_over:
        return {"status": "already_finished", "winner": s.winner.value if s.winner else None}

    # —— LLM 模式：创建 Agent 实例 ——
    agents = {}
    if llm:
        from app.config import get_llm_config
        llm_cfg = get_llm_config()
        api_key = llm_cfg.get("api_key", "")
        if not api_key:
            return {"status": "error", "message": "请在 config.yaml 的 llm.api_key 中填写 API Key"}

        from app.agents.llm_adapter import create_deepseek_adapter
        from app.agents.werewolf_agent import WerewolfAgent
        from app.agents.seer_agent import SeerAgent
        from app.agents.witch_agent import WitchAgent
        from app.agents.villager_agent import VillagerAgent

        adapter = create_deepseek_adapter(api_key=api_key)
        role_cls = {
            Role.WEREWOLF: WerewolfAgent,
            Role.SEER: SeerAgent,
            Role.WITCH: WitchAgent,
            Role.VILLAGER: VillagerAgent,
        }
        for p in s.players:
            cls = role_cls[p.role]
            agent = cls(player_id=p.player_id, name=p.name, llm_adapter=adapter)
            # 初始化狼人队友信息
            if p.role == Role.WEREWOLF:
                teammates = [
                    other.player_id for other in s.players
                    if other.role == Role.WEREWOLF and other.player_id != p.player_id
                ]
                agent._teammates = teammates
            agents[p.player_id] = agent

    def _random_target(exclude_self=True, exclude_wolves=False):
        candidates = [p for p in s.players if p.is_alive]
        if exclude_self:
            candidates = [p for p in candidates if p.player_id != pid]
        if exclude_wolves:
            candidates = [p for p in candidates if p.role != Role.WEREWOLF]
        return random.choice(candidates).player_id if candidates else None

    max_steps = 200
    step_count = 0
    while not s.is_game_over and step_count < max_steps:
        phase = s.sm.phase
        round_num = s.sm.round
        alive = [p.player_id for p in s.players if p.is_alive]
        decisions = {}

        for p in s.players:
            if not p.is_alive:
                continue
            if p.role not in s.sm.get_active_roles():
                continue
            pid = p.player_id

            if llm and pid in agents:
                # ——— LLM 驱动 ———
                agent = agents[pid]
                valid_actions = s.sm.get_valid_actions(p.role, pid, alive)
                context = {
                    "my_role": p.role.value,
                    "my_id": pid,
                    "my_name": p.name,
                    "my_style": "balanced",
                    "phase": phase.value,
                    "round": round_num,
                    "valid_actions": valid_actions,
                    "alive_players": alive,
                    "teammates": getattr(agent, '_teammates', []),
                    "unchecked": alive,
                    "gold_water": [],
                    "wolf_check": [],
                    "checked_count": 0,
                    "revealed_info": [],
                    "recent_memory": [],
                    "kill_history": [],
                    "has_antidote": getattr(agent, '_has_antidote', True),
                    "has_poison": getattr(agent, '_has_poison', False),
                    "night_kill_target": s.night_kill_target,
                    "saved_players": [],
                    "is_first_night": round_num == 1,
                    "suspicions": {},
                    "observed_claims": {},
                }
                try:
                    # decide() 内部调用 LLM + JSON 解析，返回的 AgentDecision 已含正确的 thought
                    decision = await agent._decide(context, phase, valid_actions)
                    decisions[pid] = decision
                except Exception as e:
                    logger.warning(f"Agent {pid} LLM调用失败，使用随机: {e}")
                    decisions[pid] = AgentDecision(
                        action="skip", target=None, thought=f"[LLM异常] {str(e)[:100]}",
                    )
            else:
                # ——— 随机决策（快速演示）———
                if phase == Phase.NIGHT_WOLF and p.role == Role.WEREWOLF:
                    target = _random_target(exclude_self=True, exclude_wolves=True)
                    decisions[pid] = AgentDecision(
                        action="kill", target=target,
                        thought=f"[自动] 狼人{pid}号击杀{target}号",
                    )
                elif phase == Phase.NIGHT_SEER and p.role == Role.SEER:
                    target = _random_target(exclude_self=True)
                    decisions[pid] = AgentDecision(
                        action="verify", target=target,
                        thought=f"[自动] 预言家{pid}号查验{target}号",
                    )
                elif phase == Phase.NIGHT_WITCH and p.role == Role.WITCH:
                    decisions[pid] = AgentDecision(
                        action="nosave", target=None,
                        thought=f"[自动] 女巫{pid}号不使用解药",
                    )
                elif phase == Phase.VOTE:
                    target = _random_target(exclude_self=True)
                    decisions[pid] = AgentDecision(
                        action="vote", target=target,
                        thought=f"[自动] {pid}号投票放逐{target}号",
                    )
                elif phase == Phase.SPEECH:
                    decisions[pid] = AgentDecision(
                        action="speak",
                        content=f"我是{pid}号，我认为我们应该仔细分析每个人的发言逻辑。",
                        thought=f"[自动] {pid}号发言",
                    )

        await gm.step(decisions)
        step_count += 1

    # 持久化 — 写入全部三张表
    try:
        from app.db.repository import UnitOfWork
        from app.schemas.actions import MemoryEntry

        uow = UnitOfWork()

        # 1. 标记对局结束
        uow.persist_game_end(game_id, s.winner, s.win_reason or "unknown", s.sm.round)

        # 2. 玩家信息
        for p in s.players:
            if p.role.value != "unknown":
                uow.players.set_role(game_id, p.player_id, p.role)
            uow.players.set_alive(game_id, p.player_id, p.is_alive)

        # 3. 行动日志 → action_logs 表
        for entry in s.action_log:
            decision = AgentDecision(
                action=entry.get("action", "skip"),
                target=entry.get("target"),
                thought=entry.get("thought", ""),
                content=entry.get("content", ""),
            )
            uow.actions.log_action(
                game_id,
                entry.get("round", 0),
                entry.get("phase", ""),
                entry.get("player_id", 0),
                decision,
            )

        # 4. 公开日志 → game_events 表
        for log_text in s.public_log:
            # 推断事件类型
            event_type = "system"
            if "发言" in log_text:
                event_type = "player_speak"
            elif "死讯" in log_text:
                event_type = "player_death"
            elif "投票" in log_text or "放逐" in log_text:
                event_type = "vote_result"
            elif "警长" in log_text:
                event_type = "sheriff_elected"
            elif "平安夜" in log_text:
                event_type = "peaceful_night"

            round_num = s.sm.round  # 近似，日志不记录精确轮次
            uow.events.log_event(
                game_id, round_num, "",
                event_type,
                {"message": log_text},
            )

        # 5. Agent 记忆 → agent_memories 表（LLM 模式下 Agent 有真实 CoT）
        if llm and agents:
            for pid, agent in agents.items():
                for mem in agent.memory:
                    uow.memories.save_memory(game_id, pid, mem)

        logger.info("对局 %s 数据已全部持久化: %d actions, %d events, %d memories",
                    game_id, len(s.action_log), len(s.public_log),
                    sum(len(a.memory) for a in agents.values()) if agents else 0)

    except Exception as e:
        logger.exception("持久化失败: %s", e)

    # 构建玩家身份信息
    player_roles = [
        {"player_id": p.player_id, "name": p.name, "role": p.role.value}
        for p in s.players
    ]

    return {
        "status": "finished",
        "game_id": game_id,
        "winner": s.winner.value if s.winner else None,
        "reason": s.win_reason,
        "rounds": s.sm.round,
        "steps": step_count,
        "mode": "llm" if llm else "random",
        "player_roles": player_roles,
        "public_log": s.public_log[-30:],
        "action_log": s.action_log[-30:],
    }


@router.post("/config/reload")
async def reload_config():
    """热重载 Prompt 配置"""
    return {"status": "ok", "message": "Prompt templates will be reloaded on next request"}
