"""WebSocket 消息协议 — Pydantic Discriminated Union 严格校验"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ============================================================
# 枚举
# ============================================================

class Role(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"


class Phase(str, Enum):
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH = "night_witch"
    NIGHT_RESULT = "night_result"
    DAY_START = "day_start"
    SPEECH = "speech"
    VOTE = "vote"
    DAY_END = "day_end"


class ActionType(str, Enum):
    KILL = "kill"
    VERIFY = "verify"
    SAVE = "save"         # 女巫解药
    POISON = "poison"     # 女巫毒药
    VOTE = "vote"
    SPEAK = "speak"
    SKIP = "skip"
    SELF_DESTRUCT = "self_destruct"


class Winner(str, Enum):
    GOOD = "good"
    EVIL = "evil"


# ============================================================
# 游戏状态模型（共享）
# ============================================================

class PlayerState(BaseModel):
    """对局中玩家的公开快照"""
    player_id: int = Field(ge=0, le=8, description="座位号 0-8")
    name: str
    is_alive: bool = True
    is_sheriff: bool = False


class DeathRecord(BaseModel):
    player_id: int
    role: Role
    cause: str  # "killed_by_wolves" | "poisoned" | "voted_out" | "shot_by_hunter"


class VoteRecord(BaseModel):
    voter_id: int
    target_id: int | None  # None = 弃票


# ============================================================
# Server → Client 下行消息
# ============================================================

class _ServerBase(BaseModel):
    timestamp: float = Field(default_factory=time.time)


class GameStartMessage(_ServerBase):
    type: Literal["game_start"] = "game_start"
    player_id: int
    role: Role
    teammates: list[int] = Field(default_factory=list)  # 仅狼人有值
    player_names: dict[int, str]  # {seat: name}


class PhaseChangeMessage(_ServerBase):
    type: Literal["phase_change"] = "phase_change"
    phase: Phase
    round: int
    timeout_sec: int = 30


class PrivateInfoMessage(_ServerBase):
    """角色私有信息，仅特定角色可收到 — 信息隔离核心"""
    type: Literal["private_info"] = "private_info"
    info_type: str  # "verify_result" | "kill_target" | "teammate_chat" | "saved_notice"
    payload: dict = Field(default_factory=dict)


class PublicBroadcastMessage(_ServerBase):
    type: Literal["public_broadcast"] = "public_broadcast"
    event: str  # "player_speak" | "player_death" | "vote_result" | "sheriff_elected"
    round: int
    content: dict = Field(default_factory=dict)


class ActionRequestMessage(_ServerBase):
    type: Literal["action_request"] = "action_request"
    phase: Phase
    round: int
    valid_actions: list[str]  # e.g. ["kill_1","kill_2","skip"]
    deadline: float  # Unix timestamp, 超时后 GM 走兜底
    context: dict = Field(default_factory=dict)


class GameOverMessage(_ServerBase):
    type: Literal["game_over"] = "game_over"
    winner: Winner
    reason: str  # "all_wolves_dead" | "all_gods_dead" | "all_villagers_dead"
    players: list[PlayerState]


ServerMessage = Annotated[
    Union[
        GameStartMessage,
        PhaseChangeMessage,
        PrivateInfoMessage,
        PublicBroadcastMessage,
        ActionRequestMessage,
        GameOverMessage,
    ],
    Field(discriminator="type"),
]


# ============================================================
# Client → Server 上行消息
# ============================================================

class _ClientBase(BaseModel):
    timestamp: float = Field(default_factory=time.time)


class ReadyMessage(_ClientBase):
    type: Literal["ready"] = "ready"


class ActionMessage(_ClientBase):
    type: Literal["action"] = "action"
    action: ActionType
    target: int | None = None  # None for skip
    thought: str = ""           # Agent CoT 内心独白（日志用）


class SpeakMessage(_ClientBase):
    type: Literal["speak"] = "speak"
    content: str = Field(min_length=1, max_length=2000)
    thought: str = ""


class SelfDestructMessage(_ClientBase):
    type: Literal["self_destruct"] = "self_destruct"
    thought: str = ""


ClientMessage = Annotated[
    Union[
        ReadyMessage,
        ActionMessage,
        SpeakMessage,
        SelfDestructMessage,
    ],
    Field(discriminator="type"),
]

# ============================================================
# 消息解析 helper（TypeAdapter 支持 Discriminated Union）
# ============================================================

from pydantic import TypeAdapter

_client_adapter = TypeAdapter(ClientMessage)
_server_adapter = TypeAdapter(ServerMessage)


def parse_client_message(data: dict) -> ClientMessage:
    """解析客户端上行消息"""
    return _client_adapter.validate_python(data)


def parse_server_message(data: dict) -> ServerMessage:
    """解析服务端下行消息"""
    return _server_adapter.validate_python(data)


def dump_message(msg: ServerMessage | ClientMessage) -> str:
    """序列化消息为 JSON 字符串"""
    return msg.model_dump_json()
