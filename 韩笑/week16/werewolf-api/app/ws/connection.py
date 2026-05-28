"""WebSocket 连接管理器

职责:
  - 连接池管理（按 game_id + player_id 索引）
  - 消息收发（Pydantic 序列化/反序列化）
  - 广播 / 单播 / 按角色组播
  - 断线检测与清理
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas.messages import (
    ClientMessage,
    ServerMessage,
    parse_client_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])


# ============================================================
# 连接池
# ============================================================

class ConnectionPool:
    """按 game_id → player_id 管理 WebSocket 连接"""

    def __init__(self):
        # {game_id: {player_id: WebSocket}}
        self._connections: dict[str, dict[int, WebSocket]] = {}
        # {game_id: {player_id: asyncio.Queue}} — 每个玩家的入站消息队列
        self._inbound: dict[str, dict[int, asyncio.Queue]] = {}

    # —— 连接生命周期 ——

    async def connect(self, game_id: str, player_id: int, ws: WebSocket) -> None:
        await ws.accept()
        if game_id not in self._connections:
            self._connections[game_id] = {}
            self._inbound[game_id] = {}
        self._connections[game_id][player_id] = ws
        self._inbound[game_id][player_id] = asyncio.Queue()
        logger.info("[WS] 玩家 %d 加入游戏 %s (在线: %d)", player_id, game_id, len(self._connections[game_id]))

    async def disconnect(self, game_id: str, player_id: int) -> None:
        if game_id in self._connections:
            self._connections[game_id].pop(player_id, None)
            if not self._connections[game_id]:
                del self._connections[game_id]
                self._inbound.pop(game_id, None)
            elif game_id in self._inbound:
                self._inbound[game_id].pop(player_id, None)
        logger.info("[WS] 玩家 %d 离开游戏 %s", player_id, game_id)

    def is_connected(self, game_id: str, player_id: int) -> bool:
        return (
            game_id in self._connections
            and player_id in self._connections[game_id]
        )

    def get_online_count(self, game_id: str) -> int:
        return len(self._connections.get(game_id, {}))

    # —— 发送 ——

    async def send(self, game_id: str, player_id: int, msg: ServerMessage) -> bool:
        """单播：发送给指定玩家"""
        ws = self._connections.get(game_id, {}).get(player_id)
        if ws is None:
            return False
        try:
            await ws.send_text(msg.model_dump_json())
            return True
        except Exception:
            logger.exception("[WS] 发送失败 player=%d game=%s", player_id, game_id)
            await self.disconnect(game_id, player_id)
            return False

    async def broadcast(self, game_id: str, msg: ServerMessage) -> int:
        """广播：发送给游戏内全体玩家"""
        count = 0
        if game_id not in self._connections:
            return 0
        for pid in list(self._connections[game_id].keys()):
            if await self.send(game_id, pid, msg):
                count += 1
        return count

    async def send_to_role(
        self, game_id: str, role_filter: str, msg: ServerMessage,
        role_map: dict[int, str] | None = None,
    ) -> int:
        """按角色组播"""
        # 简单实现：发送给所有连接的玩家
        # 复杂信息隔离由 GameMaster 在上层处理
        return await self.broadcast(game_id, msg)

    # —— 接收（入站队列） ——

    async def enqueue_inbound(self, game_id: str, player_id: int, raw: str) -> Optional[ClientMessage]:
        """解析入站消息并放入该玩家的队列"""
        try:
            data = json.loads(raw)
            msg = parse_client_message(data)
            q = self._inbound.get(game_id, {}).get(player_id)
            if q is not None:
                await q.put(msg)
            return msg
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("[WS] 无效消息 player=%d: %s", player_id, str(e)[:100])
            return None

    async def receive(self, game_id: str, player_id: int, timeout: float = 30.0) -> Optional[ClientMessage]:
        """从指定玩家的入站队列获取消息（超时返回 None）"""
        q = self._inbound.get(game_id, {}).get(player_id)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # —— 清理 ——

    async def cleanup_game(self, game_id: str) -> None:
        """清理某局游戏的所有连接"""
        conns = self._connections.pop(game_id, {})
        for ws in list(conns.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._inbound.pop(game_id, None)
        logger.info("[WS] 游戏 %s 全部连接已清理", game_id)


# ============================================================
# 全局单例
# ============================================================

pool = ConnectionPool()


# ============================================================
# WebSocket 端点
# ============================================================

async def _ws_handler(ws: WebSocket, game_id: str, player_id: int):
    """WebSocket 消息循环"""
    await pool.connect(game_id, player_id, ws)

    try:
        while True:
            raw = await ws.receive_text()
            msg = await pool.enqueue_inbound(game_id, player_id, raw)
            if msg is not None:
                logger.debug(
                    "[WS←%d] type=%s game=%s",
                    player_id,
                    msg.type if hasattr(msg, "type") else "?",
                    game_id,
                )
    except WebSocketDisconnect:
        logger.info("[WS] 玩家 %d 断开连接 game=%s", player_id, game_id)
    except Exception:
        logger.exception("[WS] 玩家 %d 异常 game=%s", player_id, game_id)
    finally:
        await pool.disconnect(game_id, player_id)


@router.websocket("/ws/{game_id}/{player_id}")
async def game_websocket(ws: WebSocket, game_id: str, player_id: str):
    """Agent / 观战 实时通信端点

    player_id:
      - 数字 (0-8): Agent 连接
      - "observer": 上帝视角观战

    连接后:
      1. Server → Client: 按阶段发送 action_request / public_broadcast / game_over
      2. Client → Server: 发送 action / speak / ready
    """
    pid = -1 if player_id == "observer" else int(player_id)
    await _ws_handler(ws, game_id, player_id=pid)
