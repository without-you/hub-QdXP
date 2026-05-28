"""Repository 数据访问层 — 封装所有 CRUD 操作"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.db.connection import DatabaseManager
from app.schemas.actions import AgentDecision, MemoryEntry
from app.schemas.messages import DeathRecord, PlayerState, Role, VoteRecord, Winner

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


# ============================================================
# GameRepository
# ============================================================

class GameRepository:
    """对局 CRUD"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    # —— 创建 ——

    def create_game(
        self,
        game_id: str,
        board_type: str = "standard_6",
        llm_model: str = "deepseek-v4-pro",
    ) -> None:
        self.db.execute(
            """INSERT INTO games (game_id, board_type, status, llm_model, created_at)
               VALUES (?, ?, 'created', ?, ?)""",
            (game_id, board_type, llm_model, _now()),
        )

    # —— 更新 ——

    def set_status(self, game_id: str, status: str) -> None:
        self.db.execute(
            "UPDATE games SET status = ? WHERE game_id = ?",
            (status, game_id),
        )

    def set_running(self, game_id: str) -> None:
        self.set_status(game_id, "running")

    def finish_game(self, game_id: str, winner: Winner, reason: str, total_rounds: int) -> None:
        self.db.execute(
            """UPDATE games
               SET status = 'finished', winner = ?, win_reason = ?,
                   total_rounds = ?, finished_at = ?
               WHERE game_id = ?""",
            (winner.value, reason, total_rounds, _now(), game_id),
        )

    # —— 查询 ——

    def get_game(self, game_id: str) -> Optional[dict]:
        row = self.db.fetchone("SELECT * FROM games WHERE game_id = ?", (game_id,))
        return _row_to_dict(row) if row else None

    def get_game_status(self, game_id: str) -> Optional[str]:
        row = self.db.fetchone("SELECT status FROM games WHERE game_id = ?", (game_id,))
        return row["status"] if row else None

    def list_games(self, limit: int = 20) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM games ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_dict(r) for r in rows]


# ============================================================
# PlayerRepository
# ============================================================

class PlayerRepository:
    """玩家状态 CRUD"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def insert_players(self, game_id: str, players: list[PlayerState]) -> None:
        """批量插入玩家"""
        rows = [
            (game_id, p.player_id, p.name, "unknown", 1, 0)
            for p in players
        ]
        self.db.executemany(
            """INSERT INTO players (game_id, player_id, name, role, is_alive, is_sheriff)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )

    def set_role(self, game_id: str, player_id: int, role: Role) -> None:
        self.db.execute(
            "UPDATE players SET role = ? WHERE game_id = ? AND player_id = ?",
            (role.value, game_id, player_id),
        )

    def set_alive(self, game_id: str, player_id: int, alive: bool) -> None:
        self.db.execute(
            "UPDATE players SET is_alive = ? WHERE game_id = ? AND player_id = ?",
            (1 if alive else 0, game_id, player_id),
        )

    def set_sheriff(self, game_id: str, player_id: int) -> None:
        # 先清掉旧警长
        self.db.execute(
            "UPDATE players SET is_sheriff = 0 WHERE game_id = ?", (game_id,)
        )
        self.db.execute(
            "UPDATE players SET is_sheriff = 1 WHERE game_id = ? AND player_id = ?",
            (game_id, player_id),
        )

    def get_players(self, game_id: str) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM players WHERE game_id = ? ORDER BY player_id",
            (game_id,),
        )
        return [_row_to_dict(r) for r in rows]

    def get_alive_players(self, game_id: str) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM players WHERE game_id = ? AND is_alive = 1 ORDER BY player_id",
            (game_id,),
        )
        return [_row_to_dict(r) for r in rows]

    def get_player(self, game_id: str, player_id: int) -> Optional[dict]:
        row = self.db.fetchone(
            "SELECT * FROM players WHERE game_id = ? AND player_id = ?",
            (game_id, player_id),
        )
        return _row_to_dict(row)


# ============================================================
# ActionLogRepository
# ============================================================

class ActionLogRepository:
    """Agent 行动日志 CRUD"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def log_action(
        self,
        game_id: str,
        round_num: int,
        phase: str,
        player_id: int,
        decision: AgentDecision,
    ) -> int:
        """记录一次 Agent 行动，返回自增 ID"""
        cur = self.db.execute(
            """INSERT INTO action_logs (game_id, round, phase, player_id, action, target, thought, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id, round_num, phase, player_id,
                decision.action, decision.target,
                decision.thought, decision.content,
                _now(),
            ),
        )
        return cur.lastrowid

    def get_actions_by_round(self, game_id: str, round_num: int) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM action_logs WHERE game_id = ? AND round = ? ORDER BY id",
            (game_id, round_num),
        )
        return [_row_to_dict(r) for r in rows]

    def get_actions_by_player(self, game_id: str, player_id: int) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM action_logs WHERE game_id = ? AND player_id = ? ORDER BY id",
            (game_id, player_id),
        )
        return [_row_to_dict(r) for r in rows]

    def get_replay_log(self, game_id: str) -> list[dict]:
        """获取完整对局回放日志（按时间排序）"""
        rows = self.db.fetchall(
            "SELECT * FROM action_logs WHERE game_id = ? ORDER BY id",
            (game_id,),
        )
        return [_row_to_dict(r) for r in rows]


# ============================================================
# MemoryRepository
# ============================================================

class MemoryRepository:
    """Agent 记忆持久化"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_memory(self, game_id: str, player_id: int, entry: MemoryEntry) -> int:
        cur = self.db.execute(
            """INSERT INTO agent_memories (game_id, player_id, round, phase, event_type, content, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id, player_id, entry.round, entry.phase,
                entry.event_type,
                json.dumps(entry.content, ensure_ascii=False),
                entry.importance,
            ),
        )
        return cur.lastrowid

    def save_memories(self, game_id: str, player_id: int, entries: list[MemoryEntry]) -> None:
        rows = [
            (game_id, player_id, e.round, e.phase, e.event_type,
             json.dumps(e.content, ensure_ascii=False), e.importance)
            for e in entries
        ]
        self.db.executemany(
            """INSERT INTO agent_memories (game_id, player_id, round, phase, event_type, content, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    def load_memories(
        self,
        game_id: str,
        player_id: int,
        min_importance: int = 0,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        rows = self.db.fetchall(
            """SELECT * FROM agent_memories
               WHERE game_id = ? AND player_id = ? AND importance >= ?
               ORDER BY id DESC
               LIMIT ?""",
            (game_id, player_id, min_importance, limit),
        )
        results = []
        for r in reversed(rows):  # 恢复时间顺序
            content = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
            results.append(MemoryEntry(
                round=r["round"],
                phase=r["phase"],
                event_type=r["event_type"],
                content=content,
                importance=r["importance"],
            ))
        return results

    def load_important_memories(self, game_id: str, player_id: int) -> list[MemoryEntry]:
        """仅加载高重要性记忆（≥3）"""
        return self.load_memories(game_id, player_id, min_importance=3)


# ============================================================
# GameEventRepository
# ============================================================

class GameEventRepository:
    """公开游戏事件（用于回放）"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def log_event(
        self,
        game_id: str,
        round_num: int,
        phase: str,
        event_type: str,
        content: dict | None = None,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO game_events (game_id, round, phase, event_type, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                game_id, round_num, phase, event_type,
                json.dumps(content or {}, ensure_ascii=False),
                _now(),
            ),
        )
        return cur.lastrowid

    def get_events_by_round(self, game_id: str, round_num: int) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT * FROM game_events WHERE game_id = ? AND round = ? ORDER BY id",
            (game_id, round_num),
        )
        return [_row_to_dict(r) for r in rows]

    def get_full_timeline(self, game_id: str) -> list[dict]:
        """获取完整公开时间线"""
        rows = self.db.fetchall(
            "SELECT * FROM game_events WHERE game_id = ? ORDER BY id",
            (game_id,),
        )
        return [_row_to_dict(r) for r in rows]


# ============================================================
# UnitOfWork — 业务层聚合入口
# ============================================================

class UnitOfWork:
    """聚合所有 Repository，提供统一数据访问入口"""

    def __init__(self, db: DatabaseManager | None = None, db_path: str = "werewolf.db"):
        self._db = db or DatabaseManager.get_instance(db_path)
        self.games = GameRepository(self._db)
        self.players = PlayerRepository(self._db)
        self.actions = ActionLogRepository(self._db)
        self.memories = MemoryRepository(self._db)
        self.events = GameEventRepository(self._db)

    # —— 便捷方法 ——

    def persist_game_start(
        self,
        game_id: str,
        board_type: str,
        llm_model: str,
        player_states: list[PlayerState],
    ) -> None:
        """完整持久化游戏开局状态"""
        self._db.begin()
        try:
            self.games.create_game(game_id, board_type, llm_model)
            self.players.insert_players(game_id, player_states)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def persist_roles(self, game_id: str, roles: dict[int, Role]) -> None:
        """持久化角色分配"""
        self._db.begin()
        try:
            for player_id, role in roles.items():
                self.players.set_role(game_id, player_id, role)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def persist_game_end(self, game_id: str, winner: Winner, reason: str, total_rounds: int) -> None:
        self.games.finish_game(game_id, winner, reason, total_rounds)

    def persist_round(
        self,
        game_id: str,
        round_num: int,
        phase: str,
        action_logs: list[dict],
        game_events: list[dict],
        memories: list[dict],
    ) -> None:
        """批量持久化一轮游戏数据"""
        self._db.begin()
        try:
            for a in action_logs:
                self.actions.log_action(
                    game_id, round_num, phase,
                    a["player_id"],
                    a["decision"],
                )
            for e in game_events:
                self.events.log_event(
                    game_id, round_num, phase,
                    e["event_type"], e.get("content", {}),
                )
            for m in memories:
                self.memories.save_memory(
                    game_id, m["player_id"], m["entry"],
                )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def load_player_memories(self, game_id: str, player_id: int) -> list[MemoryEntry]:
        return self.memories.load_important_memories(game_id, player_id)
