"""DatabaseManager — SQLite 连接管理 + 自动迁移

特点:
  - WAL 模式，支持并发读
  - 自动建表
  - 外键强制
  - 连接复用
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseManager:
    """SQLite 数据库管理器（进程级单例）"""

    _instance: Optional[DatabaseManager] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = "werewolf.db"):
        self.db_path = db_path
        self._local = threading.local()

    # ================================================================
    # 单例
    # ================================================================

    @classmethod
    def get_instance(cls, db_path: str = "werewolf.db") -> DatabaseManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path=db_path)
                    cls._instance._init_db()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                try:
                    if hasattr(cls._instance._local, "conn"):
                        cls._instance._local.conn.close()
                except Exception:
                    pass
                cls._instance = None

    # ================================================================
    # 初始化
    # ================================================================

    def _init_db(self) -> None:
        """首次创建时执行建表迁移"""
        conn = self._get_conn()
        try:
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
            conn.executescript(schema)
            conn.commit()
            logger.info("数据库初始化完成: %s", self.db_path)
        except Exception:
            logger.exception("数据库初始化失败")

    # ================================================================
    # 连接获取
    # ================================================================

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        return self._get_conn()

    # ================================================================
    # 上下文管理器
    # ================================================================

    def connection(self) -> sqlite3.Connection:
        """获取连接（用于 with 语句或手动管理）"""
        return self._get_conn()

    def cursor(self):
        """便捷：获取游标（调用方负责 commit）"""
        return self._get_conn().cursor()

    def commit(self) -> None:
        self._get_conn().commit()

    def rollback(self) -> None:
        self._get_conn().rollback()

    # ================================================================
    # 便捷执行
    # ================================================================

    def execute(self, sql: str, params: tuple | dict | None = None) -> sqlite3.Cursor:
        """执行 SQL，自动 commit"""
        conn = self._get_conn()
        cur = conn.execute(sql, params or ())
        conn.commit()
        return cur

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        conn = self._get_conn()
        cur = conn.executemany(sql, params_list)
        conn.commit()
        return cur

    def fetchone(self, sql: str, params: tuple | dict | None = None) -> Optional[sqlite3.Row]:
        cur = self._get_conn().execute(sql, params or ())
        return cur.fetchone()

    def fetchall(self, sql: str, params: tuple | dict | None = None) -> list[sqlite3.Row]:
        cur = self._get_conn().execute(sql, params or ())
        return cur.fetchall()

    # ================================================================
    # 事务
    # ================================================================

    def begin(self) -> None:
        self.execute("BEGIN")

    # ================================================================
    # 清理
    # ================================================================

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def vacuum(self) -> None:
        self._get_conn().execute("VACUUM")
