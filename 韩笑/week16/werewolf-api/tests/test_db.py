"""数据库层测试 — Schema / Repository / UnitOfWork"""

from __future__ import annotations

import os
import tempfile

import pytest

from app.db.connection import DatabaseManager
from app.db.repository import (
    ActionLogRepository,
    GameEventRepository,
    GameRepository,
    MemoryRepository,
    PlayerRepository,
    UnitOfWork,
)
from app.schemas.actions import AgentDecision, MemoryEntry
from app.schemas.messages import PlayerState, Role, Winner


# ============================================================
# Fixtures — 每个测试使用独立的临时数据库
# ============================================================

@pytest.fixture
def db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="werewolf_test_")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def db(db_path) -> DatabaseManager:
    DatabaseManager.reset_instance()
    return DatabaseManager.get_instance(db_path)


@pytest.fixture
def uow(db) -> UnitOfWork:
    return UnitOfWork(db)


# ============================================================
# DatabaseManager
# ============================================================

class TestDatabaseManager:
    def test_singleton_same_instance(self, db, db_path):
        db2 = DatabaseManager.get_instance(db_path)
        assert db is db2

    def test_tables_created(self, db):
        tables = db.fetchall("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [r["name"] for r in tables]
        assert "games" in table_names
        assert "players" in table_names
        assert "action_logs" in table_names
        assert "agent_memories" in table_names
        assert "game_events" in table_names

    def test_foreign_keys_enabled(self, db):
        row = db.fetchone("PRAGMA foreign_keys")
        assert row[0] == 1

    def test_wal_mode(self, db):
        row = db.fetchone("PRAGMA journal_mode")
        assert row[0] == "wal"

    def test_persists_across_sessions(self, db, db_path):
        """相同路径重新获取应保留数据"""
        db.execute(
            "INSERT INTO games (game_id, board_type, status) VALUES ('g1', 'standard_6', 'running')"
        )
        DatabaseManager.reset_instance()
        db2 = DatabaseManager.get_instance(db_path)
        row = db2.fetchone("SELECT * FROM games WHERE game_id = ?", ("g1",))
        assert row is not None
        assert row["status"] == "running"


# ============================================================
# GameRepository
# ============================================================

class TestGameRepository:
    @pytest.fixture
    def repo(self, db) -> GameRepository:
        return GameRepository(db)

    def test_create_and_get(self, repo):
        repo.create_game("g_test", board_type="standard_6", llm_model="deepseek-v4-pro")
        game = repo.get_game("g_test")
        assert game is not None
        assert game["game_id"] == "g_test"
        assert game["board_type"] == "standard_6"
        assert game["status"] == "created"

    def test_set_status(self, repo):
        repo.create_game("g_s1")
        repo.set_running("g_s1")
        assert repo.get_game_status("g_s1") == "running"

    def test_finish_game(self, repo):
        repo.create_game("g_fin")
        repo.finish_game("g_fin", Winner.GOOD, "all_wolves_dead", 3)
        game = repo.get_game("g_fin")
        assert game["status"] == "finished"
        assert game["winner"] == "good"
        assert game["win_reason"] == "all_wolves_dead"
        assert game["total_rounds"] == 3
        assert game["finished_at"] is not None

    def test_list_games(self, repo):
        repo.create_game("g_a")
        repo.create_game("g_b")
        repo.create_game("g_c")
        games = repo.list_games(limit=2)
        assert len(games) == 2

    def test_get_nonexistent(self, repo):
        assert repo.get_game("no_such") is None
        assert repo.get_game_status("no_such") is None


# ============================================================
# PlayerRepository
# ============================================================

class TestPlayerRepository:
    @pytest.fixture
    def repo(self, db) -> PlayerRepository:
        return PlayerRepository(db)

    @pytest.fixture
    def _init_game(self, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_p', 'standard_6')")

    def test_insert_and_get(self, repo, _init_game):
        players = [
            PlayerState(player_id=0, name="P0"),
            PlayerState(player_id=1, name="P1"),
        ]
        repo.insert_players("g_p", players)
        rows = repo.get_players("g_p")
        assert len(rows) == 2

    def test_set_role(self, repo, _init_game):
        repo.insert_players("g_p", [PlayerState(player_id=0, name="P0")])
        repo.set_role("g_p", 0, Role.WEREWOLF)
        p = repo.get_player("g_p", 0)
        assert p["role"] == "werewolf"

    def test_set_alive(self, repo, _init_game):
        repo.insert_players("g_p", [PlayerState(player_id=0, name="P0")])
        repo.set_alive("g_p", 0, False)
        p = repo.get_player("g_p", 0)
        assert p["is_alive"] == 0

    def test_set_sheriff(self, repo, _init_game):
        repo.insert_players("g_p", [
            PlayerState(player_id=0, name="P0"),
            PlayerState(player_id=1, name="P1"),
        ])
        repo.set_role("g_p", 0, Role.SEER)
        repo.set_sheriff("g_p", 0)
        p0 = repo.get_player("g_p", 0)
        p1 = repo.get_player("g_p", 1)
        assert p0["is_sheriff"] == 1
        assert p1["is_sheriff"] == 0

    def test_get_alive_players(self, repo, _init_game):
        repo.insert_players("g_p", [
            PlayerState(player_id=0, name="P0"),
            PlayerState(player_id=1, name="P1"),
            PlayerState(player_id=2, name="P2"),
        ])
        repo.set_alive("g_p", 1, False)
        alive = repo.get_alive_players("g_p")
        assert len(alive) == 2


# ============================================================
# ActionLogRepository
# ============================================================

class TestActionLogRepository:
    @pytest.fixture
    def repo(self, db) -> ActionLogRepository:
        return ActionLogRepository(db)

    @pytest.fixture
    def _init_game(self, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_al', 'standard_6')")

    def test_log_action(self, repo, _init_game):
        decision = AgentDecision(action="kill", target=3, thought="我怀疑3号是预言家", content="")
        rid = repo.log_action("g_al", 1, "night_wolf", 0, decision)
        assert rid is not None

    def test_get_actions_by_round(self, repo, _init_game):
        repo.log_action("g_al", 1, "night_wolf", 0, AgentDecision(action="kill", target=2, thought="刀2"))
        repo.log_action("g_al", 1, "night_wolf", 1, AgentDecision(action="kill", target=2, thought="同意"))
        actions = repo.get_actions_by_round("g_al", 1)
        assert len(actions) == 2

    def test_get_actions_by_player(self, repo, _init_game):
        repo.log_action("g_al", 1, "night_wolf", 0, AgentDecision(action="kill", target=2, thought="刀"))
        repo.log_action("g_al", 2, "night_wolf", 0, AgentDecision(action="kill", target=3, thought="换目标"))
        actions = repo.get_actions_by_player("g_al", 0)
        assert len(actions) == 2

    def test_replay_log_ordered(self, repo, _init_game):
        repo.log_action("g_al", 1, "night_wolf", 0, AgentDecision(action="kill", target=2, thought="1"))
        repo.log_action("g_al", 1, "night_seer", 2, AgentDecision(action="verify", target=1, thought="2"))
        repo.log_action("g_al", 1, "vote", 0, AgentDecision(action="vote", target=2, thought="3"))
        replay = repo.get_replay_log("g_al")
        assert len(replay) == 3
        assert replay[0]["phase"] == "night_wolf"
        assert replay[1]["phase"] == "night_seer"
        assert replay[2]["phase"] == "vote"


# ============================================================
# MemoryRepository
# ============================================================

class TestMemoryRepository:
    @pytest.fixture
    def repo(self, db) -> MemoryRepository:
        return MemoryRepository(db)

    @pytest.fixture
    def _init_game(self, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_mem', 'standard_6')")

    def test_save_and_load(self, repo, _init_game):
        entry = MemoryEntry(
            round=1, phase="night_seer", event_type="verify_result",
            content={"target": 3, "is_wolf": True}, importance=5,
        )
        repo.save_memory("g_mem", 2, entry)
        loaded = repo.load_memories("g_mem", 2)
        assert len(loaded) == 1
        assert loaded[0].content["target"] == 3
        assert loaded[0].content["is_wolf"] is True

    def test_load_with_min_importance(self, repo, _init_game):
        repo.save_memory("g_mem", 0, MemoryEntry(round=1, phase="speech", event_type="heard_speech", content={"text": "x"}, importance=1))
        repo.save_memory("g_mem", 0, MemoryEntry(round=1, phase="night_seer", event_type="verify_result", content={"target": 4, "is_wolf": True}, importance=5))
        loaded = repo.load_memories("g_mem", 0, min_importance=3)
        assert len(loaded) == 1
        assert loaded[0].event_type == "verify_result"

    def test_batch_save(self, repo, _init_game):
        entries = [
            MemoryEntry(round=1, phase="night_wolf", event_type="teammate_info", content={"teammates": [2]}, importance=5),
            MemoryEntry(round=1, phase="night_wolf", event_type="team_chat", content={"msg": "刀3"}, importance=3),
        ]
        repo.save_memories("g_mem", 0, entries)
        loaded = repo.load_memories("g_mem", 0)
        assert len(loaded) == 2

    def test_load_important_only(self, repo, _init_game):
        repo.save_memory("g_mem", 1, MemoryEntry(round=1, phase="init", event_type="role", content={}, importance=1))
        repo.save_memory("g_mem", 1, MemoryEntry(round=1, phase="night_seer", event_type="verify_result", content={"target": 2, "is_wolf": False}, importance=5))
        important = repo.load_important_memories("g_mem", 1)
        assert len(important) == 1

    def test_load_empty_returns_empty(self, repo, _init_game):
        loaded = repo.load_memories("g_mem", 99)
        assert loaded == []


# ============================================================
# GameEventRepository
# ============================================================

class TestGameEventRepository:
    @pytest.fixture
    def repo(self, db) -> GameEventRepository:
        return GameEventRepository(db)

    @pytest.fixture
    def _init_game(self, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_ev', 'standard_6')")

    def test_log_and_get_events(self, repo, _init_game):
        repo.log_event("g_ev", 1, "speech", "player_speak", {"player_id": 0, "content": "我是预言家"})
        repo.log_event("g_ev", 1, "vote", "vote_result", {"tally": {"2": 3}})
        events = repo.get_events_by_round("g_ev", 1)
        assert len(events) == 2

    def test_full_timeline(self, repo, _init_game):
        repo.log_event("g_ev", 1, "night_wolf", "wolf_kill", {"target": 3})
        repo.log_event("g_ev", 1, "day_start", "death_announce", {})
        repo.log_event("g_ev", 2, "night_wolf", "wolf_kill", {"target": 1})
        timeline = repo.get_full_timeline("g_ev")
        assert len(timeline) == 3
        assert timeline[0]["round"] == 1
        assert timeline[-1]["round"] == 2


# ============================================================
# UnitOfWork
# ============================================================

class TestUnitOfWork:
    @pytest.fixture
    def _init_game(self, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_uow', 'standard_6')")

    def test_persist_game_start(self, uow):
        players = [
            PlayerState(player_id=0, name="狼A"),
            PlayerState(player_id=1, name="预言家"),
        ]
        uow.persist_game_start("g_start", "standard_6", "deepseek-v4-pro", players)
        game = uow.games.get_game("g_start")
        assert game is not None
        assert len(uow.players.get_players("g_start")) == 2

    def test_persist_roles(self, uow):
        uow.games.create_game("g_role", "standard_6")
        uow.players.insert_players("g_role", [
            PlayerState(player_id=0, name="P0"),
            PlayerState(player_id=1, name="P1"),
        ])
        uow.persist_roles("g_role", {0: Role.WEREWOLF, 1: Role.VILLAGER})
        p0 = uow.players.get_player("g_role", 0)
        assert p0["role"] == "werewolf"

    def test_persist_game_end(self, uow):
        uow.games.create_game("g_end")
        uow.persist_game_end("g_end", Winner.EVIL, "all_gods_dead", 4)
        game = uow.games.get_game("g_end")
        assert game["status"] == "finished"
        assert game["winner"] == "evil"

    def test_load_player_memories(self, uow, db):
        db.execute("INSERT INTO games (game_id, board_type) VALUES ('g_lm', 'standard_6')")
        uow.memories.save_memory("g_lm", 1, MemoryEntry(
            round=1, phase="night_seer", event_type="verify_result",
            content={"target": 3, "is_wolf": True}, importance=5,
        ))
        mems = uow.load_player_memories("g_lm", 1)
        assert len(mems) == 1
