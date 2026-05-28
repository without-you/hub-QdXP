-- ============================================================
-- 狼人杀 Team 系统 — SQLite 数据库 Schema
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------
-- 1. 对局表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_id      TEXT PRIMARY KEY,
    board_type   TEXT    NOT NULL DEFAULT 'standard_6',
    status       TEXT    NOT NULL DEFAULT 'created',   -- created | running | finished
    winner       TEXT,                                  -- good | evil | NULL
    win_reason   TEXT,                                  -- all_wolves_dead | all_gods_dead | all_villagers_dead
    total_rounds INTEGER NOT NULL DEFAULT 0,
    llm_model    TEXT    NOT NULL DEFAULT 'deepseek-v4-pro',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT
);

-- -----------------------------------------------------------
-- 2. 玩家表
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT    NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id  INTEGER NOT NULL,   -- 座位号 0-based
    name       TEXT    NOT NULL,
    role       TEXT    NOT NULL,   -- werewolf | seer | witch | villager
    is_alive   INTEGER NOT NULL DEFAULT 1,
    is_sheriff INTEGER NOT NULL DEFAULT 0,
    UNIQUE(game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_players_game ON players(game_id);

-- -----------------------------------------------------------
-- 3. 行动日志表（Agent CoT + 决策 + 发言）
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS action_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT    NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    round      INTEGER NOT NULL,
    phase      TEXT    NOT NULL,   -- night_wolf | night_seer | night_witch | speech | vote | ...
    player_id  INTEGER NOT NULL,
    action     TEXT    NOT NULL,   -- kill | verify | save | poison | vote | speak | skip | self_destruct
    target     INTEGER,            -- 目标玩家编号，NULL = 无目标
    thought    TEXT    DEFAULT '',  -- CoT 内心独白
    content    TEXT    DEFAULT '',  -- 发言内容（仅 speak action 有值）
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_action_logs_game_round ON action_logs(game_id, round);
CREATE INDEX IF NOT EXISTS idx_action_logs_game_player ON action_logs(game_id, player_id);

-- -----------------------------------------------------------
-- 4. Agent 记忆表（持久化角色私有记忆）
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT    NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id  INTEGER NOT NULL,
    round      INTEGER NOT NULL,
    phase      TEXT    NOT NULL,
    event_type TEXT    NOT NULL,   -- verify_result | teammate_info | wolf_kill_info | suspicion | ...
    content    TEXT    NOT NULL DEFAULT '{}',  -- JSON
    importance INTEGER NOT NULL DEFAULT 1      -- 1-5, 越高越重要
);

CREATE INDEX IF NOT EXISTS idx_memories_game_player ON agent_memories(game_id, player_id);
CREATE INDEX IF NOT EXISTS idx_memories_importance  ON agent_memories(game_id, player_id, importance);

-- -----------------------------------------------------------
-- 5. 游戏事件表（公开日志，用于回放）
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT    NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    round      INTEGER NOT NULL,
    phase      TEXT    NOT NULL,
    event_type TEXT    NOT NULL,   -- player_speak | player_death | vote_result | sheriff_elected | phase_change
    content    TEXT    NOT NULL DEFAULT '{}',  -- JSON
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_game_round ON game_events(game_id, round);
