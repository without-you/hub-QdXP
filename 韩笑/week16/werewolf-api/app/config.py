"""全局配置加载 — 从项目根目录 config.yaml 读取"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_config: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载配置文件（全局单例）"""
    global _config
    if _config is not None:
        return _config

    target = Path(path) if path else _CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"配置文件不存在: {target}")

    with open(target, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
    return _config


def get_llm_config() -> dict[str, Any]:
    """获取 LLM 配置"""
    cfg = load_config()
    return cfg.get("llm", {})


def get_server_config() -> dict[str, Any]:
    """获取服务器配置"""
    cfg = load_config()
    return cfg.get("server", {})


def get_db_config() -> dict[str, Any]:
    """获取数据库配置"""
    cfg = load_config()
    return cfg.get("database", {})


def get_game_config() -> dict[str, Any]:
    """获取游戏默认配置"""
    cfg = load_config()
    return cfg.get("game", {})


def reload_config() -> dict[str, Any]:
    """强制重新加载配置"""
    global _config
    _config = None
    return load_config()
