"""FastAPI 入口 — 多智能体狼人杀 Team 系统"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.connection import DatabaseManager

# ------------------------------------------------------------
# 日志
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("werewolf")

# ------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期"""
    # 启动：初始化数据库
    db_path = app.state.db_path if hasattr(app.state, "db_path") else "werewolf.db"
    db = DatabaseManager.get_instance(db_path)
    logger.info("数据库已连接: %s", db_path)
    app.state.db = db

    yield

    # 关闭
    db.close()
    logger.info("服务器关闭")


# ------------------------------------------------------------
# App 工厂
# ------------------------------------------------------------


def create_app(db_path: str = "werewolf.db") -> FastAPI:
    app = FastAPI(
        title="多智能体狼人杀 Team 系统",
        description="基于 LLM 的多智能体狼人杀博弈平台",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.db_path = db_path

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from app.routes import router as game_router
    app.include_router(game_router, prefix="/api/v1")

    # 注册 WebSocket
    from app.ws.connection import router as ws_router
    app.include_router(ws_router)

    return app


# ------------------------------------------------------------
# 直接运行
# ------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
