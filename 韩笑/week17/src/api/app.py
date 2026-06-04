"""
FastAPI Web应用
提供可视化界面的后端API
"""

import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings, Settings
from src.sdk import VectorCacheClient


# 全局状态
class AppState:
    client: Optional[VectorCacheClient] = None
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    start_time: float = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    state.start_time = time.time()
    state.client = VectorCacheClient()
    yield
    if state.client:
        state.client.close()


app = FastAPI(
    title="向量检索与智能缓存服务平台",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# 请求模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_cache: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    category: Optional[str] = None


class IntentRequest(BaseModel):
    intent_name: str
    description: str
    examples: list


class ClassifyRequest(BaseModel):
    text: str
    top_k: int = 3
    threshold: float = 0.7


# API端点
@app.get("/")
async def index():
    """返回前端页面"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "向量检索与智能缓存服务平台 API"}


@app.get("/api/health")
async def health():
    """健康检查"""
    redis_ok = False
    try:
        redis_ok = state.client.redis_client.ping()
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "uptime": round(time.time() - state.start_time, 1)
    }


@app.get("/api/config")
async def get_config():
    """获取当前配置（脱敏）"""
    settings = get_settings()
    return {
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
            "max_tokens": settings.llm.max_tokens,
            "temperature": settings.llm.temperature
        },
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
            "dimensions": settings.embedding.dimensions,
            "base_url": settings.embedding.base_url
        },
        "redis": {
            "host": settings.redis.host,
            "port": settings.redis.port,
            "db": settings.redis.db
        },
        "cache": {
            "semantic_threshold": settings.cache.semantic.similarity_threshold,
            "semantic_ttl": settings.cache.semantic.default_ttl,
            "embedding_ttl": settings.cache.embedding.default_ttl
        }
    }


@app.get("/api/stats")
async def get_stats():
    """获取缓存统计信息"""
    semantic_stats = state.client.semantic_cache.get_stats()
    embedding_stats = state.client.embedding_cache.get_stats()
    conversation_stats = state.client.conversation_history.get_stats()
    intent_stats = state.client.intent_classifier.get_stats()

    # 获取向量索引信息
    vector_info = {}
    try:
        vector_info = state.client.vector_index.info()
    except Exception:
        vector_info = {"num_docs": 0, "error": "索引未创建"}

    # 计算命中率
    total = state.cache_hits + state.cache_misses
    hit_rate = (state.cache_hits / total * 100) if total > 0 else 0

    return {
        "semantic_cache": {
            "total_entries": semantic_stats.get("total_entries", 0),
            "similarity_threshold": semantic_stats.get("similarity_threshold", 0.95)
        },
        "embedding_cache": {
            "total_entries": embedding_stats.get("total_entries", 0),
            "default_ttl": embedding_stats.get("default_ttl", 86400)
        },
        "vector_index": {
            "num_docs": vector_info.get("num_docs", 0),
            "index_name": vector_info.get("index_name", "vector_index")
        },
        "conversation": {
            "total_sessions": conversation_stats.get("total_sessions", 0),
            "total_messages": conversation_stats.get("total_messages", 0)
        },
        "intent": {
            "total_intents": intent_stats.get("total_intents", 0),
            "total_examples": intent_stats.get("total_examples", 0)
        },
        "performance": {
            "total_queries": state.total_queries,
            "cache_hits": state.cache_hits,
            "cache_misses": state.cache_misses,
            "hit_rate": round(hit_rate, 1)
        }
    }


@app.get("/api/features")
async def get_features():
    """获取项目功能列表"""
    return {
        "features": [
            {
                "id": 1,
                "name": "LLM对话",
                "description": "支持DeepSeek等多种大模型对话",
                "icon": "chat",
                "status": "active"
            },
            {
                "id": 2,
                "name": "语义缓存",
                "description": "基于语义相似度缓存LLM响应，减少重复调用",
                "icon": "cache",
                "status": "active"
            },
            {
                "id": 3,
                "name": "嵌入缓存",
                "description": "缓存文本向量化结果，避免重复计算",
                "icon": "memory",
                "status": "active"
            },
            {
                "id": 4,
                "name": "向量索引",
                "description": "基于Redis Stack的高性能向量索引",
                "icon": "index",
                "status": "active"
            },
            {
                "id": 5,
                "name": "向量搜索",
                "description": "KNN近邻搜索，支持元数据过滤",
                "icon": "search",
                "status": "active"
            },
            {
                "id": 6,
                "name": "混合查询",
                "description": "文本+向量混合搜索，精准匹配",
                "icon": "filter",
                "status": "active"
            },
            {
                "id": 7,
                "name": "多Provider支持",
                "description": "LLM和Embedding可使用不同提供商",
                "icon": "provider",
                "status": "active"
            },
            {
                "id": 8,
                "name": "Python SDK",
                "description": "统一的客户端SDK，简化集成",
                "icon": "sdk",
                "status": "active"
            },
            {
                "id": 9,
                "name": "对话历史管理",
                "description": "多轮对话上下文管理，支持会话历史",
                "icon": "history",
                "status": "active"
            },
            {
                "id": 10,
                "name": "意图识别",
                "description": "基于向量相似度的意图分类和路由",
                "icon": "intent",
                "status": "active"
            }
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """LLM对话"""
    try:
        state.total_queries += 1

        # 使用带历史的对话
        if request.session_id:
            result = state.client.chat_with_history(
                query=request.message,
                session_id=request.session_id,
                use_cache=request.use_cache
            )
        else:
            result = state.client.chat(
                query=request.message,
                use_cache=request.use_cache
            )

        from_cache = result.get("from_cache", False)
        if from_cache:
            state.cache_hits += 1
        else:
            state.cache_misses += 1

        response_data = result.get("response", {})
        content = ""
        if isinstance(response_data, dict):
            content = response_data.get("content", str(response_data))
        else:
            content = str(response_data)

        return {
            "content": content,
            "from_cache": from_cache,
            "session_id": result.get("session_id"),
            "model": state.client.llm_adapter.model_name,
            "provider": state.client.llm_adapter.provider_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vector/search")
async def vector_search(request: SearchRequest):
    """向量搜索"""
    try:
        filters = {}
        if request.category:
            filters["category"] = request.category

        results = state.client.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters if filters else None
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    try:
        state.client.semantic_cache.clear()
        state.client.embedding_cache.clear()
        state.cache_hits = 0
        state.cache_misses = 0
        state.total_queries = 0
        return {"message": "缓存已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/{session_id}")
async def get_conversation(session_id: str, limit: int = 50):
    """获取对话历史"""
    try:
        history = state.client.get_conversation_history(session_id, limit=limit)
        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """清除对话历史"""
    try:
        state.client.clear_conversation(session_id)
        return {"message": f"会话 {session_id} 已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
async def list_conversations():
    """列出所有会话"""
    try:
        sessions = state.client.conversation_history.get_sessions()
        stats = state.client.conversation_history.get_stats()
        return {
            "sessions": sessions,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intent/register")
async def register_intent(request: IntentRequest):
    """注册意图"""
    try:
        state.client.register_intent(
            intent_name=request.intent_name,
            description=request.description,
            examples=request.examples
        )
        return {"message": f"意图 '{request.intent_name}' 注册成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intent/classify")
async def classify_intent(request: ClassifyRequest):
    """意图分类"""
    try:
        results = state.client.classify_intent(
            text=request.text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return {
            "text": request.text,
            "intents": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intents")
async def list_intents():
    """列出所有意图"""
    try:
        stats = state.client.intent_classifier.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intent/stats")
async def get_intent_stats():
    """获取意图分类统计"""
    try:
        stats = state.client.intent_classifier.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app():
    """创建应用实例"""
    return app
