"""
对话历史管理模块
存储和检索对话历史记录，支持多轮对话上下文
"""

from typing import List, Dict, Any, Optional
import json
import time
import redis


class ConversationHistory:
    """
    对话历史管理

    存储用户与LLM的对话记录，支持按会话检索、清除和统计
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "conversation",
        max_history: int = 100,
        ttl: int = 86400 * 7,
        config=None
    ):
        """
        初始化对话历史管理

        Args:
            redis_client: Redis客户端
            prefix: 缓存键前缀
            max_history: 每个会话最大历史条数
            ttl: 过期时间（秒，默认7天）
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.prefix = prefix
        self.max_history = max_history
        self.default_ttl = ttl
        self.config = config

    def _make_key(self, session_id: str) -> str:
        """生成会话的Redis键"""
        return f"{self.prefix}:{session_id}"

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加一条消息到对话历史

        Args:
            session_id: 会话ID
            role: 角色（user/assistant/system）
            content: 消息内容
            metadata: 额外元数据

        Returns:
            bool: 是否添加成功
        """
        key = self._make_key(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # 使用LPUSH添加到列表头部，LTRIM限制长度
        pipe = self.redis.pipeline()
        pipe.lpush(key, json.dumps(message, ensure_ascii=False))
        pipe.ltrim(key, 0, self.max_history - 1)
        pipe.expire(key, self.default_ttl)
        pipe.execute()

        return True

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            session_id: 会话ID
            limit: 返回的最大条数（None表示全部）

        Returns:
            List[Dict]: 消息列表（按时间正序）
        """
        key = self._make_key(session_id)

        if limit:
            raw_messages = self.redis.lrange(key, 0, limit - 1)
        else:
            raw_messages = self.redis.lrange(key, 0, -1)

        messages = []
        for raw in reversed(raw_messages):
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

        return messages

    def get_context_messages(
        self,
        session_id: str,
        max_turns: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取用于LLM上下文的消息格式

        Args:
            session_id: 会话ID
            max_turns: 最大轮数

        Returns:
            List[Dict]: 格式化的消息列表（role + content）
        """
        history = self.get_history(session_id, limit=max_turns * 2)

        messages = []
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        return messages

    def clear(self, session_id: str) -> bool:
        """
        清除指定会话的历史

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否清除成功
        """
        key = self._make_key(session_id)
        return bool(self.redis.delete(key))

    def clear_all(self) -> int:
        """
        清除所有对话历史

        Returns:
            int: 清除的会话数
        """
        keys = self.redis.keys(f"{self.prefix}:*")
        if keys:
            return self.redis.delete(*keys)
        return 0

    def get_sessions(self) -> List[str]:
        """
        获取所有会话ID

        Returns:
            List[str]: 会话ID列表
        """
        keys = self.redis.keys(f"{self.prefix}:*")
        sessions = []
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            session_id = key_str.replace(f"{self.prefix}:", "")
            if session_id:
                sessions.append(session_id)
        return sessions

    def get_stats(self) -> Dict[str, Any]:
        """
        获取对话历史统计

        Returns:
            Dict: 统计信息
        """
        sessions = self.get_sessions()
        total_messages = 0

        for session_id in sessions:
            key = self._make_key(session_id)
            total_messages += self.redis.llen(key)

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "max_history": self.max_history,
            "default_ttl": self.default_ttl
        }

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取指定会话的信息

        Args:
            session_id: 会话ID

        Returns:
            Dict: 会话信息
        """
        key = self._make_key(session_id)
        message_count = self.redis.llen(key)
        ttl = self.redis.ttl(key)

        history = self.get_history(session_id, limit=1)
        last_message = history[0] if history else None

        return {
            "session_id": session_id,
            "message_count": message_count,
            "ttl": ttl,
            "last_message": last_message
        }
