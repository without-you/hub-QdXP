"""
意图识别模块
基于向量相似度的意图分类和路由
"""

from typing import List, Dict, Any, Optional, Callable
import json
import numpy as np
import redis


class IntentClassifier:
    """
    意图分类器

    通过计算用户输入与预定义意图的向量相似度来进行意图分类
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "intent",
        embedding_func: Optional[Callable] = None,
        config=None
    ):
        """
        初始化意图分类器

        Args:
            redis_client: Redis客户端
            prefix: 缓存键前缀
            embedding_func: 嵌入向量生成函数
            config: 配置对象（可选）
        """
        self.redis = redis_client
        self.prefix = prefix
        self.embedding_func = embedding_func
        self.config = config

    def _make_key(self, intent_name: str) -> str:
        """生成意图的Redis键"""
        return f"{self.prefix}:{intent_name}"

    def _make_example_key(self, intent_name: str, example_idx: int) -> str:
        """生成意图示例的Redis键"""
        return f"{self.prefix}:example:{intent_name}:{example_idx}"

    def register_intent(
        self,
        intent_name: str,
        description: str,
        examples: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册一个意图

        Args:
            intent_name: 意图名称
            description: 意图描述
            examples: 示例文本列表
            metadata: 额外元数据

        Returns:
            bool: 是否注册成功
        """
        if not self.embedding_func:
            raise ValueError("embedding_func is required for intent registration")

        # 存储意图元数据
        intent_data = {
            "name": intent_name,
            "description": description,
            "examples": examples,
            "metadata": metadata or {}
        }
        intent_key = self._make_key(intent_name)
        self.redis.set(intent_key, json.dumps(intent_data, ensure_ascii=False))

        # 为每个示例生成并存储嵌入向量
        for idx, example in enumerate(examples):
            try:
                embedding = self.embedding_func(example)
                example_key = self._make_example_key(intent_name, idx)
                self.redis.set(
                    example_key,
                    json.dumps({
                        "text": example,
                        "embedding": embedding,
                        "intent": intent_name
                    }, ensure_ascii=False)
                )
            except Exception as e:
                print(f"Failed to process example '{example}': {e}")
                continue

        return True

    def classify(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        对文本进行意图分类

        Args:
            text: 输入文本
            top_k: 返回的最大意图数
            threshold: 最低相似度阈值

        Returns:
            List[Dict]: 分类结果列表（按相似度降序）
        """
        if not self.embedding_func:
            raise ValueError("embedding_func is required for intent classification")

        # 获取输入文本的嵌入向量
        query_embedding = np.array(self.embedding_func(text), dtype=np.float32)

        # 获取所有意图示例的嵌入向量
        pattern = f"{self.prefix}:example:*"
        example_keys = self.redis.keys(pattern)

        if not example_keys:
            return []

        # 计算与每个示例的相似度
        similarities = []

        for key in example_keys:
            stored_data = self.redis.get(key)
            if not stored_data:
                continue

            try:
                data = json.loads(stored_data)
                stored_embedding = np.array(data["embedding"], dtype=np.float32)

                # 计算余弦相似度
                dot_product = np.dot(query_embedding, stored_embedding)
                norm1 = np.linalg.norm(query_embedding)
                norm2 = np.linalg.norm(stored_embedding)

                if norm1 == 0 or norm2 == 0:
                    continue

                similarity = float(dot_product / (norm1 * norm2))

                similarities.append({
                    "intent": data["intent"],
                    "example": data["text"],
                    "similarity": similarity
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        # 按相似度降序排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # 聚合每个意图的最高相似度
        intent_scores = {}
        for item in similarities:
            intent = item["intent"]
            if intent not in intent_scores:
                intent_scores[intent] = {
                    "intent": intent,
                    "max_similarity": item["similarity"],
                    "best_example": item["example"],
                    "all_similarities": []
                }
            intent_scores[intent]["all_similarities"].append(item["similarity"])

        # 计算每个意图的平均相似度
        results = []
        for intent, data in intent_scores.items():
            avg_similarity = sum(data["all_similarities"]) / len(data["all_similarities"])

            if data["max_similarity"] >= threshold:
                # 获取意图元数据
                intent_key = self._make_key(intent)
                intent_info = self.redis.get(intent_key)
                intent_metadata = {}
                if intent_info:
                    try:
                        intent_metadata = json.loads(intent_info)
                    except json.JSONDecodeError:
                        pass

                results.append({
                    "intent": intent,
                    "similarity": data["max_similarity"],
                    "avg_similarity": avg_similarity,
                    "best_example": data["best_example"],
                    "description": intent_metadata.get("description", ""),
                    "metadata": intent_metadata.get("metadata", {})
                })

        # 按相似度降序排序，返回top_k个
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_intent(self, intent_name: str) -> Optional[Dict[str, Any]]:
        """
        获取意图信息

        Args:
            intent_name: 意图名称

        Returns:
            Optional[Dict]: 意图信息
        """
        intent_key = self._make_key(intent_name)
        intent_info = self.redis.get(intent_key)
        if intent_info:
            try:
                return json.loads(intent_info)
            except json.JSONDecodeError:
                return None
        return None

    def list_intents(self) -> List[str]:
        """
        获取所有已注册的意图名称

        Returns:
            List[str]: 意图名称列表
        """
        pattern = f"{self.prefix}:*"
        keys = self.redis.keys(pattern)

        intents = set()
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            # 只处理意图元数据键，跳过示例键
            if not key_str.startswith(f"{self.prefix}:example:"):
                intent_name = key_str.replace(f"{self.prefix}:", "")
                if intent_name:
                    intents.add(intent_name)

        return list(intents)

    def delete_intent(self, intent_name: str) -> bool:
        """
        删除意图

        Args:
            intent_name: 意图名称

        Returns:
            bool: 是否删除成功
        """
        # 删除意图元数据
        intent_key = self._make_key(intent_name)
        self.redis.delete(intent_key)

        # 删除所有示例
        pattern = f"{self.prefix}:example:{intent_name}:*"
        example_keys = self.redis.keys(pattern)
        if example_keys:
            self.redis.delete(*example_keys)

        return True

    def clear(self) -> int:
        """
        清除所有意图

        Returns:
            int: 清除的意图数
        """
        intents = self.list_intents()
        for intent in intents:
            self.delete_intent(intent)
        return len(intents)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取意图分类器统计

        Returns:
            Dict: 统计信息
        """
        intents = self.list_intents()
        total_examples = 0

        for intent in intents:
            pattern = f"{self.prefix}:example:{intent}:*"
            example_keys = self.redis.keys(pattern)
            total_examples += len(example_keys)

        return {
            "total_intents": len(intents),
            "total_examples": total_examples,
            "intents": intents
        }
