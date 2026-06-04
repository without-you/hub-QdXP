"""
基础使用示例
演示如何使用向量检索与智能缓存服务平台
"""

from src.sdk import VectorCacheClient


def main():
    """主函数"""

    # 1. 初始化客户端
    print("=== 1. 初始化客户端 ===")
    client = VectorCacheClient(
        redis_url="redis://localhost:6379",
        index_name="example_index",
        embedding_model="text-embedding-ada-002"
    )
    print("客户端初始化成功")

    # 2. 创建索引
    print("\n=== 2. 创建索引 ===")
    success = client.create_index(
        dimensions=1536,
        distance_metric="COSINE"
    )
    print(f"索引创建: {'成功' if success else '失败'}")

    # 3. 添加文本
    print("\n=== 3. 添加文本 ===")
    texts = [
        {"text": "中国的首都是北京", "category": "geography", "source": "example"},
        {"text": "美国的首都是华盛顿", "category": "geography", "source": "example"},
        {"text": "日本的首都是东京", "category": "geography", "source": "example"},
        {"text": "今天天气真好", "category": "weather", "source": "example"},
        {"text": "我喜欢编程", "category": "hobby", "source": "example"},
    ]

    for item in texts:
        doc_id = client.add_text(
            text=item["text"],
            metadata={
                "category": item["category"],
                "source": item["source"]
            }
        )
        print(f"添加文档: {item['text'][:20]}... -> ID: {doc_id}")

    # 4. 语义搜索
    print("\n=== 4. 语义搜索 ===")
    query = "北京是中国的什么？"
    results = client.search(query, top_k=3)

    print(f"查询: {query}")
    print(f"找到 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('text', 'N/A')} (分数: {result.get('score', 0):.4f})")

    # 5. 使用语义缓存
    print("\n=== 5. 使用语义缓存 ===")

    # 模拟LLM响应
    def mock_llm_response(query):
        """模拟LLM响应"""
        if "情感" in query:
            return {"sentiment": "positive", "confidence": 0.95}
        elif "首都" in query:
            return {"answer": "北京", "confidence": 0.99}
        else:
            return {"answer": "我不确定", "confidence": 0.5}

    # 第一次查询 - 缓存未命中
    query1 = "请判断用户的情感：我今天很开心"
    cached_response = client.get_cached_response(query1)

    if cached_response is None:
        print(f"缓存未命中，调用LLM...")
        response = mock_llm_response(query1)
        client.set_cached_response(query1, response, ttl=3600)
        print(f"响应已缓存: {response}")
    else:
        print(f"缓存命中: {cached_response}")

    # 第二次查询 - 缓存命中
    cached_response = client.get_cached_response(query1)
    if cached_response:
        print(f"第二次查询 - 缓存命中: {cached_response}")

    # 6. 按类别搜索
    print("\n=== 6. 按类别搜索 ===")
    query_embedding = [0.1] * 1536  # 占位向量
    results = client.vector_search.search_by_category(
        query_embedding,
        category="geography",
        top_k=2
    )

    print(f"类别 'geography' 的搜索结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. ID: {result.get('id')}, 分数: {result.get('score', 0):.4f}")

    # 7. 缓存统计
    print("\n=== 7. 缓存统计 ===")
    embedding_stats = client.embedding_cache.get_stats()
    semantic_stats = client.semantic_cache.get_stats()

    print(f"嵌入缓存: {embedding_stats}")
    print(f"语义缓存: {semantic_stats}")

    # 8. 关闭连接
    print("\n=== 8. 关闭连接 ===")
    client.close()
    print("连接已关闭")


if __name__ == "__main__":
    main()