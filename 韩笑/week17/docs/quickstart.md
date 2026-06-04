# 快速开始指南

## 环境准备

### 1. Python环境

确保Python 3.9+已安装：

```bash
python --version
# Python 3.9.0 或更高版本
```

### 2. Redis环境

安装Redis并启用RediSearch模块：

```bash
# macOS (使用Homebrew)
brew install redis
brew install redis-stack-server

# Ubuntu/Debian
sudo apt-get install redis-server
sudo apt-get install redis-stack-server

# Docker
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

验证Redis运行：

```bash
redis-cli ping
# 应该返回 PONG
```

### 3. 项目依赖

安装Python依赖：

```bash
cd week17
pip install -r requirements.txt
```

## 基础使用

### 1. 初始化客户端

```python
from src.sdk import VectorCacheClient

# 创建客户端
client = VectorCacheClient(
    redis_url="redis://localhost:6379",
    index_name="my_first_index"
)

print("客户端初始化成功！")
```

### 2. 创建向量索引

```python
# 创建索引
success = client.create_index(
    dimensions=1536,  # OpenAI ada-002的维度
    distance_metric="COSINE"
)

if success:
    print("索引创建成功！")
else:
    print("索引创建失败！")
```

### 3. 添加文本数据

```python
# 添加一些示例文本
texts = [
    {
        "text": "中国的首都是北京",
        "category": "geography",
        "source": "knowledge_base"
    },
    {
        "text": "美国的首都是华盛顿",
        "category": "geography",
        "source": "knowledge_base"
    },
    {
        "text": "日本的首都是东京",
        "category": "geography",
        "source": "knowledge_base"
    },
    {
        "text": "今天天气真好",
        "category": "weather",
        "source": "conversation"
    },
    {
        "text": "我喜欢编程",
        "category": "hobby",
        "source": "conversation"
    }
]

# 添加到索引
for item in texts:
    doc_id = client.add_text(
        text=item["text"],
        metadata={
            "category": item["category"],
            "source": item["source"]
        }
    )
    print(f"添加: {item['text'][:20]}... -> ID: {doc_id}")
```

**注意**：由于我们还没有集成实际的嵌入模型，`add_text`方法会抛出`NotImplementedError`。在实际使用中，你需要集成一个嵌入模型。

### 4. 使用语义缓存

```python
# 模拟LLM响应
def mock_llm_response(query):
    """模拟大语言模型的响应"""
    if "情感" in query or "心情" in query:
        return {
            "sentiment": "positive",
            "confidence": 0.95,
            "explanation": "用户表达了积极的情感"
        }
    elif "首都" in query:
        return {
            "answer": "北京",
            "confidence": 0.99,
            "source": "knowledge_base"
        }
    else:
        return {
            "answer": "我不确定",
            "confidence": 0.5
        }

# 查询1：缓存未命中
query = "请判断用户的情感：我今天很开心"
print(f"\n查询: {query}")

# 检查缓存
cached_response = client.get_cached_response(query)

if cached_response is None:
    print("缓存未命中，调用LLM...")
    response = mock_llm_response(query)
    print(f"LLM响应: {response}")

    # 缓存响应
    client.set_cached_response(query, response, ttl=3600)
    print("响应已缓存")
else:
    print(f"缓存命中: {cached_response}")

# 查询2：缓存命中
print(f"\n再次查询: {query}")
cached_response = client.get_cached_response(query)
if cached_response:
    print(f"缓存命中: {cached_response}")
```

### 5. 使用嵌入缓存

```python
# 模拟嵌入向量
def mock_embedding(text):
    """模拟文本嵌入"""
    import hashlib
    import numpy as np

    # 使用文本哈希生成伪随机向量
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    return np.random.randn(1536).tolist()

# 缓存嵌入向量
text = "中国的首都是北京"
embedding = mock_embedding(text)

# 检查缓存
cached_embedding = client.embedding_cache.get(text)

if cached_embedding is None:
    print("嵌入缓存未命中，计算嵌入向量...")
    client.embedding_cache.set(text, embedding, ttl=86400)
    print("嵌入向量已缓存")
else:
    print("嵌入缓存命中")

# 批量操作
texts = ["文本1", "文本2", "文本3"]
embeddings = [mock_embedding(t) for t in texts]

# 批量缓存
client.embedding_cache.batch_set(texts, embeddings)
print("批量缓存完成")

# 批量获取
cached_embeddings = client.embedding_cache.batch_get(texts)
print(f"批量获取结果数量: {len(cached_embeddings)}")
```

### 6. 搜索功能

```python
# 注意：搜索功能需要实际的嵌入模型支持
# 这里展示的是接口使用方式

# 模拟搜索
def mock_search(query, top_k=5):
    """模拟搜索"""
    # 返回模拟结果
    return [
        {"id": "1", "text": "中国的首都是北京", "score": 0.95},
        {"id": "2", "text": "美国的首都是华盛顿", "score": 0.85},
        {"id": "3", "text": "日本的首都是东京", "score": 0.80}
    ][:top_k]

# 搜索示例
query = "北京是中国的什么？"
results = mock_search(query, top_k=3)

print(f"\n搜索查询: {query}")
print(f"找到 {len(results)} 个结果:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text']} (分数: {result['score']:.2f})")
```

### 7. 关闭连接

```python
# 关闭客户端连接
client.close()
print("\n连接已关闭")
```

## 完整示例

```python
"""
完整使用示例
"""

from src.sdk import VectorCacheClient

def main():
    # 1. 初始化
    print("1. 初始化客户端...")
    client = VectorCacheClient(
        redis_url="redis://localhost:6379",
        index_name="example_index"
    )

    # 2. 创建索引
    print("\n2. 创建索引...")
    client.create_index(dimensions=1536, distance_metric="COSINE")

    # 3. 使用语义缓存
    print("\n3. 使用语义缓存...")

    # 模拟LLM调用
    def call_llm(query):
        return {"answer": f"这是对'{query}'的回答"}

    # 第一次查询
    query = "什么是机器学习？"
    cached = client.get_cached_response(query)
    if not cached:
        response = call_llm(query)
        client.set_cached_response(query, response, ttl=3600)
        print(f"缓存未命中，已缓存响应")
    else:
        print(f"缓存命中")

    # 第二次查询
    cached = client.get_cached_response(query)
    if cached:
        print(f"第二次查询命中缓存: {cached}")

    # 4. 使用嵌入缓存
    print("\n4. 使用嵌入缓存...")
    text = "测试文本"
    embedding = [0.1] * 1536

    cached_embedding = client.embedding_cache.get(text)
    if not cached_embedding:
        client.embedding_cache.set(text, embedding)
        print("嵌入向量已缓存")
    else:
        print("嵌入缓存命中")

    # 5. 获取统计信息
    print("\n5. 缓存统计...")
    semantic_stats = client.semantic_cache.get_stats()
    embedding_stats = client.embedding_cache.get_stats()

    print(f"语义缓存: {semantic_stats}")
    print(f"嵌入缓存: {embedding_stats}")

    # 6. 关闭
    print("\n6. 关闭连接...")
    client.close()
    print("完成！")

if __name__ == "__main__":
    main()
```

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_sdk.py

# 运行并显示详细输出
pytest -v tests/

# 生成覆盖率报告
pytest --cov=src tests/
```

## 常见问题

### Q1: Redis连接失败

**错误**：`redis.exceptions.ConnectionError`

**解决方案**：
1. 确保Redis服务正在运行
2. 检查Redis URL是否正确
3. 检查网络连接

```bash
# 检查Redis状态
redis-cli ping

# 检查Redis端口
netstat -tlnp | grep 6379
```

### Q2: RediSearch模块未加载

**错误**：`Unknown command 'FT.CREATE'`

**解决方案**：
1. 安装RediSearch模块
2. 使用redis-stack镜像

```bash
# Docker运行带RediSearch的Redis
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### Q3: 嵌入模型未实现

**错误**：`NotImplementedError: 需要集成嵌入模型`

**解决方案**：
1. 集成实际的嵌入模型
2. 实现`_get_embedding`方法

```python
# 示例：集成OpenAI
import openai

def _get_embedding(self, text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']
```

### Q4: 内存不足

**错误**：`Redis OOM command not allowed`

**解决方案**：
1. 增加Redis内存限制
2. 配置内存淘汰策略
3. 使用Redis集群

```bash
# 修改Redis配置
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## 下一步

1. **集成嵌入模型**：实现实际的文本嵌入功能
2. **实现语义匹配**：基于向量相似度的缓存命中
3. **添加更多示例**：创建更复杂的使用场景
4. **性能优化**：优化大规模数据下的性能
5. **监控告警**：添加缓存和搜索的监控指标

## 获取帮助

- 查看API文档：`docs/api.md`
- 查看架构设计：`docs/architecture.md`
- 运行示例代码：`examples/basic_usage.py`
- 查看测试用例：`tests/`