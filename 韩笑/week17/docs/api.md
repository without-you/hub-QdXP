# API 文档

## 概述

向量检索与智能缓存服务平台提供以下核心API：

1. **VectorCacheClient** - 主客户端类
2. **SemanticCache** - 语义缓存
3. **EmbeddingCache** - 嵌入缓存
4. **VectorIndex** - 向量索引管理
5. **VectorSearch** - 向量搜索

---

## VectorCacheClient

主客户端类，提供统一的接口来管理向量索引、执行语义搜索和使用缓存功能。

### 初始化

```python
from src.sdk import VectorCacheClient

client = VectorCacheClient(
    redis_url="redis://localhost:6379",
    index_name="my_index",
    embedding_model="text-embedding-ada-002"
)
```

**参数：**
- `redis_url` (str): Redis连接URL
- `index_name` (str): 索引名称
- `embedding_model` (str): 使用的嵌入模型

### 方法

#### create_index

创建向量索引。

```python
success = client.create_index(
    dimensions=1536,
    distance_metric="COSINE"
)
```

**参数：**
- `dimensions` (int): 向量维度，默认1536
- `distance_metric` (str): 距离度量方式，可选 "COSINE", "L2", "IP"

**返回：** bool - 是否创建成功

#### add_text

添加文本到索引。

```python
doc_id = client.add_text(
    text="中国的首都是北京",
    metadata={"category": "geography"},
    id="optional_custom_id"
)
```

**参数：**
- `text` (str): 文本内容
- `metadata` (dict, optional): 元数据
- `id` (str, optional): 文档ID

**返回：** str - 文档ID

#### search

执行语义搜索。

```python
results = client.search(
    query="北京是中国的什么？",
    top_k=10,
    filters={"category": "geography"}
)
```

**参数：**
- `query` (str): 查询文本
- `top_k` (int): 返回结果数量，默认10
- `filters` (dict, optional): 过滤条件

**返回：** List[Dict] - 搜索结果列表

#### get_cached_response

获取缓存的响应。

```python
response = client.get_cached_response("查询文本")
```

**参数：**
- `query` (str): 查询文本

**返回：** Optional[Dict] - 缓存的响应

#### set_cached_response

设置缓存的响应。

```python
success = client.set_cached_response(
    query="查询文本",
    response={"answer": "答案"},
    ttl=3600
)
```

**参数：**
- `query` (str): 查询文本
- `response` (dict): 响应数据
- `ttl` (int, optional): 过期时间（秒）

**返回：** bool - 是否设置成功

#### close

关闭连接。

```python
client.close()
```

---

## SemanticCache

语义缓存实现，通过计算查询的语义相似度来命中缓存。

### 初始化

```python
from src.cache import SemanticCache

cache = SemanticCache(
    redis_client=redis_client,
    prefix="semantic_cache",
    similarity_threshold=0.95
)
```

**参数：**
- `redis_client`: Redis客户端
- `prefix` (str): 缓存键前缀
- `similarity_threshold` (float): 相似度阈值

### 方法

#### get

获取缓存的响应。

```python
response = cache.get("查询文本")
```

**返回：** Optional[Dict] - 缓存的响应

#### set

设置缓存。

```python
success = cache.set(
    query="查询文本",
    response={"answer": "答案"},
    ttl=3600
)
```

**返回：** bool - 是否设置成功

#### delete

删除缓存。

```python
success = cache.delete("查询文本")
```

**返回：** bool - 是否删除成功

#### clear

清空所有缓存。

```python
success = cache.clear()
```

**返回：** bool - 是否清空成功

#### get_stats

获取缓存统计信息。

```python
stats = cache.get_stats()
```

**返回：** Dict - 统计信息

---

## EmbeddingCache

嵌入缓存实现，缓存文本到向量的转换结果。

### 初始化

```python
from src.cache import EmbeddingCache

cache = EmbeddingCache(
    redis_client=redis_client,
    prefix="embedding_cache",
    ttl=86400
)
```

**参数：**
- `redis_client`: Redis客户端
- `prefix` (str): 缓存键前缀
- `ttl` (int): 默认过期时间（秒）

### 方法

#### get

获取缓存的嵌入向量。

```python
embedding = cache.get("文本内容")
```

**返回：** Optional[List[float]] - 嵌入向量

#### set

设置嵌入向量缓存。

```python
success = cache.set(
    text="文本内容",
    embedding=[0.1] * 1536,
    ttl=3600
)
```

**返回：** bool - 是否设置成功

#### batch_get

批量获取嵌入向量。

```python
embeddings = cache.batch_get(["文本1", "文本2"])
```

**返回：** List[Optional[List[float]]] - 嵌入向量列表

#### batch_set

批量设置嵌入向量缓存。

```python
success = cache.batch_set(
    texts=["文本1", "文本2"],
    embeddings=[[0.1] * 1536, [0.2] * 1536]
)
```

**返回：** bool - 是否全部设置成功

---

## VectorIndex

向量索引管理类。

### 初始化

```python
from src.vector import VectorIndex

index = VectorIndex(
    redis_client=redis_client,
    index_name="vector_index"
)
```

### 方法

#### create

创建向量索引。

```python
success = index.create(
    dimensions=1536,
    distance_metric="COSINE"
)
```

#### add

添加向量到索引。

```python
doc_id = index.add(
    embedding=[0.1] * 1536,
    metadata={"text": "文本内容"},
    id="optional_id"
)
```

#### add_batch

批量添加向量到索引。

```python
doc_ids = index.add_batch(
    embeddings=[[0.1] * 1536, [0.2] * 1536],
    metadata_list=[{"text": "文本1"}, {"text": "文本2"}]
)
```

#### delete

删除文档。

```python
success = index.delete("doc_id")
```

#### get

获取文档。

```python
doc = index.get("doc_id")
```

#### info

获取索引信息。

```python
info = index.info()
```

#### drop

删除索引。

```python
success = index.drop()
```

---

## VectorSearch

向量搜索类。

### 初始化

```python
from src.vector import VectorSearch

search = VectorSearch(
    redis_client=redis_client,
    index_name="vector_index"
)
```

### 方法

#### search

执行向量相似性搜索。

```python
results = search.search(
    query_embedding=[0.1] * 1536,
    top_k=10,
    filters={"category": "test"}
)
```

#### hybrid_search

执行混合搜索（向量 + 文本）。

```python
results = search.hybrid_search(
    query_embedding=[0.1] * 1536,
    text_query="搜索文本",
    top_k=10,
    vector_weight=0.7,
    text_weight=0.3
)
```

#### search_by_category

按类别搜索。

```python
results = search.search_by_category(
    query_embedding=[0.1] * 1536,
    category="test_category",
    top_k=10
)
```

#### search_by_time_range

按时间范围搜索。

```python
results = search.search_by_time_range(
    query_embedding=[0.1] * 1536,
    start_time=1000000,
    end_time=2000000,
    top_k=10
)
```

---

## 工具函数

### calculate_similarity

计算两个向量的相似度。

```python
from src.utils import calculate_similarity

similarity = calculate_similarity(
    vector1=[0.1] * 1536,
    vector2=[0.2] * 1536,
    metric="cosine"
)
```

**参数：**
- `vector1`: 向量1
- `vector2`: 向量2
- `metric`: 相似度度量方式 ("cosine", "euclidean", "dot_product")

**返回：** float - 相似度分数

### generate_id

生成唯一ID。

```python
from src.utils import generate_id

doc_id = generate_id()
```

**返回：** str - UUID字符串