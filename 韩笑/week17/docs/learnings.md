# 学习笔记与踩坑记录

## Redis Stack + RediSearch

### KNN向量搜索语法

**问题**：Python redis-py的Query对象生成的KNN查询语法不正确，导致搜索失败。

**解决方案**：使用原始`execute_command`方法，手动构建查询：

```python
results = redis.execute_command(
    'FT.SEARCH', index_name,
    '*=>[KNN 10 @embedding $BLOB AS score]',
    'PARAMS', '2', 'BLOB', query_vector,
    'SORTBY', 'score',
    'RETURN', '3', 'text', 'score', 'category',
    'DIALECT', '2'
)
```

**关键点**：
- 参数名必须是`$BLOB`（不是`$vec`）
- 必须指定`DIALECT 2`
- 向量必须转为float32字节：`np.array(embedding, dtype=np.float32).tobytes()`

### 过滤条件语法

```python
# TAG字段精确匹配
filter_str = f"@category:{{{value}}}"

# 数值范围查询
filter_str = f"@score:[{min_val} {max_val}]"
```

## LLM和Embedding分离配置

**设计决策**：LLM（chat）和Embedding使用不同的provider。

**原因**：
- DeepSeek的embedding API返回404，不支持
- Qwen text-embedding-v3效果好，1024维度合适

**实现**：
```python
# adapter.py
self.provider = self._create_provider()           # LLM provider
self.embedding_provider = self._create_embedding_provider()  # Embedding provider
```

Embedding配置独立的api_key和base_url，fallback到LLM配置。

## Qwen Embedding API

通过Dashscope的OpenAI兼容接口调用：

```python
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=base_url)
response = client.embeddings.create(model="text-embedding-v3", input=text)
```

- base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 维度：1024

## 常见错误

1. **requirements.txt编码错误**：Windows下GBK编码问题，需要用UTF-8重写
2. **init() got unexpected keyword argument 'config'**：SDK客户端传了config参数但类没接收，需要在__init__加`config=None`
3. **重复搜索结果**：旧测试数据残留，需要用`ft(index).dropindex(delete_documents=True)`清理
