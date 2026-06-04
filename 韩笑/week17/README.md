# 向量检索与智能缓存服务平台

## 项目背景

随着公司各业务线对AI能力，特别是大语言模型（LLM）和检索增强生成（RAG）应用的深入探索，公司面临一系列共通的底层技术挑战。多个团队在独立开发中重复建设向量数据库接入、语义缓存、对话记忆管理等模块，导致技术栈碎片化、资源利用率低，且难以保证生产环境的性能与稳定性。与此同时，对高并发、低延迟的实时AI推理与检索需求日益增长，亟需一个统一、高效、可靠的基座服务。

**典型场景**：
- 项目A：输入文本进行情感分析
- 项目B：输入文本进行情感分析
- 问题：重复调用大模型，浪费资金和时间
- 解决方案：Agent应用的缓存模块，避免重复调用，减少耗时和成本

本项目旨在借鉴行业先进实践（如RedisVL的设计理念），构建一个公司内部统一的、生产就绪的**向量检索与智能缓存服务平台**。其核心是充分利用公司已部署的高性能Redis集群，通过封装成熟的AI原生数据模式与操作，为上层业务提供一个简单易用、功能强大且易于扩展的Python SDK与配套服务，从而赋能各业务团队快速、低成本地构建高质量的AI应用，并确保核心组件的性能与可维护性。

## 项目架构

```
week17/
├── CLAUDE.md                          # 项目指导文档
├── README.md                          # 项目说明（本文件）
├── requirements.txt                   # Python依赖
├── setup.py                           # 安装配置
├── .gitignore                         # Git忽略文件
│
├── src/                               # 源代码目录
│   ├── __init__.py                    # 包初始化
│   │
│   ├── sdk/                           # Python SDK模块
│   │   ├── __init__.py                # SDK包初始化
│   │   └── client.py                  # 主客户端类
│   │
│   ├── cache/                         # 缓存模块
│   │   ├── __init__.py                # 缓存包初始化
│   │   ├── semantic_cache.py          # 语义缓存实现
│   │   └── embedding_cache.py         # 嵌入缓存实现
│   │
│   ├── vector/                        # 向量管理模块
│   │   ├── __init__.py                # 向量包初始化
│   │   ├── index.py                   # 向量索引管理
│   │   └── search.py                  # 向量搜索实现
│   │
│   ├── llm/                           # LLM适配器模块
│   │   ├── __init__.py                # LLM包初始化
│   │   ├── adapter.py                 # LLM适配器
│   │   └── providers.py               # LLM提供商实现
│   │
│   ├── config/                        # 配置模块
│   │   ├── __init__.py                # 配置包初始化
│   │   └── settings.py                # 配置管理
│   │
│   └── utils/                         # 工具函数模块
│       ├── __init__.py                # 工具包初始化
│       └── helpers.py                 # 辅助函数
│
├── tests/                             # 测试代码
│   ├── __init__.py                    # 测试包初始化
│   ├── test_sdk.py                    # SDK测试
│   ├── test_cache.py                  # 缓存模块测试
│   └── test_vector.py                 # 向量模块测试
│
├── docs/                              # 文档目录
│   └── api.md                         # API文档
│
└── examples/                          # 使用示例
    └── basic_usage.py                 # 基础使用示例
```

## 核心功能模块

### 1. Python SDK (src/sdk/)

**主客户端类：VectorCacheClient**

提供统一的接口来管理向量索引、执行语义搜索和使用缓存功能。

```python
from src.sdk import VectorCacheClient

client = VectorCacheClient(
    redis_url="redis://localhost:6379",
    index_name="my_index"
)

# 创建索引
client.create_index(dimensions=1536, distance_metric="COSINE")

# 添加文本
doc_id = client.add_text("中国的首都是北京", metadata={"category": "geography"})

# 语义搜索
results = client.search("北京是中国的什么？", top_k=10)
```

### 2. 缓存模块 (src/cache/)

#### 语义缓存 (SemanticCache)

基于语义相似度的缓存，用于减少LLM调用。

- **精确匹配**：通过哈希快速查找
- **语义匹配**：基于向量相似度查找（待实现）
- **自动过期**：支持TTL过期策略

#### 嵌入缓存 (EmbeddingCache)

缓存文本到向量的转换结果，避免重复计算。

- **批量操作**：支持批量获取和设置
- **自动过期**：默认24小时过期
- **节省成本**：避免重复调用嵌入模型

### 3. 向量管理模块 (src/vector/)

#### 向量索引 (VectorIndex)

管理向量索引的创建、更新和删除。

- **索引创建**：支持多种距离度量（COSINE, L2, IP）
- **批量操作**：支持批量添加和删除
- **元数据存储**：支持丰富的元数据字段

#### 向量搜索 (VectorSearch)

提供向量相似性搜索和混合查询功能。

- **向量搜索**：基于向量相似度的搜索
- **混合搜索**：向量 + 文本的混合搜索
- **过滤查询**：支持元数据过滤
- **时间范围**：支持时间范围查询

### 4. 工具函数 (src/utils/)

提供通用的辅助函数：

- `calculate_similarity()` - 计算向量相似度
- `generate_id()` - 生成唯一ID
- `normalize_vector()` - 向量归一化
- `batch_cosine_similarity()` - 批量计算余弦相似度

## 技术栈

- **向量数据库**：Redis（利用现有集群，支持RediSearch模块）
- **向量索引**：支持FLAT和HNSW索引类型
- **相似度度量**：余弦相似度、欧氏距离、点积
- **LLM提供商**：DeepSeek、OpenAI、通义千问、LiteLLM
- **配置管理**：YAML配置文件，支持环境变量
- **后端框架**：Python 3.9+
- **测试框架**：pytest
- **代码质量**：black, flake8, mypy

## 应用场景

### 场景1：情感分析Agent缓存

```python
# 第一次查询 - 缓存未命中，调用LLM
query = "请判断用户的情感：我今天很开心"
response = client.get_cached_response(query)
if response is None:
    response = call_llm(query)  # 调用大模型
    client.set_cached_response(query, response, ttl=3600)

# 第二次查询 - 缓存命中，直接返回
response = client.get_cached_response(query)  # 命中缓存，不调用LLM
```

**效果**：
- 减少LLM调用次数
- 降低响应延迟
- 节约API成本

### 场景2：RAG系统中的文档检索

```python
# 添加文档到索引
for doc in documents:
    client.add_text(
        text=doc["content"],
        metadata={
            "category": doc["category"],
            "source": doc["source"],
            "timestamp": doc["timestamp"]
        }
    )

# 语义搜索
results = client.search(
    query="用户的查询",
    top_k=5,
    filters={"category": "technical"}
)
```

### 场景3：智能客服知识库

```python
# 缓存常见问题的回答
faq_cache = SemanticCache(redis_client, prefix="faq")

# 用户提问
question = "如何重置密码？"

# 检查缓存
cached_answer = faq_cache.get(question)
if cached_answer:
    return cached_answer
else:
    # 查询知识库
    answer = search_knowledge_base(question)
    faq_cache.set(question, answer, ttl=86400)  # 缓存24小时
    return answer
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置项目

#### 方式1：使用配置文件

编辑 `config.yaml` 文件，配置LLM和Redis参数：

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  api_key: your_api_key_here
  base_url: https://api.deepseek.com

redis:
  host: localhost
  port: 6379
  db: 0
```

#### 方式2：使用环境变量

复制 `.env.example` 为 `.env` 并填入配置：

```bash
cp .env.example .env
# 编辑 .env 文件，填入实际配置
```

### 3. 启动Redis

确保Redis服务正在运行，并已安装RediSearch模块。

```bash
redis-server
```

### 4. 测试配置

```bash
# 测试Redis连接
python run.py --test-redis

# 测试LLM连接
python run.py --test-llm

# 显示当前配置
python run.py --show-config
```

### 5. 运行示例

```bash
# 运行基础示例
python examples/basic_usage.py

# 运行配置示例
python examples/config_usage.py
```

### 6. 运行测试

```bash
pytest tests/
```

## 配置系统

本项目使用YAML格式的配置文件，支持多种配置源：

- **配置文件**：`config.yaml`
- **环境变量**：`.env`文件
- **代码默认值**

### 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.provider` | LLM提供商 | deepseek |
| `llm.model` | 模型名称 | deepseek-v4-pro |
| `llm.api_key` | API密钥 | - |
| `llm.base_url` | API基础URL | https://api.deepseek.com |
| `redis.host` | Redis主机 | localhost |
| `redis.port` | Redis端口 | 6379 |
| `vector.default_dimensions` | 向量维度 | 1536 |
| `cache.semantic.default_ttl` | 语义缓存过期时间 | 3600 |

### 配置使用示例

```python
from src.config import get_settings
from src.sdk import VectorCacheClient

# 获取配置
settings = get_settings()

# 使用配置创建客户端
client = VectorCacheClient(config=settings)

# 或者直接使用默认配置
client = VectorCacheClient()
```

详细配置说明请参考：[配置系统使用指南](docs/configuration.md)

## 开发指南

### 代码规范

- 使用black进行代码格式化
- 使用flake8进行代码检查
- 使用mypy进行类型检查

```bash
# 格式化代码
black src/ tests/

# 检查代码
flake8 src/ tests/

# 类型检查
mypy src/
```

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_sdk.py

# 生成覆盖率报告
pytest --cov=src tests/
```

### 文档

API文档位于 `docs/api.md`，包含所有类和方法的详细说明。

## 验收标准

- [x] 实现基础向量索引创建和管理
- [x] 实现向量相似性搜索
- [x] 实现元数据过滤和混合查询
- [x] 实现语义缓存功能
- [x] 实现嵌入缓存功能
- [x] 实现对话历史管理
- [x] 提供完整的Python SDK
- [x] 编写单元测试和集成测试
- [x] 编写使用文档和示例代码

## 设计原则

1. **统一性**：提供标准化接口，结束重复造轮子
2. **高性能**：基于Redis集群，支持高并发低延迟
3. **易用性**：简单易用的Python SDK
4. **可扩展性**：模块化设计，易于扩展新功能
5. **生产就绪**：经过充分测试，保证生产环境稳定性

## 注意事项

1. 优先利用公司现有Redis集群，避免引入新的基础设施
2. 确保SDK接口设计简洁，降低业务团队使用门槛
3. 充分考虑生产环境的性能和稳定性要求
4. 遵循公司代码规范和安全要求

## 后续计划

1. 集成实际的嵌入模型（OpenAI、HuggingFace等）
2. 实现基于向量相似度的语义匹配
3. 添加分布式锁和并发控制
4. 实现缓存预热和降级策略
5. 添加监控和告警功能
6. 优化大规模数据下的性能

## 许可证

内部项目，仅供公司内部使用。