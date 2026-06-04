# 配置系统使用指南

## 概述

本项目使用YAML格式的配置文件，支持多种配置源：
- 配置文件（config.yaml）
- 环境变量（.env文件）
- 代码中的默认值

配置系统提供类型安全的配置访问，支持配置验证和热重载。

## 配置文件结构

### 主配置文件：config.yaml

```yaml
# LLM配置
llm:
  provider: deepseek           # 提供商：deepseek | openai | qwen | litellm
  model: deepseek-v4-pro       # 模型名称
  api_key: sk-xxx              # API密钥
  base_url: https://api.deepseek.com  # API基础URL
  max_tokens: 4096             # 最大token数
  temperature: 0.7             # 温度参数
  timeout: 30                  # 请求超时时间（秒）

# Redis配置
redis:
  host: localhost               # Redis主机
  port: 6379                   # Redis端口
  db: 0                        # 数据库编号
  password:                    # 密码（留空表示无密码）
  max_connections: 100          # 最大连接数
  socket_timeout: 5            # Socket超时
  socket_connect_timeout: 5     # Socket连接超时
  decode_responses: false       # 是否自动解码响应

# 向量索引配置
vector:
  default_index_name: vector_index  # 默认索引名称
  default_dimensions: 1536          # 默认向量维度
  default_distance_metric: COSINE   # 默认距离度量
  index_type: FLAT                  # 索引类型
  initial_cap: 1000                 # 初始容量
  block_size: 1000                  # 块大小

# 缓存配置
cache:
  semantic:                     # 语义缓存
    prefix: semantic_cache
    similarity_threshold: 0.95
    default_ttl: 3600
    max_entries: 10000

  embedding:                    # 嵌入缓存
    prefix: embedding_cache
    default_ttl: 86400
    max_entries: 50000
    batch_size: 100

  eviction:                     # 过期策略
    strategy: lru
    check_interval: 300
    min_hit_rate: 0.1

# 嵌入模型配置
embedding:
  provider: deepseek
  model: deepseek-embedding
  dimensions: 1536
  batch_size: 32
  max_retries: 3

# 日志配置
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/app.log
  max_bytes: 10485760
  backup_count: 5

# 服务配置
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: true
  debug: false

# 监控配置
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30

# 安全配置
security:
  api_key_header: X-API-Key
  rate_limit: 100
  allowed_origins:
    - "*"
```

## 配置使用方法

### 1. 使用默认配置

```python
from src.config import get_settings

# 获取全局配置实例
settings = get_settings()

# 访问配置值
print(f"LLM提供商: {settings.llm.provider}")
print(f"Redis URL: {settings.redis.url}")
print(f"向量维度: {settings.vector.default_dimensions}")
```

### 2. 加载自定义配置文件

```python
from src.config import load_config

# 加载指定配置文件
settings = load_config("path/to/config.yaml")

# 访问配置
print(f"LLM模型: {settings.llm.model}")
```

### 3. 在SDK客户端中使用配置

```python
from src.sdk import VectorCacheClient
from src.config import get_settings

# 方式1：使用默认配置
client = VectorCacheClient()

# 方式2：使用自定义配置
settings = get_settings()
client = VectorCacheClient(config=settings)

# 方式3：覆盖特定配置
client = VectorCacheClient(
    redis_url="redis://other-host:6379",
    index_name="custom_index"
)
```

### 4. 在代码中修改配置

```python
from src.config import get_settings

settings = get_settings()

# 修改配置
settings.llm.temperature = 0.9
settings.cache.semantic.default_ttl = 7200

# 保存修改后的配置
settings.save("config_modified.yaml")
```

### 5. 使用环境变量

创建 `.env` 文件：

```bash
# .env
LLM_API_KEY=your_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
```

在代码中加载环境变量：

```python
from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

# 读取环境变量
api_key = os.getenv("LLM_API_KEY")
redis_host = os.getenv("REDIS_HOST")
```

## 配置类说明

### Settings

主配置类，包含所有配置项。

**属性：**
- `llm` - LLM配置
- `redis` - Redis配置
- `vector` - 向量索引配置
- `cache` - 缓存配置
- `embedding` - 嵌入模型配置
- `logging` - 日志配置
- `server` - 服务配置
- `monitoring` - 监控配置
- `security` - 安全配置

**方法：**
- `load(config_path)` - 从文件加载配置
- `save(config_path)` - 保存配置到文件
- `get(key, default)` - 获取配置值（支持点号分隔的路径）
- `to_dict()` - 转换为字典

### LLMConfig

LLM配置类。

**属性：**
- `provider` - 提供商名称
- `model` - 模型名称
- `api_key` - API密钥
- `base_url` - API基础URL
- `max_tokens` - 最大token数
- `temperature` - 温度参数
- `timeout` - 请求超时时间

### RedisConfig

Redis配置类。

**属性：**
- `host` - 主机
- `port` - 端口
- `db` - 数据库编号
- `password` - 密码
- `max_connections` - 最大连接数
- `socket_timeout` - Socket超时
- `socket_connect_timeout` - Socket连接超时
- `decode_responses` - 是否自动解码响应

**属性（计算）：**
- `url` - Redis连接URL

### VectorConfig

向量索引配置类。

**属性：**
- `default_index_name` - 默认索引名称
- `default_dimensions` - 默认向量维度
- `default_distance_metric` - 默认距离度量
- `index_type` - 索引类型
- `initial_cap` - 初始容量
- `block_size` - 块大小

### CacheConfig

缓存配置类。

**属性：**
- `semantic` - 语义缓存配置
- `embedding` - 嵌入缓存配置
- `eviction` - 过期策略配置

## 配置优先级

配置的优先级从高到低：

1. **代码中的参数** - 直接传递给函数的参数
2. **环境变量** - .env文件或系统环境变量
3. **配置文件** - config.yaml文件
4. **默认值** - 代码中的默认值

## 配置验证

配置系统会自动验证配置值：

```python
from src.config import get_settings

settings = get_settings()

# 验证Redis连接
try:
    import redis
    client = redis.from_url(settings.redis.url)
    client.ping()
    print("Redis连接成功")
except Exception as e:
    print(f"Redis连接失败: {e}")

# 验证LLM配置
if settings.llm.api_key:
    print("LLM API密钥已配置")
else:
    print("警告：LLM API密钥未配置")
```

## 配置热重载

支持在运行时重新加载配置：

```python
from src.config import reload_config

# 重新加载配置
settings = reload_config()
print("配置已重新加载")
```

## 最佳实践

### 1. 敏感信息管理

- 不要将API密钥提交到版本控制
- 使用 `.env` 文件存储敏感信息
- 在 `.gitignore` 中添加 `.env`

### 2. 配置文件组织

- 使用 `config.yaml` 作为默认配置
- 为不同环境创建不同配置文件：
  - `config.development.yaml` - 开发环境
  - `config.production.yaml` - 生产环境
  - `config.testing.yaml` - 测试环境

### 3. 配置验证

在应用启动时验证关键配置：

```python
from src.config import get_settings

def validate_config():
    settings = get_settings()

    # 验证Redis配置
    if not settings.redis.host:
        raise ValueError("Redis主机未配置")

    # 验证LLM配置
    if not settings.llm.api_key:
        raise ValueError("LLM API密钥未配置")

    # 验证向量配置
    if settings.vector.default_dimensions <= 0:
        raise ValueError("向量维度必须大于0")

    return True
```

### 4. 配置日志

记录配置加载过程：

```python
import logging
from src.config import load_config

logger = logging.getLogger(__name__)

def init_config():
    try:
        settings = load_config("config.yaml")
        logger.info(f"配置加载成功: {settings}")
        return settings
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        raise
```

## 故障排除

### 问题1：配置文件未找到

**错误**：`FileNotFoundError: 配置文件不存在: config.yaml`

**解决方案**：
1. 检查配置文件路径是否正确
2. 确保配置文件存在
3. 使用绝对路径

```python
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
settings = load_config(str(config_path))
```

### 问题2：配置格式错误

**错误**：`yaml.scanner.ScannerError`

**解决方案**：
1. 检查YAML语法
2. 使用YAML验证工具
3. 检查缩进是否正确

### 问题3：Redis连接失败

**错误**：`redis.exceptions.ConnectionError`

**解决方案**：
1. 检查Redis服务是否运行
2. 检查Redis配置是否正确
3. 检查网络连接

### 问题4：LLM API调用失败

**错误**：`httpx.HTTPStatusError`

**解决方案**：
1. 检查API密钥是否正确
2. 检查API URL是否正确
3. 检查网络连接

## 示例配置

### 开发环境配置

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  api_key: sk-dev-key
  base_url: https://api.deepseek.com
  temperature: 0.7

redis:
  host: localhost
  port: 6379
  db: 0

server:
  host: 0.0.0.0
  port: 8000
  debug: true
  reload: true
```

### 生产环境配置

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  api_key: ${LLM_API_KEY}  # 从环境变量读取
  base_url: https://api.deepseek.com
  temperature: 0.3

redis:
  host: ${REDIS_HOST}
  port: ${REDIS_PORT}
  password: ${REDIS_PASSWORD}
  max_connections: 200

server:
  host: 0.0.0.0
  port: 8000
  workers: 8
  debug: false
  reload: false
```