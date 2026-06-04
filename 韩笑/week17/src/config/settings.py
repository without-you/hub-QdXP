"""
配置管理模块
支持从YAML文件加载配置，并提供类型安全的配置访问
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = False

    @property
    def url(self) -> str:
        """生成Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class VectorConfig:
    """向量索引配置"""
    default_index_name: str = "vector_index"
    default_dimensions: int = 1536
    default_distance_metric: str = "COSINE"
    index_type: str = "FLAT"
    initial_cap: int = 1000
    block_size: int = 1000


@dataclass
class SemanticCacheConfig:
    """语义缓存配置"""
    prefix: str = "semantic_cache"
    similarity_threshold: float = 0.95
    default_ttl: int = 3600
    max_entries: int = 10000


@dataclass
class EmbeddingCacheConfig:
    """嵌入缓存配置"""
    prefix: str = "embedding_cache"
    default_ttl: int = 86400
    max_entries: int = 50000
    batch_size: int = 100


@dataclass
class EvictionConfig:
    """过期策略配置"""
    strategy: str = "lru"
    check_interval: int = 300
    min_hit_rate: float = 0.1


@dataclass
class CacheConfig:
    """缓存配置"""
    semantic: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    embedding: EmbeddingCacheConfig = field(default_factory=EmbeddingCacheConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)


@dataclass
class EmbeddingModelConfig:
    """嵌入模型配置"""
    provider: str = "qwen"
    model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dimensions: int = 1024
    batch_size: int = 32
    max_retries: int = 3


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class ServerConfig:
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    debug: bool = False


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """安全配置"""
    api_key_header: str = "X-API-Key"
    rate_limit: int = 100
    allowed_origins: list = field(default_factory=lambda: ["*"])


class Settings:
    """
    应用配置类

    提供类型安全的配置访问，支持从YAML文件加载配置
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.yaml
        """
        self._config_path = config_path or self._find_config_file()
        self._config_data: Dict[str, Any] = {}

        # 初始化配置对象
        self.llm = LLMConfig()
        self.redis = RedisConfig()
        self.vector = VectorConfig()
        self.cache = CacheConfig()
        self.embedding = EmbeddingModelConfig()
        self.logging = LoggingConfig()
        self.server = ServerConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()

        # 加载配置
        if self._config_path:
            self.load(self._config_path)

    def _find_config_file(self) -> Optional[str]:
        """
        查找配置文件

        Returns:
            Optional[str]: 配置文件路径
        """
        # 按优先级查找配置文件
        search_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path("config/config.yaml"),
            Path("config/config.yml"),
            Path.home() / ".vector-cache" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        return None

    def load(self, config_path: str) -> None:
        """
        从文件加载配置

        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config_data = yaml.safe_load(f) or {}

        # 更新配置对象
        self._update_from_dict(self._config_data)

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典更新配置

        Args:
            data: 配置字典
        """
        # 更新LLM配置
        if "llm" in data:
            llm_data = data["llm"]
            self.llm = LLMConfig(
                provider=llm_data.get("provider", self.llm.provider),
                model=llm_data.get("model", self.llm.model),
                api_key=llm_data.get("api_key", self.llm.api_key),
                base_url=llm_data.get("base_url", self.llm.base_url),
                max_tokens=llm_data.get("max_tokens", self.llm.max_tokens),
                temperature=llm_data.get("temperature", self.llm.temperature),
                timeout=llm_data.get("timeout", self.llm.timeout)
            )

        # 更新Redis配置
        if "redis" in data:
            redis_data = data["redis"]
            self.redis = RedisConfig(
                host=redis_data.get("host", self.redis.host),
                port=redis_data.get("port", self.redis.port),
                db=redis_data.get("db", self.redis.db),
                password=redis_data.get("password", self.redis.password),
                max_connections=redis_data.get("max_connections", self.redis.max_connections),
                socket_timeout=redis_data.get("socket_timeout", self.redis.socket_timeout),
                socket_connect_timeout=redis_data.get("socket_connect_timeout", self.redis.socket_connect_timeout),
                decode_responses=redis_data.get("decode_responses", self.redis.decode_responses)
            )

        # 更新向量配置
        if "vector" in data:
            vector_data = data["vector"]
            self.vector = VectorConfig(
                default_index_name=vector_data.get("default_index_name", self.vector.default_index_name),
                default_dimensions=vector_data.get("default_dimensions", self.vector.default_dimensions),
                default_distance_metric=vector_data.get("default_distance_metric", self.vector.default_distance_metric),
                index_type=vector_data.get("index_type", self.vector.index_type),
                initial_cap=vector_data.get("initial_cap", self.vector.initial_cap),
                block_size=vector_data.get("block_size", self.vector.block_size)
            )

        # 更新缓存配置
        if "cache" in data:
            cache_data = data["cache"]

            # 语义缓存配置
            if "semantic" in cache_data:
                semantic_data = cache_data["semantic"]
                self.cache.semantic = SemanticCacheConfig(
                    prefix=semantic_data.get("prefix", self.cache.semantic.prefix),
                    similarity_threshold=semantic_data.get("similarity_threshold", self.cache.semantic.similarity_threshold),
                    default_ttl=semantic_data.get("default_ttl", self.cache.semantic.default_ttl),
                    max_entries=semantic_data.get("max_entries", self.cache.semantic.max_entries)
                )

            # 嵌入缓存配置
            if "embedding" in cache_data:
                embedding_data = cache_data["embedding"]
                self.cache.embedding = EmbeddingCacheConfig(
                    prefix=embedding_data.get("prefix", self.cache.embedding.prefix),
                    default_ttl=embedding_data.get("default_ttl", self.cache.embedding.default_ttl),
                    max_entries=embedding_data.get("max_entries", self.cache.embedding.max_entries),
                    batch_size=embedding_data.get("batch_size", self.cache.embedding.batch_size)
                )

            # 过期策略配置
            if "eviction" in cache_data:
                eviction_data = cache_data["eviction"]
                self.cache.eviction = EvictionConfig(
                    strategy=eviction_data.get("strategy", self.cache.eviction.strategy),
                    check_interval=eviction_data.get("check_interval", self.cache.eviction.check_interval),
                    min_hit_rate=eviction_data.get("min_hit_rate", self.cache.eviction.min_hit_rate)
                )

        # 更新嵌入模型配置
        if "embedding" in data:
            embedding_data = data["embedding"]
            self.embedding = EmbeddingModelConfig(
                provider=embedding_data.get("provider", self.embedding.provider),
                model=embedding_data.get("model", self.embedding.model),
                api_key=embedding_data.get("api_key", self.embedding.api_key),
                base_url=embedding_data.get("base_url", self.embedding.base_url),
                dimensions=embedding_data.get("dimensions", self.embedding.dimensions),
                batch_size=embedding_data.get("batch_size", self.embedding.batch_size),
                max_retries=embedding_data.get("max_retries", self.embedding.max_retries)
            )

        # 更新日志配置
        if "logging" in data:
            logging_data = data["logging"]
            self.logging = LoggingConfig(
                level=logging_data.get("level", self.logging.level),
                format=logging_data.get("format", self.logging.format),
                file=logging_data.get("file", self.logging.file),
                max_bytes=logging_data.get("max_bytes", self.logging.max_bytes),
                backup_count=logging_data.get("backup_count", self.logging.backup_count)
            )

        # 更新服务配置
        if "server" in data:
            server_data = data["server"]
            self.server = ServerConfig(
                host=server_data.get("host", self.server.host),
                port=server_data.get("port", self.server.port),
                workers=server_data.get("workers", self.server.workers),
                reload=server_data.get("reload", self.server.reload),
                debug=server_data.get("debug", self.server.debug)
            )

        # 更新监控配置
        if "monitoring" in data:
            monitoring_data = data["monitoring"]
            self.monitoring = MonitoringConfig(
                enabled=monitoring_data.get("enabled", self.monitoring.enabled),
                metrics_port=monitoring_data.get("metrics_port", self.monitoring.metrics_port),
                health_check_interval=monitoring_data.get("health_check_interval", self.monitoring.health_check_interval)
            )

        # 更新安全配置
        if "security" in data:
            security_data = data["security"]
            self.security = SecurityConfig(
                api_key_header=security_data.get("api_key_header", self.security.api_key_header),
                rate_limit=security_data.get("rate_limit", self.security.rate_limit),
                allowed_origins=security_data.get("allowed_origins", self.security.allowed_origins)
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（支持点号分隔的路径，如 'llm.provider'）
            default: 默认值

        Returns:
            Any: 配置值
        """
        keys = key.split(".")
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": self.llm.api_key,
                "base_url": self.llm.base_url,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "timeout": self.llm.timeout
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "password": self.redis.password,
                "max_connections": self.redis.max_connections,
                "socket_timeout": self.redis.socket_timeout,
                "socket_connect_timeout": self.redis.socket_connect_timeout,
                "decode_responses": self.redis.decode_responses
            },
            "vector": {
                "default_index_name": self.vector.default_index_name,
                "default_dimensions": self.vector.default_dimensions,
                "default_distance_metric": self.vector.default_distance_metric,
                "index_type": self.vector.index_type,
                "initial_cap": self.vector.initial_cap,
                "block_size": self.vector.block_size
            },
            "cache": {
                "semantic": {
                    "prefix": self.cache.semantic.prefix,
                    "similarity_threshold": self.cache.semantic.similarity_threshold,
                    "default_ttl": self.cache.semantic.default_ttl,
                    "max_entries": self.cache.semantic.max_entries
                },
                "embedding": {
                    "prefix": self.cache.embedding.prefix,
                    "default_ttl": self.cache.embedding.default_ttl,
                    "max_entries": self.cache.embedding.max_entries,
                    "batch_size": self.cache.embedding.batch_size
                },
                "eviction": {
                    "strategy": self.cache.eviction.strategy,
                    "check_interval": self.cache.eviction.check_interval,
                    "min_hit_rate": self.cache.eviction.min_hit_rate
                }
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
                "batch_size": self.embedding.batch_size,
                "max_retries": self.embedding.max_retries
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "reload": self.server.reload,
                "debug": self.server.debug
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_port": self.monitoring.metrics_port,
                "health_check_interval": self.monitoring.health_check_interval
            },
            "security": {
                "api_key_header": self.security.api_key_header,
                "rate_limit": self.security.rate_limit,
                "allowed_origins": self.security.allowed_origins
            }
        }

    def save(self, config_path: Optional[str] = None) -> None:
        """
        保存配置到文件

        Args:
            config_path: 配置文件路径，默认为原配置文件路径
        """
        save_path = config_path or self._config_path
        if not save_path:
            raise ValueError("未指定配置文件路径")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def __repr__(self) -> str:
        return f"Settings(config_path={self._config_path})"


# 全局配置实例
_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局配置实例

    Returns:
        Settings: 配置实例
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings


def load_config(config_path: str) -> Settings:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Settings: 配置实例
    """
    global _global_settings
    _global_settings = Settings(config_path)
    return _global_settings


def reload_config() -> Settings:
    """
    重新加载配置

    Returns:
        Settings: 配置实例
    """
    global _global_settings
    if _global_settings and _global_settings._config_path:
        _global_settings.load(_global_settings._config_path)
    return _global_settings