"""
配置模块测试
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config import Settings, get_settings, load_config


class TestSettings:
    """测试Settings类"""

    def test_init_with_default(self):
        """测试默认初始化"""
        settings = Settings()
        assert settings.llm.provider == "deepseek"
        assert settings.redis.host == "127.0.0.1"
        assert settings.redis.port == 6379
        assert settings.vector.default_dimensions == 1536

    def test_load_from_file(self):
        """测试从文件加载配置"""
        # 创建临时配置文件
        config_content = """
llm:
  provider: openai
  model: gpt-4
  api_key: test_key
  base_url: https://api.openai.com

redis:
  host: test_host
  port: 6380
  db: 1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            settings = Settings(config_path)
            assert settings.llm.provider == "openai"
            assert settings.llm.model == "gpt-4"
            assert settings.llm.api_key == "test_key"
            assert settings.redis.host == "test_host"
            assert settings.redis.port == 6380
            assert settings.redis.db == 1
        finally:
            os.unlink(config_path)

    def test_redis_url(self):
        """测试Redis URL生成"""
        settings = Settings()
        # 测试带密码的URL
        assert settings.redis.url == "redis://:123456@127.0.0.1:6379/0"

        # 测试不带密码的URL
        settings.redis.password = None
        assert settings.redis.url == "redis://127.0.0.1:6379/0"

    def test_get_method(self):
        """测试get方法"""
        settings = Settings()

        # 测试获取存在的配置
        assert settings.get("llm.provider") == "deepseek"
        assert settings.get("redis.host") == "127.0.0.1"

        # 测试获取不存在的配置
        assert settings.get("nonexistent.key") is None
        assert settings.get("nonexistent.key", "default") == "default"

    def test_to_dict(self):
        """测试转换为字典"""
        settings = Settings()
        config_dict = settings.to_dict()

        assert "llm" in config_dict
        assert "redis" in config_dict
        assert "vector" in config_dict
        assert "cache" in config_dict
        assert config_dict["llm"]["provider"] == "deepseek"

    def test_save(self):
        """测试保存配置"""
        settings = Settings()
        settings.llm.temperature = 0.9

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            settings.save(config_path)

            # 重新加载配置
            new_settings = Settings(config_path)
            assert new_settings.llm.temperature == 0.9
        finally:
            os.unlink(config_path)


class TestGlobalSettings:
    """测试全局配置函数"""

    def test_get_settings(self):
        """测试获取全局配置"""
        settings1 = get_settings()
        settings2 = get_settings()

        # 应该返回同一个实例
        assert settings1 is settings2

    def test_load_config(self):
        """测试加载配置"""
        # 创建临时配置文件
        config_content = """
llm:
  provider: qwen
  model: qwen-turbo
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            settings = load_config(config_path)
            assert settings.llm.provider == "qwen"
            assert settings.llm.model == "qwen-turbo"
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])