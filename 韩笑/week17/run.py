#!/usr/bin/env python3
"""
快速启动脚本
用于启动向量检索与智能缓存服务平台
"""

import sys
import io
import argparse
from pathlib import Path

# Windows下设置stdout编码为UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import get_settings, load_config
from src.sdk import VectorCacheClient


def test_connection():
    """测试Redis连接"""
    print("测试Redis连接...")
    settings = get_settings()

    try:
        import redis
        client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password
        )
        client.ping()
        print("✓ Redis连接成功")
        return True
    except Exception as e:
        print(f"✗ Redis连接失败: {e}")
        return False


def test_llm():
    """测试LLM连接"""
    print("\n测试LLM连接...")
    settings = get_settings()

    try:
        from src.llm import LLMAdapter
        adapter = LLMAdapter(settings)

        response = adapter.chat("你好")
        print(f"✓ LLM连接成功")
        print(f"  响应: {response.get('content', 'N/A')[:100]}...")
        return True
    except Exception as e:
        print(f"✗ LLM连接失败: {e}")
        return False


def run_example():
    """运行示例"""
    print("\n运行示例...")
    try:
        from examples.basic_usage import main
        main()
    except Exception as e:
        print(f"运行示例失败: {e}")


def show_config():
    """显示当前配置"""
    print("\n当前配置:")
    settings = get_settings()

    print(f"  LLM提供商: {settings.llm.provider}")
    print(f"  LLM模型: {settings.llm.model}")
    print(f"  LLM Base URL: {settings.llm.base_url}")
    print(f"  Redis主机: {settings.redis.host}:{settings.redis.port}")
    print(f"  默认索引: {settings.vector.default_index_name}")
    print(f"  向量维度: {settings.vector.default_dimensions}")


def serve(host: str = "0.0.0.0", port: int = 8000):
    """启动Web服务"""
    try:
        import uvicorn
        print(f"\n启动Web服务...")
        print(f"  地址: http://{host}:{port}")
        print(f"  按 Ctrl+C 停止服务\n")
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=False
        )
    except ImportError:
        print("✗ 缺少uvicorn依赖，请执行: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n服务已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="向量检索与智能缓存服务平台"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径",
        default="config.yaml"
    )
    parser.add_argument(
        "--test-redis",
        action="store_true",
        help="测试Redis连接"
    )
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="测试LLM连接"
    )
    parser.add_argument(
        "--run-example",
        action="store_true",
        help="运行示例"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="显示当前配置"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="启动Web服务"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web服务主机（默认0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web服务端口（默认8000）"
    )

    args = parser.parse_args()

    # 加载配置
    if args.config:
        try:
            load_config(args.config)
            print(f"✓ 配置文件加载成功: {args.config}")
        except FileNotFoundError:
            print(f"✗ 配置文件未找到: {args.config}")
            print("使用默认配置")

    # 执行命令
    if args.test_redis:
        test_connection()

    if args.test_llm:
        test_llm()

    if args.run_example:
        run_example()

    if args.show_config:
        show_config()

    if args.serve:
        serve(args.host, args.port)

    # 如果没有指定任何命令，显示帮助
    if not any([args.test_redis, args.test_llm, args.run_example, args.show_config, args.serve]):
        parser.print_help()
        print("\n示例用法:")
        print("  python run.py --test-redis      # 测试Redis连接")
        print("  python run.py --test-llm        # 测试LLM连接")
        print("  python run.py --show-config     # 显示当前配置")
        print("  python run.py --run-example     # 运行示例")
        print("  python run.py --serve           # 启动Web服务")


if __name__ == "__main__":
    main()