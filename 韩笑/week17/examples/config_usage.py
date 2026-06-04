"""
配置使用示例
演示如何使用配置系统
"""

from src.config import get_settings, load_config
from src.sdk import VectorCacheClient


def main():
    """主函数"""

    # 1. 使用默认配置
    print("=== 1. 使用默认配置 ===")
    settings = get_settings()

    print(f"LLM提供商: {settings.llm.provider}")
    print(f"LLM模型: {settings.llm.model}")
    print(f"LLM Base URL: {settings.llm.base_url}")
    print(f"Redis URL: {settings.redis.url}")
    print(f"默认索引名称: {settings.vector.default_index_name}")
    print(f"向量维度: {settings.vector.default_dimensions}")

    # 2. 加载自定义配置文件
    print("\n=== 2. 加载自定义配置文件 ===")
    try:
        custom_settings = load_config("config.yaml")
        print(f"配置文件加载成功: {custom_settings}")
        print(f"LLM提供商: {custom_settings.llm.provider}")
        print(f"LLM模型: {custom_settings.llm.model}")
    except FileNotFoundError as e:
        print(f"配置文件未找到: {e}")

    # 3. 使用配置创建客户端
    print("\n=== 3. 使用配置创建客户端 ===")
    try:
        client = VectorCacheClient(config=settings)
        print("客户端创建成功")
        print(f"索引名称: {client.index_name}")
        print(f"Redis连接: {client.config.redis.url}")

        # 测试LLM调用
        print("\n=== 4. 测试LLM调用 ===")
        response = client.chat(
            query="你好，请介绍一下自己",
            use_cache=False
        )

        if response.get("response"):
            print(f"LLM响应: {response['response'].get('content', 'N/A')}")
            print(f"是否缓存: {response.get('from_cache', False)}")
        else:
            print("LLM调用失败或未配置")

        # 关闭连接
        client.close()
        print("\n连接已关闭")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    # 4. 展示配置的完整结构
    print("\n=== 5. 配置完整结构 ===")
    config_dict = settings.to_dict()
    print("配置字典:")
    for key, value in config_dict.items():
        print(f"  {key}: {type(value).__name__}")

    # 5. 展示如何修改配置
    print("\n=== 6. 修改配置示例 ===")
    settings.llm.temperature = 0.9
    settings.cache.semantic.default_ttl = 7200
    print(f"修改后的温度: {settings.llm.temperature}")
    print(f"修改后的缓存TTL: {settings.cache.semantic.default_ttl}")

    # 6. 保存配置
    print("\n=== 7. 保存配置 ===")
    try:
        settings.save("config_modified.yaml")
        print("配置已保存到 config_modified.yaml")
    except Exception as e:
        print(f"保存失败: {e}")


if __name__ == "__main__":
    main()