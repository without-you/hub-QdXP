"""端到端集成测试脚本"""
import requests
import json
import os
import time

BASE_URL = "http://localhost:8000"
PDF_PATH = "docs/【第一章】机器学习和scikit-learn介绍.pdf"


def test_health():
    """测试1: 健康检查"""
    print("\n" + "="*60)
    print("测试1: 健康检查")
    print("="*60)
    resp = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.json()}")
    assert resp.status_code == 200, "健康检查失败"
    print("✓ 通过")


def test_root():
    """测试2: 根路径"""
    print("\n" + "="*60)
    print("测试2: 根路径")
    print("="*60)
    resp = requests.get(f"{BASE_URL}/")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.json()}")
    assert resp.status_code == 200
    print("✓ 通过")


def test_list_documents():
    """测试3: 文档列表"""
    print("\n" + "="*60)
    print("测试3: 获取文档列表")
    print("="*60)
    resp = requests.get(f"{BASE_URL}/api/document/list")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.json()}")
    assert resp.status_code == 200
    print("✓ 通过")


def test_upload_document():
    """测试4: 上传文档"""
    print("\n" + "="*60)
    print("测试4: 上传PDF文档")
    print("="*60)
    if not os.path.exists(PDF_PATH):
        print(f"PDF文件不存在: {PDF_PATH}")
        return None

    with open(PDF_PATH, "rb") as f:
        files = {"file": ("机器学习第一章.pdf", f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/api/document/upload", files=files)

    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.json()}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"文档ID: {data.get('document_id')}")
        print(f"状态: {data.get('status')}")
        return data.get("document_id")
    return None


def check_document_status(document_id):
    """测试5: 查看文档处理状态"""
    print("\n" + "="*60)
    print(f"测试5: 查看文档状态 ({document_id})")
    print("="*60)
    resp = requests.get(f"{BASE_URL}/api/document/{document_id}")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {resp.json()}")
    return resp.json()


def test_chat_query():
    """测试6: 问答测试"""
    print("\n" + "="*60)
    print("测试6: 问答测试")
    print("="*60)

    # 先检查Milvus中是否有数据
    print("注意: Milvus检索需要先有索引数据，Worker处理完文档后才能检索")

    queries = [
        "什么是机器学习？",
        "scikit-learn是什么？",
    ]

    for query in queries:
        print(f"\n问题: {query}")
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={"query": query, "top_k": 3}
        )
        print(f"状态码: {resp.status_code}")
        result = resp.json()
        print(f"答案: {result.get('answer', '')[:200]}...")
        print(f"来源文件: {result.get('source_files', [])}")
        print(f"来源数量: {len(result.get('sources', []))}")


def main():
    print("="*60)
    print("多模态RAG系统 - 端到端测试")
    print("="*60)

    # 1-3: 基础接口测试
    test_health()
    test_root()
    test_list_documents()

    # 4: 上传文档（会发送到Kafka，但Worker需单独启动）
    doc_id = test_upload_document()

    # 5: 如果拿到了doc_id，检查状态
    if doc_id:
        # 等待Worker处理
        print("\n提示: Worker需单独启动处理Kafka消息")
        print("请在另一个终端运行: python -m app.workers.document_worker")
        time.sleep(2)
        check_document_status(doc_id)

    # 6: 问答测试（需要文档处理完成后才能检索到内容）
    test_chat_query()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()