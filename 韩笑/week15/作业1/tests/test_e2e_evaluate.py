"""完整的端到端测试脚本（含评价）"""
import requests
import json
import os
import time
from 作业1.app.services.evaluate import evaluate_answer, format_evaluation_report

BASE_URL = "http://localhost:8000"
PDF_PATH = "docs/【第一章】机器学习和scikit-learn介绍.pdf"


def test_health():
    """测试1: 健康检查"""
    print("\n" + "="*60)
    print("【测试1】健康检查")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"状态码: {resp.status_code}")
        print(f"响应: {resp.json()}")
        assert resp.status_code == 200
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_root():
    """测试2: 根路径"""
    print("\n" + "="*60)
    print("【测试2】根路径")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"状态码: {resp.status_code}")
        print(f"响应: {resp.json()}")
        assert resp.status_code == 200
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_list_documents():
    """测试3: 文档列表"""
    print("\n" + "="*60)
    print("【测试3】获取文档列表")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/api/document/list", timeout=10)
        print(f"状态码: {resp.status_code}")
        data = resp.json()
        print(f"响应: {data}")
        print(f"文档总数: {data.get('total', 0)}")
        assert resp.status_code == 200
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_upload_document():
    """测试4: 上传文档"""
    print("\n" + "="*60)
    print("【测试4】上传PDF文档")
    print("="*60)
    if not os.path.exists(PDF_PATH):
        print(f"PDF文件不存在: {PDF_PATH}")
        return None

    try:
        with open(PDF_PATH, "rb") as f:
            files = {"file": ("机器学习第一章.pdf", f, "application/pdf")}
            resp = requests.post(f"{BASE_URL}/api/document/upload", files=files, timeout=30)

        print(f"状态码: {resp.status_code}")
        print(f"响应: {resp.json()}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"文档ID: {data.get('document_id')}")
            print(f"文件名: {data.get('filename')}")
            print(f"状态: {data.get('status')}")
            print("✓ 通过")
            return data.get("document_id")
        else:
            print("✗ 失败")
            return None
    except Exception as e:
        print(f"✗ 失败: {e}")
        return None


def check_document_status(document_id):
    """测试5: 查看文档处理状态"""
    print("\n" + "="*60)
    print(f"【测试5】查看文档状态")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/api/document/{document_id}", timeout=10)
        print(f"状态码: {resp.status_code}")
        data = resp.json()
        print(f"响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
        return data
    except Exception as e:
        print(f"查询失败: {e}")
        return None


def test_chat_with_evaluation():
    """测试6: 问答测试（含评价）"""
    print("\n" + "="*60)
    print("【测试6】问答测试 - 检索 + 生成 + 评价")
    print("="*60)

    # 测试问题（需要根据实际文档内容调整）
    test_queries = [
        {
            "query": "什么是机器学习？",
            "expected_page": 1,
            "expected_file": "机器学习",
            "expected_answer": "机器学习是人工智能的一个分支"
        },
        {
            "query": "scikit-learn是什么？",
            "expected_page": 1,
            "expected_file": "机器学习",
            "expected_answer": "scikit-learn是Python的机器学习库"
        }
    ]

    results = []
    for i, q in enumerate(test_queries):
        print(f"\n--- 问题 {i+1} ---")
        print(f"问题: {q['query']}")

        try:
            resp = requests.post(
                f"{BASE_URL}/api/chat",
                json={"query": q["query"], "top_k": 5},
                timeout=60
            )
            print(f"状态码: {resp.status_code}")

            if resp.status_code == 200:
                result = resp.json()
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                source_files = result.get("source_files", [])

                print(f"生成答案: {answer[:200]}...")
                print(f"来源文件: {source_files}")
                print(f"来源数量: {len(sources)}")

                # 提取页面号
                predicted_pages = [s.get("page_number", 0) for s in sources]

                # 评价
                scores = evaluate_answer(
                    predicted_answer=answer,
                    expected_answer=q["expected_answer"],
                    expected_page=q["expected_page"],
                    predicted_pages=predicted_pages,
                    expected_file=q["expected_file"],
                    predicted_files=source_files
                )

                print(format_evaluation_report(scores))

                results.append({
                    "query": q["query"],
                    "scores": scores,
                    "success": True
                })
            else:
                print(f"请求失败: {resp.text}")
                results.append({
                    "query": q["query"],
                    "scores": None,
                    "success": False,
                    "error": resp.text
                })

        except Exception as e:
            print(f"✗ 失败: {e}")
            results.append({
                "query": q["query"],
                "scores": None,
                "success": False,
                "error": str(e)
            })

    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    total_score = 0
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['query']}")
        if r["scores"]:
            print(f"   总分: {r['scores']['total']['score']:.2f}/{r['scores']['total']['max']:.2f}")
            total_score += r['scores']['total']['score']

    if results:
        avg_score = total_score / len(results)
        print(f"\n平均得分: {avg_score:.2f}/1.00 ({avg_score*100:.1f}%)")

    return results


def main():
    print("="*60)
    print("多模态RAG系统 - 端到端测试（含评价）")
    print("="*60)
    print(f"API地址: {BASE_URL}")
    print(f"测试PDF: {PDF_PATH}")

    # 1-3: 基础接口测试
    if not test_health():
        print("\n服务未启动，请先运行: python -m uvicorn app.main:app --reload")
        return

    test_root()
    test_list_documents()

    # 4: 上传文档
    doc_id = test_upload_document()

    # 5: 检查文档状态
    if doc_id:
        print("\n注意: Kafka Consumer Worker需要单独启动")
        print("请在另一个终端运行: python -m app.workers.document_worker")
        print("等待处理完成后继续...")
        time.sleep(3)
        check_document_status(doc_id)

    # 6: 问答评价测试
    test_chat_with_evaluation()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()