"""答案评价函数"""
from typing import List


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """计算两个文本的Jaccard相似系数"""
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def evaluate_answer(
    predicted_answer: str,
    expected_answer: str,
    expected_page: int,
    predicted_pages: List[int],
    expected_file: str,
    predicted_files: List[str]
) -> dict:
    """
    评估答案质量

    参数:
        predicted_answer: 模型预测的答案
        expected_answer: 期望的答案
        expected_page: 期望的页面号
        predicted_pages: 预测的页面号列表
        expected_file: 期望的文件名
        predicted_files: 预测的文件名列表

    返回:
        dict: 包含各项得分和总分
    """
    scores = {}

    # 1. 页面匹配度 (满分0.25)
    page_score = 0.25 if expected_page in predicted_pages else 0.0
    scores["page_match"] = {
        "score": page_score,
        "max": 0.25,
        "expected": expected_page,
        "predicted": predicted_pages,
        "matched": expected_page in predicted_pages
    }

    # 2. 文件名匹配度 (满分0.25)
    expected_basename = expected_file.split("/")[-1].split("\\")[-1]
    pred_basenames = [f.split("/")[-1].split("\\")[-1] for f in predicted_files]
    file_score = 0.25 if expected_basename in pred_basenames else 0.0
    scores["file_match"] = {
        "score": file_score,
        "max": 0.25,
        "expected": expected_basename,
        "predicted": pred_basenames,
        "matched": expected_basename in pred_basenames
    }

    # 3. 答案内容相似度 (满分0.5)
    jaccard = calculate_jaccard_similarity(predicted_answer, expected_answer)
    content_score = jaccard * 0.5
    scores["content_similarity"] = {
        "score": content_score,
        "max": 0.5,
        "jaccard": jaccard,
        "expected_length": len(expected_answer),
        "predicted_length": len(predicted_answer)
    }

    # 总分
    total = page_score + file_score + content_score
    scores["total"] = {
        "score": total,
        "max": 1.0,
        "percentage": f"{total*100:.1f}%"
    }

    return scores


def format_evaluation_report(scores: dict) -> str:
    """格式化评价报告"""
    report = []
    report.append("\n" + "="*50)
    report.append("答案评价报告")
    report.append("="*50)

    # 页面匹配
    pm = scores["page_match"]
    status = "✓" if pm["matched"] else "✗"
    report.append(f"{status} 页面匹配度: {pm['score']:.2f}/{pm['max']:.2f}")
    report.append(f"   期望页面: {pm['expected']}, 预测页面: {pm['predicted']}")

    # 文件匹配
    fm = scores["file_match"]
    status = "✓" if fm["matched"] else "✗"
    report.append(f"{status} 文件名匹配度: {fm['score']:.2f}/{fm['max']:.2f}")
    report.append(f"   期望文件: {fm['expected']}, 预测文件: {fm['predicted']}")

    # 内容相似度
    cs = scores["content_similarity"]
    report.append(f"  内容相似度: {cs['score']:.2f}/{cs['max']:.2f}")
    report.append(f"   Jaccard系数: {cs['jaccard']:.4f}")

    # 总分
    total = scores["total"]
    report.append("-"*50)
    report.append(f"总分: {total['score']:.2f}/{total['max']:.2f} ({total['percentage']})")
    report.append("="*50)

    return "\n".join(report)


if __name__ == "__main__":
    # 测试用例
    test_result = evaluate_answer(
        predicted_answer="机器学习是人工智能的一个分支，主要涉及从数据中学习模式。",
        expected_answer="机器学习是人工智能的分支，专注于从数据中自动学习模式。",
        expected_page=3,
        predicted_pages=[1, 3, 5],
        expected_file="机器学习简介.pdf",
        predicted_files=["机器学习简介.pdf", "深度学习.pdf"]
    )
    print(format_evaluation_report(test_result))
    print(f"\n详细得分: {test_result}")