"""MinerU文档解析服务（Agent轻量API — 签名上传模式）"""
import time
import requests
from 作业1.app.core.config import settings


class MineruService:
    def __init__(self):
        self.base_url = settings.MINERU_BASE_URL

    def parse_pdf_sync(self, file_path: str, language: str = "ch",
                       enable_table: bool = True, is_ocr: bool = False,
                       enable_formula: bool = True, page_range: str = "1-20",
                       timeout: int = 300) -> dict:
        """
        签名上传流程：
        1. POST /parse/file → task_id + OSS签名上传URL
        2. PUT 文件到 OSS
        3. 轮询 GET /parse/{task_id} → markdown_url
        4. 下载 markdown
        """
        file_name = file_path.replace("\\", "/").split("/")[-1]

        # Step 1: 获取签名上传 URL
        payload = {
            "file_name": file_name,
            "language": language,
            "enable_table": enable_table,
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "page_range": page_range,
        }
        resp = requests.post(f"{self.base_url}/parse/file", json=payload, timeout=30)
        result = resp.json()
        if result["code"] != 0:
            raise Exception(f"MinerU任务创建失败: {result.get('msg', '未知错误')}")

        task_id = result["data"]["task_id"]
        file_url = result["data"]["file_url"]
        print(f"[MinerU] 任务已创建, task_id: {task_id}")

        # Step 2: PUT 上传文件到 OSS
        with open(file_path, "rb") as f:
            put_resp = requests.put(file_url, data=f, timeout=120)
            if put_resp.status_code not in (200, 201):
                raise Exception(f"MinerU文件上传失败, HTTP {put_resp.status_code}")
        print(f"[MinerU] 文件上传成功")

        # Step 3: 轮询等待结果
        markdown_text = self._poll_result(task_id, timeout)

        return {"markdown": markdown_text, "total_pages": 0}

    def _poll_result(self, task_id: str, timeout: int = 300, interval: int = 3) -> str:
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{self.base_url}/parse/{task_id}", timeout=15)
            result = resp.json()
            state = result["data"]["state"]
            elapsed = int(time.time() - start)

            if state == "done":
                markdown_url = result["data"]["markdown_url"]
                print(f"[MinerU] [{elapsed}s] 解析完成")
                md_resp = requests.get(markdown_url, timeout=60)
                return md_resp.text

            if state == "failed":
                err_msg = result["data"].get("err_msg", "未知错误")
                raise Exception(f"MinerU解析失败: {err_msg}")

            print(f"[MinerU] [{elapsed}s] {state}...")
            time.sleep(interval)

        raise Exception(f"MinerU轮询超时 ({timeout}s), task_id: {task_id}")

    def parse_pdf(self, file_path: str) -> dict:
        """异步兼容"""
        return self.parse_pdf_sync(file_path)


mineru_service = MineruService()
