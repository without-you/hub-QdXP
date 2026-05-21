"""文档解析Worker - 消费Kafka消息，调用mineru解析PDF"""
import os
from 作业1.app.services.kafka_service import kafka_service
from 作业1.app.services.mineru_service import mineru_service
from 作业1.app.services.embedding_service import embedding_service
from 作业1.app.services.milvus_service import milvus_service
from 作业1.app.services.sqlite_service import sqlite_service


def process_document(document_id: str, file_path: str):
    """处理单个文档：调用mineru解析 → chunk切分 → embedding → 存储"""
    print(f"[Worker] 开始处理文档: {document_id}")

    # 1. 更新状态为processing
    sqlite_service.update_document(document_id, status="processing")

    try:
        # 2. 调用mineru解析
        parse_result = mineru_service.parse_pdf_sync(file_path)
        # parse_result = {"markdown": "...", "images": [...], "total_pages": N}

        markdown = parse_result.get("markdown", "")
        images = parse_result.get("images", [])
        total_pages = parse_result.get("total_pages", 0)

        # 3. 处理文本chunk
        text_chunks_count = 0
        if markdown:
            text_chunks = _split_text(markdown)
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():
                    chunk_id = f"{document_id}_text_{i}"
                    embedding = embedding_service.embed_text(chunk)
                    page_num = _estimate_page(chunk)

                    # 存SQLite
                    sqlite_service.create_chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=chunk,
                        content_type="text",
                        source_file=file_path,
                        page_number=page_num
                    )
                    # 存Milvus
                    milvus_service.insert(
                        id=chunk_id,
                        content=chunk,
                        content_type="text",
                        source_file=file_path,
                        page_number=page_num,
                        embedding=embedding
                    )
                    text_chunks_count += 1

        # 4. 处理图片chunk
        images_count = 0
        for img in images:
            chunk_id = f"{document_id}_img_{img['page']}"
            image_path = img.get("path", "")
            caption = img.get("caption", "图片")

            if os.path.exists(image_path):
                try:
                    embedding = embedding_service.embed_text(caption)

                    sqlite_service.create_chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=caption,
                        content_type="image",
                        source_file=file_path,
                        page_number=img["page"],
                        image_path=image_path
                    )
                    milvus_service.insert(
                        id=chunk_id,
                        content=caption,
                        content_type="image",
                        source_file=file_path,
                        page_number=img["page"],
                        embedding=embedding,
                        image_path=image_path
                    )
                    images_count += 1
                except Exception as e:
                    print(f"[Worker] 处理图片失败 {image_path}: {e}")

        # 5. 更新文档状态
        sqlite_service.update_document(
            document_id,
            status="completed",
            total_pages=total_pages,
            text_chunks=text_chunks_count,
            images=images_count
        )
        print(f"[Worker] 文档 {document_id} 处理完成: {text_chunks_count}文本chunks, {images_count}张图片")

    except Exception as e:
        print(f"[Worker] 文档 {document_id} 处理失败: {e}")
        sqlite_service.update_document(document_id, status="failed")


def _split_text(text: str, chunk_size: int = 500) -> list[str]:
    """按段落切分文本"""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _estimate_page(text: str) -> int:
    """估算页码"""
    import re
    match = re.search(r'第\s*(\d+)\s*页', text)
    return int(match.group(1)) if match else 1


def main():
    """Worker主循环"""
    print("[Worker] 启动文档解析Worker...")
    consumer = kafka_service.get_consumer()
    for message in consumer:
        data = message.value
        if data.get("action") == "parse":
            process_document(data["document_id"], data["file_path"])


if __name__ == "__main__":
    main()