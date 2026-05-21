"""文档上传接口"""
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from 作业1.app.models.schema import DocumentUploadResponse
from 作业1.app.core.config import settings
from 作业1.app.services.kafka_service import kafka_service
from 作业1.app.services.sqlite_service import sqlite_service

router = APIRouter(prefix="/api/document", tags=["文档管理"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传PDF文档，发送到Kafka等待解析"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    document_id = str(uuid.uuid4())
    filename = f"{document_id}.pdf"
    file_path = os.path.join(settings.PDF_DIR, filename)

    os.makedirs(settings.PDF_DIR, exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 创建文档记录
    sqlite_service.create_document(document_id, file.filename, file_path)

    # 发送到Kafka
    try:
        kafka_service.send_parse_task(document_id, file_path, file.filename)
    except Exception as e:
        sqlite_service.update_document(document_id, status="failed")
        raise HTTPException(status_code=500, detail=f"任务提交失败: {str(e)}")

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="pending"
    )


@router.get("/list")
async def list_documents():
    """获取文档列表"""
    docs = sqlite_service.list_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/{document_id}")
async def get_document(document_id: str):
    """获取单个文档详情"""
    doc = sqlite_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc