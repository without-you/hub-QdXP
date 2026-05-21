"""SQLite元数据存储服务"""
import sqlite3
from typing import Optional, List
from 作业1.app.core.config import settings
import os


class SQLiteService:
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "metadata.db")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                total_pages INTEGER DEFAULT 0,
                text_chunks INTEGER DEFAULT 0,
                images INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT,
                content_type TEXT,
                source_file TEXT,
                page_number INTEGER,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        conn.commit()
        conn.close()

    def create_document(self, document_id: str, filename: str, file_path: str) -> str:
        """创建文档记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, filename, file_path) VALUES (?, ?, ?)",
            (document_id, filename, file_path)
        )
        conn.commit()
        conn.close()
        return document_id

    def update_document(self, document_id: str, status: str = None, total_pages: int = None,
                        text_chunks: int = None, images: int = None):
        """更新文档状态"""
        conn = self._get_conn()
        cursor = conn.cursor()
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        if status:
            updates.append(f"status = '{status}'")
        if total_pages is not None:
            updates.append(f"total_pages = {total_pages}")
        if text_chunks is not None:
            updates.append(f"text_chunks = {text_chunks}")
        if images is not None:
            updates.append(f"images = {images}")

        cursor.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
            (document_id,)
        )
        conn.commit()
        conn.close()

    def get_document(self, document_id: str) -> Optional[dict]:
        """获取文档信息"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "file_path": row[2],
                "status": row[3],
                "total_pages": row[4],
                "text_chunks": row[5],
                "images": row[6],
                "created_at": row[7],
                "updated_at": row[8]
            }
        return None

    def list_documents(self) -> List[dict]:
        """列出所有文档"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "filename": r[1],
                "file_path": r[2],
                "status": r[3],
                "total_pages": r[4],
                "text_chunks": r[5],
                "images": r[6],
                "created_at": r[7],
                "updated_at": r[8]
            }
            for r in rows
        ]

    def create_chunk(self, chunk_id: str, document_id: str, content: str,
                     content_type: str, source_file: str, page_number: int,
                     image_path: str = None) -> str:
        """创建chunk记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO chunks
               (id, document_id, content, content_type, source_file, page_number, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, document_id, content, content_type, source_file, page_number, image_path)
        )
        conn.commit()
        conn.close()
        return chunk_id

    def delete_document(self, document_id: str):
        """删除文档及其chunks"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        conn.close()


sqlite_service = SQLiteService()