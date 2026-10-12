"""C13：图片文件留在磁盘，SQLite 只登记 ID、路径和来源。"""

import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path


class ImageStorage:
    """沿用上游 image_index 表；每次操作独立打开和关闭连接。"""

    def __init__(self, db_path: str = "data/db/image_index.db", images_root: str = "data/images"):
        self.db_path = str(resolve_path(db_path))
        self.images_root = resolve_path(images_root)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS image_index (
                image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
                collection TEXT, doc_hash TEXT, page_num INTEGER, created_at TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")
            connection.commit()

    def register_image(self, image_id: str, file_path: str | Path, collection: str | None = None,
                       doc_hash: str | None = None, page_num: int | None = None) -> str:
        """登记 Loader 已保存的文件，不复制或移动；Markdown 页码允许为空。"""
        if not image_id.strip():
            raise ValueError("image_id must be nonempty")
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("Image file does not exist")
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("""INSERT OR REPLACE INTO image_index
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""", (image_id, str(path), collection, doc_hash, page_num,
                                               datetime.now(timezone.utc).isoformat()))
            connection.commit()
        return str(path)

    def save_image(self, image_id: str, image_data: bytes | Path | str, collection: str | None = None,
                   doc_hash: str | None = None, page_num: int | None = None, extension: str = "png") -> str:
        """按上游复制原始字节；extension 只是后缀，不执行格式转换。"""
        namespace = collection if collection is not None else "default"
        for component in (namespace, image_id, extension):
            if not component.strip() or component in {".", ".."} or any(char in component for char in "/\\"):
                raise ValueError("Image destination components must be single names")
        destination = self.images_root / namespace / f"{image_id}.{extension}"
        # 防止已有目录或文件符号链接把写入引向图片根目录之外。
        if not destination.resolve().is_relative_to(self.images_root.resolve()):
            raise ValueError("Image destination escapes images_root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image_data, bytes):
            destination.write_bytes(image_data)
        elif isinstance(image_data, (str, Path)):
            source = Path(image_data)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        else:
            raise ValueError("image_data must be bytes or a file path")
        return self.register_image(image_id, destination, collection, doc_hash, page_num)

    def get_image_path(self, image_id: str) -> str | None:
        """返回登记的绝对路径；登记存在不代表文件尚在磁盘上。"""
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute("SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)).fetchone()
        return row[0] if row else None

    def image_exists(self, image_id: str) -> bool:
        """与上游一致，只判断登记是否存在。"""
        return self.get_image_path(image_id) is not None

    @staticmethod
    def get_existing_image_path(image_id: str, db_path: str | Path, collection: str | None = None) -> str | None:
        """MCP 读取已有登记；不创建数据库、表或目录，指定集合时不跨集合查图。"""
        path = resolve_path(db_path)
        if not path.is_file():
            return None
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            query, parameters = "SELECT file_path FROM image_index WHERE image_id = ?", [image_id]
            if collection is not None:
                query += " AND collection = ?"
                parameters.append(collection)
            row = connection.execute(query, parameters).fetchone()
        return row[0] if row else None

    def list_images(self, collection: str | None = None, doc_hash: str | None = None) -> list[dict[str, Any]]:
        """按集合及文档哈希筛选元数据，不返回图片字节。"""
        query, values = "SELECT * FROM image_index WHERE 1=1", []
        if collection is not None:
            query += " AND collection = ?"
            values.append(collection)
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            values.append(doc_hash)
        query += " ORDER BY created_at, image_id"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def close(self) -> None:
        """兼容上游调用入口；本实现没有需要延后关闭的连接。"""

    @staticmethod
    def list_existing(db_path: str | Path, collection: str | None = None, doc_hash: str | None = None) -> list[dict]:
        """只读已有图片登记，不创建目录或 SQLite 表。"""
        path = resolve_path(db_path)
        if not path.is_file():
            return []
        query, values = "SELECT * FROM image_index WHERE 1=1", []
        for key, value in (("collection", collection), ("doc_hash", doc_hash)):
            if value is not None:
                query += f" AND {key} = ?"
                values.append(value)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query + " ORDER BY created_at, image_id", values)]

    def delete_image(self, image_id: str, *, collection: str) -> bool:
        """只清理该集合登记，未共享的受管理文件移到 .trash，不碰源图片。"""
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute("SELECT file_path FROM image_index WHERE image_id = ? AND collection = ?",
                                     (image_id, collection)).fetchone()
            if row is None:
                return False
            path = Path(row[0]).resolve()
            root = self.images_root.resolve()
            if not path.is_relative_to(root):
                raise ValueError("Registered image is outside managed images root")
            shared = connection.execute("SELECT COUNT(*) FROM image_index WHERE file_path = ? AND image_id != ?",
                                        (row[0], image_id)).fetchone()[0]
            if not shared and path.is_file():
                trash = root / ".trash"
                if not trash.resolve().is_relative_to(root):
                    raise ValueError("Image trash escapes managed root")
                trash.mkdir(parents=True, exist_ok=True)
                path.rename(trash / f"{uuid.uuid4().hex}_{path.name}")
            connection.execute("DELETE FROM image_index WHERE image_id = ? AND collection = ?", (image_id, collection))
            connection.commit()
        return True
