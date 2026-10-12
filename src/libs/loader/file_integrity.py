"""SHA256 与 SQLite 处理记录；跳过范围按上游仅由内容哈希决定。"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(file_path: str | Path) -> str:
    """分段读取原始文件；Loader 复用此实现，但不因此创建处理数据库。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise OSError(f"Path is not a file: {file_path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class FileIntegrityChecker(ABC):
    """替换处理记录存储的 interface；不负责解析或索引写入。"""

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """计算文件内容指纹。"""
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """仅已成功处理的内容可跳过。"""
        raise NotImplementedError

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, collection: str | None = None) -> None:
        """由完整摄取流程在成功后调用。"""
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        """记录失败，不阻止下一次重试。"""
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """每次操作自行关闭连接；WAL 不等于跨索引事务或任务抢占锁。"""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS ingestion_history (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                collection TEXT,
                error_msg TEXT,
                processed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)")
            connection.commit()

    def compute_sha256(self, file_path: str) -> str:
        """按 64 KiB 分段计算，避免把整个大文件放入内存。"""
        return _sha256_file(file_path)

    def should_skip(self, file_hash: str) -> bool:
        """collection 和处理配置不参与本阶段的上游跳过规则。"""
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute("SELECT status FROM ingestion_history WHERE file_hash = ?", (file_hash,)).fetchone()
        return row is not None and row[0] == "success"

    def mark_success(self, file_hash: str, file_path: str, collection: str | None = None) -> None:
        """保留首次记录时间，更新成功状态及 collection，清空旧错误。"""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with closing(sqlite3.connect(self.db_path)) as connection:
                connection.execute("""INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                    ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path, status='success', collection=excluded.collection,
                    error_msg=NULL, updated_at=excluded.updated_at""", (file_hash, file_path, collection, now, now))
                connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to mark success: {type(exc).__name__}") from None

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        """失败记录保留已有 collection 和首次时间，与上游行为一致。"""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with closing(sqlite3.connect(self.db_path)) as connection:
                connection.execute("""INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                    ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path, status='failed', error_msg=excluded.error_msg,
                    updated_at=excluded.updated_at""", (file_hash, file_path, error_msg, now, now))
                connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to mark failure: {type(exc).__name__}") from None

    def close(self) -> None:
        """保留上游生命周期入口；没有持久连接需要额外关闭。"""

    @staticmethod
    def list_existing(db_path: str | Path, collection: str | None = None) -> list[dict]:
        """只读已有历史，不为浏览页面创建数据库。"""
        path = Path(db_path).resolve()
        if not path.is_file():
            return []
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            query, values = "SELECT * FROM ingestion_history", []
            if collection is not None:
                query += " WHERE collection = ?"
                values.append(collection)
            return [dict(row) for row in connection.execute(query + " ORDER BY updated_at DESC", values)]

    def list_processed(self, collection: str | None = None) -> list[dict]:
        """供管理接口读取历史；失败记录也返回，由调用方展示真实状态。"""
        return self.list_existing(self.db_path, collection)

    def remove_record(self, file_hash: str, *, file_path: str | None = None, collection: str | None = None) -> bool:
        """删除明确的历史记录；可同时约束来源路径和集合。"""
        query, values = "DELETE FROM ingestion_history WHERE file_hash = ?", [file_hash]
        if file_path is not None:
            query += " AND file_path = ?"
            values.append(file_path)
        if collection is not None:
            query += " AND collection = ?"
            values.append(collection)
        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute(query, values).rowcount
            connection.commit()
        return count > 0
