"""G2–G4：页面调用的读写入口；浏览不创建空库，不持有共享的当前集合。"""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import uuid

from src.core.trace import TraceCollector, TraceContext
from src.ingestion.document_manager import DeleteResult, DocumentManager
from src.ingestion.pipeline import IngestionPipeline, SUPPORTED_EXTENSIONS
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.mcp_server.tools._storage import configured_storage, validate_collection
from src.mcp_server.tools.list_collections import ListCollectionsTool


class _ExistingImages:
    def __init__(self, path):
        self.path = path

    def list_images(self, collection=None, doc_hash=None):
        return ImageStorage.list_existing(self.path, collection, doc_hash)


class _ExistingHistory:
    def __init__(self, path):
        self.path = path

    def list_processed(self, collection=None):
        return SQLiteIntegrityChecker.list_existing(self.path, collection)


class DataService:
    def __init__(self, settings=None, *, data_dir=None, embedding_client=None):
        self.settings, self.data_root = configured_storage(settings, data_dir)
        self._pipeline_data_dir = self.data_root if data_dir is not None else None
        self.embedding_client = embedding_client

    def list_collections(self):
        return [item.name for item in ListCollectionsTool(self.settings).list_collections()]

    @contextmanager
    def _manager(self, collection, *, writable=False):
        validate_collection(collection)
        store = VectorStoreFactory.create(self.settings, collection_name=collection, create_if_missing=False)
        try:
            history_path = self.data_root / "db" / "ingestion_history.db"
            image_path = self.data_root / "db" / "image_index.db"
            images = ImageStorage(str(image_path), str(self.data_root / "images")) if writable else _ExistingImages(image_path)
            history = SQLiteIntegrityChecker(str(history_path)) if writable else _ExistingHistory(history_path)
            yield DocumentManager(store, BM25Indexer(str(self.data_root / "db" / "bm25" / collection)), images, history)
        finally:
            store.client.close()

    def list_documents(self, collection="default"):
        if collection not in self.list_collections():
            return []
        with self._manager(collection) as manager:
            return [asdict(doc) for doc in manager.list_documents(collection)]

    def get_document_detail(self, doc_id, collection="default", *, source_path=None):
        if collection not in self.list_collections():
            return None
        with self._manager(collection) as manager:
            detail = manager.get_document_detail(doc_id, source_path=source_path)
            return asdict(detail) if detail is not None else None

    def get_chunks(self, source_hash, collection="default", *, source_path=None):
        if collection not in self.list_collections():
            return []
        with self._manager(collection) as manager:
            filters = {"source_path": source_path} if source_path is not None else {"doc_hash": source_hash}
            return sorted(manager.chroma.get_by_metadata(filters), key=lambda c: (c["metadata"].get("chunk_index", 0), c["id"]))

    def get_images(self, source_hash, collection="default"):
        return ImageStorage.list_existing(self.data_root / "db" / "image_index.db", collection, source_hash)

    def get_collection_stats(self, collection="default"):
        if collection not in self.list_collections():
            return {"collection": collection, "document_count": 0, "chunk_count": 0, "image_count": 0}
        with self._manager(collection) as manager:
            return asdict(manager.get_collection_stats(collection))

    def delete_document(self, source_path, collection="default", source_hash=None):
        documents = self.list_documents(collection)
        if not any(d["source_path"] == source_path and (source_hash is None or source_hash in d["source_hashes"]) for d in documents):
            return DeleteResult(success=False, errors=["Document is not present in the selected collection"])
        with self._manager(collection, writable=True) as manager:
            return manager.delete_document(source_path, collection, source_hash)

    def ingest_file(self, path, collection="default", *, force=False, source_root=None, on_progress=None):
        """显式执行才调用模型，关闭 Pipeline 并按配置收集一次 Trace。"""
        validate_collection(collection)
        trace = TraceContext(trace_type="ingestion") if self.settings.observability.trace_enabled else None
        if trace is not None:
            trace.metadata["source"] = "dashboard"
        pipeline = IngestionPipeline(self.settings, collection, force, embedding=self.embedding_client,
                                     data_dir=self._pipeline_data_dir, source_root=source_root)
        try:
            result = pipeline.run(path, trace=trace, on_progress=on_progress)
            return {"result": result.to_dict(), "trace_id": trace.trace_id if trace is not None else None}
        finally:
            try:
                pipeline.close()
            finally:
                if trace is not None:
                    trace.finish()
                    TraceCollector.from_settings(self.settings).collect(trace)

    def ingest_uploads(self, files: list[tuple[str, bytes]], collection="default", *, force=False,
                       overwrite=False, on_progress=None):
        """上传源保存在稳定路径；支持同级图片附件，校验整批后再写入。"""
        validate_collection(collection)
        upload_root = self.data_root / "uploads"
        directory = upload_root / collection
        resources = {".png", ".jpg", ".jpeg", ".webp"}
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.data_root.resolve()):
            raise ValueError("Upload directory escapes managed data root")
        names, documents = set(), []
        for name, content in files:
            if (not name or name in {".", ".."} or any(c in name for c in "/\\") or name in names
                    or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS | resources):
                raise ValueError("Upload names must be unique supported filenames")
            names.add(name)
            if not isinstance(content, bytes) or len(content) > 50 * 1024 * 1024:
                raise ValueError("Each upload must be bytes within 50 MiB")
            target = directory / name
            if not target.resolve().is_relative_to(directory.resolve()) or target.is_symlink():
                raise ValueError("Upload destination escapes managed directory")
            if target.exists() and target.read_bytes() != content and not overwrite:
                raise FileExistsError("A different upload with this name already exists")
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(target)
        if not documents:
            raise ValueError("At least one PDF or Markdown document is required")
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files:
            target = directory / name
            if target.is_file() and target.read_bytes() != content:
                backup = upload_root / ".history" / f"{uuid.uuid4().hex}_{name}"
                if not backup.resolve().is_relative_to(upload_root.resolve()):
                    raise ValueError("Upload backup escapes managed directory")
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(target.read_bytes())
            target.write_bytes(content)
        return [self.ingest_file(path, collection, force=force, source_root=directory, on_progress=on_progress) for path in documents]
