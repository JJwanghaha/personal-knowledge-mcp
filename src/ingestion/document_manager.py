"""G2：协调文档浏览与四种存储的删除，不删除原始文档。"""

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass
class DocumentInfo:
    source_path: str
    source_hash: str
    collection: str
    chunk_count: int = 0
    image_count: int = 0
    processed_at: str | None = None
    status: str = "indexed"
    source_hashes: list[str] = field(default_factory=list)


@dataclass
class DocumentDetail(DocumentInfo):
    chunk_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    success: bool = True
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    collection: str
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


class DocumentManager:
    """实例绑定一个 Chroma 集合；不把内容哈希误当成跨来源唯一身份。"""

    def __init__(self, chroma_store, bm25_indexer, image_storage, file_integrity):
        self.chroma, self.bm25, self.images, self.integrity = chroma_store, bm25_indexer, image_storage, file_integrity

    def _collection(self, collection):
        selected = collection or self.chroma.collection_name
        if selected != self.chroma.collection_name:
            raise ValueError("Manager is bound to another collection")
        return selected

    def list_documents(self, collection=None) -> list[DocumentInfo]:
        selected = self._collection(collection)
        records = self.integrity.list_processed(selected)
        images = self.images.list_images(collection=selected)
        grouped = {}
        # 历史补时间/状态，实际块决定数量；失败/部分删除的历史也可见，以便重试。
        for history in records:
            grouped.setdefault(history["file_path"], {"chunks": [], "hashes": set(), "history": history})["hashes"].add(history["file_hash"])
        for chunk in self.chroma.get_by_metadata():
            source = chunk["metadata"].get("source_path")
            if source:
                group = grouped.setdefault(source, {"chunks": [], "hashes": set(), "history": {}})
                group["chunks"].append(chunk)
                if chunk["metadata"].get("doc_hash"):
                    group["hashes"].add(chunk["metadata"]["doc_hash"])
        result = []
        for source, group in sorted(grouped.items()):
            history = group["history"]
            hashes = sorted(group["hashes"])
            result.append(DocumentInfo(source, history.get("file_hash", hashes[0] if hashes else ""), selected,
                len(group["chunks"]), sum(image["doc_hash"] in hashes for image in images),
                history.get("updated_at"), history.get("status", "indexed"), hashes))
        return result

    def get_document_detail(self, doc_id: str, *, source_path: str | None = None) -> DocumentDetail | None:
        matches = [doc for doc in self.list_documents() if (source_path is None or doc.source_path == source_path)
                   and (doc_id == doc.source_path or doc_id in doc.source_hashes)]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError("Hash matches multiple source paths; supply source_path")
        doc = matches[0]
        chunks = self.chroma.get_by_metadata({"source_path": doc.source_path})
        images = self.images.list_images(collection=doc.collection)
        return DocumentDetail(**vars(doc), chunk_ids=[c["id"] for c in chunks],
                              image_ids=[img["image_id"] for img in images if img["doc_hash"] in doc.source_hashes])

    def get_collection_stats(self, collection=None) -> CollectionStats:
        selected = self._collection(collection)
        return CollectionStats(selected, len(self.list_documents(selected)), self.chroma.collection.count(),
                               len(self.images.list_images(collection=selected)))

    def delete_document(self, source_path: str, collection: str = "default", source_hash: str | None = None) -> DeleteResult:
        """删除所选来源在该集合的全部版本；部分失败保留历史，允许重试。"""
        selected = self._collection(collection)
        doc = next((d for d in self.list_documents(selected) if d.source_path == source_path), None)
        if doc is None or source_hash is not None and source_hash not in doc.source_hashes:
            return DeleteResult(success=False, errors=["Document is not present in the selected collection"])
        other_hashes = {value for d in self.list_documents(selected) if d.source_path != source_path for value in d.source_hashes}
        result = DeleteResult()
        try:
            result.chunks_deleted = self.chroma.delete_by_metadata({"source_path": source_path})
        except Exception as exc:
            result.errors.append(f"Chroma: {type(exc).__name__}")
        try:
            # C12 最终 ID 以来源路径 SHA256 前8位开头，不以 doc_hash 开头。
            prefix = sha256(source_path.encode("utf-8")).hexdigest()[:8] + "_"
            result.bm25_removed = self.bm25.remove_document(prefix, selected)
        except Exception as exc:
            result.errors.append(f"BM25: {type(exc).__name__}")
        # 若块删除失败，保留图片；相同内容的另一来源仍使用图片时也不删。
        if not any(error.startswith("Chroma:") for error in result.errors):
            try:
                for image in self.images.list_images(collection=selected):
                    if image["doc_hash"] in doc.source_hashes and image["doc_hash"] not in other_hashes:
                        result.images_deleted += self.images.delete_image(image["image_id"], collection=selected)
            except Exception as exc:
                result.errors.append(f"Images: {type(exc).__name__}")
        if not result.errors:
            try:
                for value in doc.source_hashes:
                    result.integrity_removed = self.integrity.remove_record(value, file_path=source_path, collection=selected) or result.integrity_removed
            except Exception as exc:
                result.errors.append(f"History: {type(exc).__name__}")
        result.success = not result.errors
        return result
