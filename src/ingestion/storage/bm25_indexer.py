"""C11：词项统计 → 倒排表 → BM25 查询；JSON 按 collection 保存。"""

import json
import math
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path


class BM25Indexer:
    """沿用上游 IDF 和打分公式；增量加入仍通过全量重建实现。"""

    def __init__(self, index_dir: str = "data/db/bm25", k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and b in [0, 1]")
        self.index_dir = resolve_path(index_dir)
        self.k1, self.b = k1, b
        self._index: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}
        self._documents: dict[str, dict[str, Any]] = {}

    def build(self, term_stats: list[dict[str, Any]], collection: str = "default", trace: Any = None) -> None:
        """覆盖该集合的完整索引；保留零词项块参与平均长度统计。"""
        if not term_stats:
            raise ValueError("Cannot build index from empty term_stats")
        self._get_index_path(collection)
        documents = {}
        for stat in term_stats:
            chunk_id = stat["chunk_id"]
            frequencies = stat["term_frequencies"]
            length = stat["doc_length"]
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in documents:
                raise ValueError("chunk_id must be a unique nonempty string")
            if not isinstance(frequencies, dict) or not isinstance(length, int) or isinstance(length, bool) or length < 0:
                raise ValueError("Invalid term statistics")
            if any(not isinstance(term, str) or not term or not isinstance(tf, int) or isinstance(tf, bool) or tf <= 0
                   for term, tf in frequencies.items()) or sum(frequencies.values()) != length:
                raise ValueError("Term frequencies must be positive counts summing to doc_length")
            documents[chunk_id] = {"chunk_id": chunk_id, "term_frequencies": dict(frequencies), "doc_length": length}
        index: dict[str, dict[str, Any]] = {}
        for stat in documents.values():
            for term, tf in stat["term_frequencies"].items():
                entry = index.setdefault(term, {"postings": []})
                entry["postings"].append({"chunk_id": stat["chunk_id"], "tf": tf, "doc_length": stat["doc_length"]})
        count = len(documents)
        for entry in index.values():
            entry["df"] = len(entry["postings"])
            entry["idf"] = math.log((count - entry["df"] + 0.5) / (entry["df"] + 0.5))
        metadata = {"num_docs": count, "avg_doc_length": sum(stat["doc_length"] for stat in documents.values()) / count,
                    "total_terms": len(index), "collection": collection}
        self._save(collection, {"metadata": metadata, "index": index, "documents": documents})
        self._metadata, self._index, self._documents = metadata, index, documents

    def load(self, collection: str = "default", trace: Any = None) -> bool:
        """缺文件时清空已加载集合，避免误用上一集合的数据。"""
        path = self._get_index_path(collection)
        if not path.exists():
            self._metadata, self._index, self._documents = {}, {}, {}
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata, index = data["metadata"], data["index"]
            if not isinstance(metadata, dict) or not isinstance(index, dict) or metadata["collection"] != collection:
                raise ValueError("Invalid index structure or collection")
            documents = data.get("documents")
            if documents is None:
                # 兼容上游仅有 metadata/index 的格式；无法恢复没有词项的块。
                documents = {}
                for term, entry in index.items():
                    for posting in entry["postings"]:
                        stat = documents.setdefault(posting["chunk_id"], {"chunk_id": posting["chunk_id"],
                                                   "term_frequencies": {}, "doc_length": posting["doc_length"]})
                        stat["term_frequencies"][term] = posting["tf"]
            if not isinstance(documents, dict):
                raise ValueError("Invalid stored documents")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid BM25 index file") from exc
        self._metadata, self._index, self._documents = metadata, index, documents
        return True

    def query(self, query_terms: list[str], top_k: int = 10, trace: Any = None) -> list[dict[str, Any]]:
        """输入已经分词的查询；不在这里调用模型或返回原文。"""
        if not self._metadata:
            raise ValueError("Call load() or build() before query()")
        if not query_terms or top_k <= 0:
            raise ValueError("query_terms must be nonempty and top_k positive")
        scores: dict[str, float] = {}
        average = self._metadata["avg_doc_length"] or 1.0
        for term in query_terms:
            entry = self._index.get(term.lower())
            if entry is None:
                continue
            for posting in entry["postings"]:
                tf = posting["tf"]
                denominator = tf + self.k1 * (1 - self.b + self.b * posting["doc_length"] / average)
                score = entry["idf"] * tf * (self.k1 + 1) / denominator
                cid = posting["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + score
        return sorted(({"chunk_id": cid, "score": score} for cid, score in scores.items()),
                      key=lambda item: item["score"], reverse=True)[:top_k]

    def rebuild(self, term_stats: list[dict[str, Any]], collection: str = "default", trace: Any = None) -> None:
        """显式表达覆盖重建，行为等同 build。"""
        self.build(term_stats, collection, trace)

    def add_documents(self, term_stats: list[dict[str, Any]], collection: str = "default",
                      doc_id: str | None = None, trace: Any = None) -> None:
        """按 ID 覆盖加入；doc_id 仍按上游前缀匹配旧块，调用方须传准确前缀。"""
        if not term_stats:
            return
        if self._metadata.get("collection") != collection:
            self.load(collection)
        existing = {cid: stat for cid, stat in self._documents.items()
                    if doc_id is None or not cid.startswith(doc_id)}
        # 同一批重复 ID 应报错，而不是被字典悄悄覆盖。
        new_ids = [stat["chunk_id"] for stat in term_stats]
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("Duplicate chunk_id in new term statistics")
        existing.update({stat["chunk_id"]: stat for stat in term_stats})
        self.build(list(existing.values()), collection, trace)

    def _get_index_path(self, collection: str) -> Path:
        if not collection.strip() or collection in {".", ".."} or any(char in collection for char in "/\\"):
            raise ValueError("collection must be a single path component")
        return self.index_dir / f"{collection}_bm25.json"

    def remove_document(self, doc_id: str, collection: str = "default", *, chunk_ids: list[str] | None = None) -> bool:
        """按准确的最终 Chunk ID 删除；未提供 ID 时兼容上游的前缀入口。"""
        if not doc_id:
            raise ValueError("Document identifier cannot be empty")
        if not self.load(collection):
            return False
        identifiers = set(chunk_ids) if chunk_ids is not None else {cid for cid in self._documents if cid.startswith(doc_id)}
        remaining = [stat for cid, stat in self._documents.items() if cid not in identifiers]
        if len(remaining) == len(self._documents):
            return False
        if remaining:
            self.build(remaining, collection)
        else:
            self._metadata = {"num_docs": 0, "avg_doc_length": 0, "total_terms": 0, "collection": collection}
            self._index, self._documents = {}, {}
            self._save(collection, {"metadata": self._metadata, "index": {}, "documents": {}})
        return True

    def _save(self, collection: str, data: dict[str, Any]) -> None:
        """单写入者下原子替换；不宣称多进程增量写入安全。"""
        path = self._get_index_path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
