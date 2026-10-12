"""G2：用真实本地索引验证 DocumentManager 的文档生命周期接口。"""

from contextlib import contextmanager
from pathlib import Path
import socket

from PIL import Image
import pytest
import yaml

from src.core.settings import load_settings
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.unit


class OfflineEmbedding(BaseEmbedding):
    """稳定二维向量，只替代外部 Embedding 服务。"""

    def embed(self, texts, trace=None, **kwargs):
        return [[1.0, float(len(text) % 11 + 1)] for text in texts]

    def get_dimension(self):
        return 2


def _settings_file(tmp_path: Path) -> Path:
    values = {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "offline", "dimensions": 2},
        "vector_store": {"provider": "chroma", "persist_directory": str(tmp_path / "unused"),
                         "collection_name": "default"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": False, "provider": "offline", "metrics": []},
        "observability": {"log_level": "ERROR", "trace_enabled": False,
                          "trace_file": str(tmp_path / "traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 180, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _deny_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("DocumentManager tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def _write_markdown(path: Path, body: str, image_name: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = f"\n![Local diagram]({image_name})\n" if image_name else ""
    path.write_text(f"# Offline lifecycle note\n\n{body}{image}", encoding="utf-8")
    if image_name:
        Image.new("RGB", (16, 12), (210, 45, 65)).save(path.parent / image_name, format="PNG")
    return path


def _ingest(settings, data_dir: Path, source: Path, collection="notes", *, force=False):
    pipeline = IngestionPipeline(settings, collection, force=force, embedding=OfflineEmbedding(),
                                 data_dir=data_dir, source_root=source.parent)
    try:
        result = pipeline.run(source)
    finally:
        pipeline.close()
    assert result.success, result.error
    return result


@contextmanager
def _manager(settings, data_dir: Path, collection="notes"):
    store = ChromaStore(settings, persist_directory=data_dir / "db" / "chroma",
                        collection_name=collection, create_if_missing=False)
    bm25 = BM25Indexer(str(data_dir / "db" / "bm25" / collection))
    images = ImageStorage(str(data_dir / "db" / "image_index.db"), str(data_dir / "images"))
    history = SQLiteIntegrityChecker(str(data_dir / "db" / "ingestion_history.db"))
    try:
        yield DocumentManager(store, bm25, images, history), store, bm25, images, history
    finally:
        store.client.close()
        images.close()
        history.close()


def test_document_manager_lists_real_chunks_images_and_collection_stats(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    source = _write_markdown(tmp_path / "sources" / "guide.md", "A genuine local passage for document browsing.", "figure.png")
    _ingest(settings, data_dir, source)

    with _manager(settings, data_dir) as (manager, _store, _bm25, _images, _history):
        documents = manager.list_documents("notes")
        assert len(documents) == 1
        document = documents[0]
        assert document.source_path == str(source.resolve())
        assert document.collection == "notes"
        assert document.chunk_count >= 1
        assert document.image_count == 1
        assert document.source_hash
        assert document.status == "success"

        detail = manager.get_document_detail(document.source_hash, source_path=document.source_path)
        assert detail is not None
        assert detail.chunk_ids
        assert detail.image_ids
        assert manager.get_document_detail("missing-document") is None

        stats = manager.get_collection_stats("notes")
        assert (stats.collection, stats.document_count, stats.chunk_count, stats.image_count) == (
            "notes", 1, document.chunk_count, 1,
        )


def test_delete_removes_all_path_versions_and_bm25_while_preserving_source(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    source = _write_markdown(tmp_path / "sources" / "changing.md", "Initial unique amber passage.", "figure.png")
    first = _ingest(settings, data_dir, source)
    _write_markdown(source, "Replacement unique cobalt passage.", "figure.png")
    second = _ingest(settings, data_dir, source, force=True)

    with _manager(settings, data_dir) as (manager, store, bm25, images, history):
        before = next(doc for doc in manager.list_documents("notes") if doc.source_path == str(source.resolve()))
        assert len(before.source_hashes) == 2
        assert before.chunk_count == first.chunk_count + second.chunk_count
        assert len(manager.get_document_detail(before.source_hash, source_path=before.source_path).chunk_ids) == before.chunk_count
        assert bm25.load("notes") is True
        assert bm25.query(["replacement"], top_k=20)

        result = manager.delete_document(before.source_path, "notes", before.source_hash)

        assert result.success
        assert result.chunks_deleted == before.chunk_count
        assert result.bm25_removed
        assert result.images_deleted == 2
        assert result.integrity_removed
        assert manager.list_documents("notes") == []
        assert store.collection.count() == 0
        assert history.list_processed("notes") == []
        assert bm25.load("notes") is True
        assert bm25.query(["replacement"], top_k=20) == []
        assert source.is_file()
        assert source.read_text(encoding="utf-8").find("Replacement unique cobalt passage") >= 0
        trashed = list((data_dir / "images" / ".trash").glob("*"))
        assert len(trashed) == 2
        assert all(path.is_file() for path in trashed)
        assert images.list_images(collection="notes") == []


def test_wrong_path_hash_or_collection_cannot_delete_an_index(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    source = _write_markdown(tmp_path / "sources" / "protected.md", "Protected searchable violet passage.")
    _ingest(settings, data_dir, source)

    with _manager(settings, data_dir) as (manager, store, _bm25, _images, _history):
        document = manager.list_documents("notes")[0]
        wrong_path = manager.delete_document(document.source_path + ".elsewhere", "notes")
        wrong_hash = manager.delete_document(document.source_path, "notes", "0" * 64)
        with pytest.raises(ValueError, match="another collection"):
            manager.delete_document(document.source_path, "other")

        assert not wrong_path.success and not wrong_hash.success
        assert store.collection.count() == document.chunk_count
        assert [item.source_path for item in manager.list_documents("notes")] == [document.source_path]


def test_partial_image_delete_keeps_history_and_can_be_retried(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    source = _write_markdown(tmp_path / "sources" / "retry.md", "Retryable local image evidence.", "figure.png")
    _ingest(settings, data_dir, source)

    with _manager(settings, data_dir) as (manager, _store, _bm25, images, history):
        document = manager.list_documents("notes")[0]
        original_delete = images.delete_image

        with monkeypatch.context() as patcher:
            def fail_image_delete(image_id, *, collection):
                raise OSError("simulated image store outage")

            patcher.setattr(images, "delete_image", fail_image_delete)
            failed = manager.delete_document(document.source_path, "notes", document.source_hash)

        assert not failed.success
        assert any(error.startswith("Images: OSError") for error in failed.errors)
        assert history.list_processed("notes")
        assert manager.list_documents("notes")[0].status == "success"

        retried = manager.delete_document(document.source_path, "notes", document.source_hash)
        assert retried.success
        assert history.list_processed("notes") == []
        assert manager.list_documents("notes") == []
        assert len(list((data_dir / "images" / ".trash").glob("*"))) == 1
        assert callable(original_delete)


def test_shared_content_hash_keeps_another_sources_registered_image(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    body = "Identical bytes deliberately create a shared content hash."
    source_a = _write_markdown(tmp_path / "source-a" / "same.md", body, "shared.png")
    source_b = _write_markdown(tmp_path / "source-b" / "same.md", body, "shared.png")
    _ingest(settings, data_dir, source_a)
    _ingest(settings, data_dir, source_b, force=True)

    with _manager(settings, data_dir) as (manager, _store, _bm25, images, _history):
        shared_hash = manager.list_documents("notes")[0].source_hash
        other_image_source = tmp_path / "source-b" / "another-managed-image.png"
        Image.new("RGB", (10, 10), (20, 170, 80)).save(other_image_source, format="PNG")
        images.save_image("other-source-shared-image", other_image_source, collection="notes", doc_hash=shared_hash)
        image_ids_before = {image["image_id"] for image in images.list_images(collection="notes")}
        document_a = next(doc for doc in manager.list_documents("notes") if doc.source_path == str(source_a.resolve()))

        result = manager.delete_document(document_a.source_path, "notes", document_a.source_hash)

        assert result.success
        assert result.images_deleted == 0
        assert "other-source-shared-image" in {image["image_id"] for image in images.list_images(collection="notes")}
        assert {image["image_id"] for image in images.list_images(collection="notes")} == image_ids_before
        assert other_image_source.is_file()
        remaining = manager.list_documents("notes")
        assert [doc.source_path for doc in remaining] == [str(source_b.resolve())]
