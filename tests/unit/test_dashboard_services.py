"""G1/G2/G5：验证后台服务配置、离线浏览、上传边界与 Trace 读取。"""

import json
from pathlib import Path
import socket

from PIL import Image
import pytest
import yaml

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.chroma_store import ChromaStore
from src.observability.dashboard.services.config_service import ConfigService
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.trace_service import TraceService
from src.ingestion.storage.image_storage import ImageStorage


pytestmark = pytest.mark.unit


MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


class OfflineEmbedding(BaseEmbedding):
    def embed(self, texts, trace=None, **kwargs):
        return [[1.0, float(len(text) % 13 + 1)] for text in texts]

    def get_dimension(self):
        return 2


def _settings_file(tmp_path: Path, *, trace_enabled=False, trace_file=None, persist=None) -> Path:
    values = {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "offline", "dimensions": 2},
        "vector_store": {"provider": "chroma", "persist_directory": str(persist or tmp_path / "unused"),
                         "collection_name": "default"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": False, "provider": "offline", "metrics": []},
        "observability": {"log_level": "ERROR", "trace_enabled": trace_enabled,
                          "trace_file": str(trace_file or tmp_path / "logs" / "traces.jsonl"),
                          "structured_logging": False},
        "ingestion": {"chunk_size": 180, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _deny_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Dashboard service tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def _write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_read_only_browsing_missing_database_does_not_create_storage(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "not-created"
    service = DataService(settings, data_dir=data_dir)

    assert service.list_collections() == []
    assert service.list_documents("default") == []
    assert service.get_document_detail("unknown", "default") is None
    assert service.get_collection_stats("default") == {
        "collection": "default", "document_count": 0, "chunk_count": 0, "image_count": 0,
    }
    assert service.get_images("unknown-hash", "default") == []
    assert not data_dir.exists()


def test_unknown_delete_path_and_hash_return_before_opening_writable_sqlite(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "isolated-data"
    source = _write_markdown(tmp_path / "source.md", "A record used to test safe delete selection.")
    service = DataService(settings, data_dir=data_dir, embedding_client=OfflineEmbedding())
    outcome = service.ingest_file(source, "notes")
    assert outcome["result"]["success"]
    document = service.list_documents("notes")[0]

    def writable_open_is_a_failure(*args, **kwargs):
        raise AssertionError("an unknown delete target must not open writable SQLite stores")

    monkeypatch.setattr(ImageStorage, "__init__", writable_open_is_a_failure)
    monkeypatch.setattr(SQLiteIntegrityChecker, "__init__", writable_open_is_a_failure)

    missing_path = service.delete_document(str(tmp_path / "not-selected.md"), "notes")
    wrong_hash = service.delete_document(document["source_path"], "notes", "0" * 64)

    assert not missing_path.success
    assert not wrong_hash.success
    assert service.list_documents("notes")[0]["source_path"] == document["source_path"]


def test_upload_validates_the_whole_batch_overwrite_backup_and_collection_symlink(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "uploads-data"
    service = DataService(settings, data_dir=data_dir, embedding_client=OfflineEmbedding())

    with pytest.raises(ValueError, match="supported filenames"):
        service.ingest_uploads([("safe.md", b"# safe"), ("../escape.md", b"# escape")], "notes")
    assert not (data_dir / "uploads").exists()

    original = b"# Upload record\n\nFirst version for offline storage.\n"
    replacement = b"# Upload record\n\nSecond version for offline storage.\n"
    first = service.ingest_uploads([("upload.md", original)], "notes")
    assert first[0]["result"]["success"]
    target = data_dir / "uploads" / "notes" / "upload.md"
    assert target.read_bytes() == original

    with pytest.raises(FileExistsError, match="already exists"):
        service.ingest_uploads([("upload.md", replacement)], "notes")
    assert target.read_bytes() == original

    overwritten = service.ingest_uploads([("upload.md", replacement)], "notes", overwrite=True)
    assert overwritten[0]["result"]["success"]
    assert target.read_bytes() == replacement
    backups = list((data_dir / "uploads" / ".history").glob("*_upload.md"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original

    external = tmp_path / "outside"
    external.mkdir()
    (data_dir / "uploads" / "linked").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes managed data root"):
        service.ingest_uploads([("safe.md", b"# outside")], "linked")
    assert list(external.iterdir()) == []


def test_ingest_without_data_dir_preserves_custom_chroma_persist_directory(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    import src.core.settings as settings_module

    monkeypatch.setattr(settings_module, "REPO_ROOT", tmp_path)
    custom_chroma = tmp_path / "custom-chroma-store"
    settings = load_settings(_settings_file(tmp_path, persist=custom_chroma))
    source = _write_markdown(tmp_path / "source.md", "A custom Chroma path must remain selected.")
    service = DataService(settings, embedding_client=OfflineEmbedding())

    result = service.ingest_file(source)

    assert result["result"]["success"]
    assert (custom_chroma / "chroma.sqlite3").is_file()
    assert not (tmp_path / "data" / "db" / "chroma" / "chroma.sqlite3").exists()
    assert service.list_collections() == ["default"]
    assert service.list_documents("default")[0]["source_path"] == str(source.resolve())


def test_trace_service_filters_bad_lines_orders_records_and_sanitizes_stage_data(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    ingestion = {
        "trace_id": "ing-new", "trace_type": "ingestion", "started_at": "2026-10-10T12:00:00Z",
        "metadata": {"collection": "notes", "source_path": "guide.md"},
        "stages": [
            {"stage": "load", "elapsed_ms": 2.75, "data": {"method": "markdown"}},
            {"stage": "bad-bool", "elapsed_ms": True, "data": "not-an-object"},
            {"stage": "bad-negative", "elapsed_ms": -3, "data": None},
            {"stage": "bad-infinite", "elapsed_ms": float("inf"), "data": {"status": "ok"}},
            {"data": {"status": "ok"}},
            "not-a-stage-object",
        ],
    }
    query = {
        "trace_id": "query-old", "trace_type": "query", "started_at": "2026-10-09T12:00:00Z",
        "metadata": {"collection": "notes", "query": "violet keyword"}, "stages": [],
    }
    other_collection = {
        "trace_id": "ing-other", "trace_type": "ingestion", "started_at": "2026-10-08T12:00:00Z",
        "metadata": {"collection": "archive", "source_path": "other.md"}, "stages": [],
    }
    lines = [
        json.dumps(ingestion, ensure_ascii=False),
        json.dumps(query, ensure_ascii=False),
        json.dumps(other_collection, ensure_ascii=False),
        "ordinary logger line",
        json.dumps({"level": "INFO", "message": "not a trace"}),
        json.dumps(["wrong root type"]),
        json.dumps({**query, "trace_id": "unknown-kind", "trace_type": "worker"}),
        json.dumps({**query, "trace_id": "bad-stages", "stages": {"stage": "load"}}),
        json.dumps({**query, "trace_id": "bad-metadata", "metadata": []}),
    ]
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    service = TraceService(trace_path)

    selected = service.list_traces("ingestion", limit=10, keyword="guide", collection="notes")
    assert [trace["trace_id"] for trace in selected] == ["ing-new"]
    assert service.skipped_lines == 6
    assert service.get_trace("query-old")["metadata"]["query"] == "violet keyword"
    assert [trace["trace_id"] for trace in service.list_traces(limit=2)] == ["ing-new", "query-old"]
    assert [trace["trace_id"] for trace in service.list_traces("query", keyword="VIOLET")] == ["query-old"]

    timings = service.get_stage_timings(ingestion)
    assert [item["stage_name"] for item in timings] == ["load", "bad-bool", "bad-negative", "bad-infinite", "unknown"]
    assert timings[0] == {"stage_name": "load", "elapsed_ms": 2.75, "data": {"method": "markdown"}}
    assert all(item["elapsed_ms"] == 0 for item in timings[1:])
    assert all(isinstance(item["data"], dict) for item in timings)
    with pytest.raises(ValueError, match="positive"):
        service.list_traces(limit=0)


def test_config_cards_report_components_without_environment_credentials(tmp_path, monkeypatch):
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)
    settings_path = _settings_file(tmp_path)
    service = ConfigService(settings_path)

    cards = service.get_component_cards()

    assert {card.name for card in cards} >= {"生成模型", "向量模型", "向量存储", "混合检索", "重排", "Trace"}
    serialized = repr(cards)
    assert "API_KEY" not in serialized
    assert "OPENAI" not in serialized
