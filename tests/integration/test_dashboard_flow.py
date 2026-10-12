"""G2–G6：真实离线上传、双索引查询和 Dashboard Trace 数据闭环。"""

import asyncio
from io import BytesIO
from pathlib import Path
import socket

from PIL import Image
import pytest
import yaml

from src.core.settings import load_settings
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.embedding.base_embedding import BaseEmbedding
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubConfig, QueryKnowledgeHubTool
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.trace_service import TraceService


pytestmark = pytest.mark.integration


class OfflineEmbedding(BaseEmbedding):
    """只替换 Embedding 外部边界；索引、查询和 Trace 均使用真实实现。"""

    def __init__(self):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.extend(texts)
        return [[1.0, float(sum(map(ord, text)) % 97 + 1)] for text in texts]

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
        "observability": {"log_level": "ERROR", "trace_enabled": True,
                          "trace_file": str(tmp_path / "logs" / "traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 240, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _deny_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Dashboard flow must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def _png_bytes(color=(230, 90, 35)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 18), color).save(output, format="PNG")
    return output.getvalue()


def _pdf_bytes(image_bytes: bytes) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    output = BytesIO()
    pdf = Canvas(output, pagesize=letter)
    pdf.drawString(54, 740, "Saffron archive PDF preserves offline retrieval evidence.")
    pdf.drawImage(ImageReader(BytesIO(image_bytes)), 54, 620, width=72, height=54)
    pdf.save()
    return output.getvalue()


def test_uploaded_markdown_pdf_and_image_generate_dashboard_ingestion_and_query_traces(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    settings = load_settings(_settings_file(tmp_path))
    data_dir = tmp_path / "dashboard-data"
    embedding = OfflineEmbedding()
    service = DataService(settings, data_dir=data_dir, embedding_client=embedding)
    image_bytes = _png_bytes()
    markdown = (
        "# Offline Aurora Guide\n\n"
        "Quartz lantern passages make the Aurora guide searchable from the local dashboard.\n\n"
        "![Dashboard figure](figure.png)\n"
    ).encode("utf-8")
    pdf = _pdf_bytes(image_bytes)
    progress = []

    outcomes = service.ingest_uploads(
        [("aurora-guide.md", markdown), ("saffron-archive.pdf", pdf), ("figure.png", image_bytes)],
        "notes",
        on_progress=lambda stage, current, total: progress.append((stage, current, total)),
    )

    assert len(outcomes) == 2
    assert all(outcome["result"]["success"] for outcome in outcomes)
    assert [step for _, step, _ in progress] == [1, 2, 3, 4, 5, 6] * 2
    assert all(total == 6 for _, _, total in progress)
    assert [stage for stage, _, _ in progress] == [
        "integrity", "load", "split", "transform", "embed", "upsert",
    ] * 2
    assert len(embedding.calls) >= 2

    documents = service.list_documents("notes")
    assert {Path(item["source_path"]).name for item in documents} == {"aurora-guide.md", "saffron-archive.pdf"}
    markdown_doc = next(item for item in documents if item["source_path"].endswith("aurora-guide.md"))
    pdf_doc = next(item for item in documents if item["source_path"].endswith("saffron-archive.pdf"))
    markdown_chunks = service.get_chunks(markdown_doc["source_hash"], "notes", source_path=markdown_doc["source_path"])
    pdf_chunks = service.get_chunks(pdf_doc["source_hash"], "notes", source_path=pdf_doc["source_path"])
    assert any("Quartz lantern passages" in chunk["text"] for chunk in markdown_chunks)
    assert any("Saffron archive PDF" in chunk["text"] for chunk in pdf_chunks)
    assert markdown_doc["image_count"] == 1
    assert Path(markdown_doc["source_path"]).is_file()
    assert Path(pdf_doc["source_path"]).is_file()
    assert ImageStorage.list_existing(data_dir / "db" / "image_index.db", "notes")

    traces = TraceService(settings.observability.trace_file)
    ingestion_traces = traces.list_traces("ingestion", limit=10)
    assert len(ingestion_traces) == 2
    assert all(trace["metadata"]["source"] == "dashboard" for trace in ingestion_traces)
    assert all(trace["metadata"]["status"] == "success" for trace in ingestion_traces)
    assert all({"load", "split", "transform", "embed", "upsert"} <= {
        stage["stage"] for stage in trace["stages"]
    } for trace in ingestion_traces)
    assert all(timing["elapsed_ms"] >= 0 for trace in ingestion_traces for timing in traces.get_stage_timings(trace))

    query_tool = QueryKnowledgeHubTool(
        settings,
        QueryKnowledgeHubConfig(default_top_k=5, max_top_k=20, default_collection="notes", enable_rerank=False),
        data_dir=data_dir,
        embedding_client=embedding,
    )
    asyncio.run(query_tool.execute("quartz lantern", top_k=5, collection="notes"))

    query_traces = traces.list_traces("query", limit=10, keyword="quartz lantern", collection="notes")
    assert len(query_traces) == 1
    query_trace = query_traces[0]
    assert query_trace["metadata"]["source"] == "mcp"
    stages = {stage["stage"]: stage["data"] for stage in query_trace["stages"]}
    assert {"query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank", "response"} <= set(stages)
    assert stages["dense_retrieval"]["status"] == "success"
    assert any("Quartz lantern passages" in item.get("text", "")
               for item in stages["dense_retrieval"].get("chunks", []))
