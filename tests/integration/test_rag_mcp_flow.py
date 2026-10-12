"""E3-E6：真实离线摄取、持久化与官方 MCP stdio 完整业务流。"""

import asyncio
import base64
import io
import os
from pathlib import Path
import socket
import sys
import textwrap

import anyio
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from PIL import Image
import pytest
import yaml

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.chroma_store import ChromaStore


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY_NAMES = {
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
}
SAFE_INHERITED_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")


class OfflineEmbedding(BaseEmbedding):
    """通过公开 Embedding seam 提供固定维度向量，不调用任何模型服务。"""

    def __init__(self):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def get_dimension(self):
        return 2


def _settings_file(tmp_path: Path) -> Path:
    """使用离线 provider 配置，摄取与 MCP 存储均显式重定向到 tmp_path。"""
    settings = {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "offline", "dimensions": 2},
        "vector_store": {
            "provider": "chroma", "persist_directory": str(tmp_path / "unused-default"),
            "collection_name": "default",
        },
        "retrieval": {"dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 20, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 10},
        "evaluation": {"enabled": False, "provider": "offline", "metrics": []},
        "observability": {
            "log_level": "ERROR", "trace_enabled": False,
            "trace_file": str(tmp_path / "unused-trace.jsonl"), "structured_logging": False,
        },
        "ingestion": {
            "chunk_size": 220, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
            "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False},
        },
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 1024},
    }
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return config_path


def _save_image(path: Path, image_format: str, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (18, 14), color).save(path, format=image_format)


def _make_pdf(path: Path, image_path: Path) -> None:
    """生成带可抽取正文和真实内嵌 PNG 的多页 PDF。"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    canvas = Canvas(str(path), pagesize=letter)
    canvas.drawString(48, 740, "Zircon PDF chapter one describes the alpha retrieval record.")
    canvas.drawImage(ImageReader(str(image_path)), 48, 620, width=72, height=56)
    canvas.showPage()
    canvas.drawString(48, 740, "Zircon PDF chapter two preserves a second searchable passage.")
    canvas.drawString(48, 710, "The final paragraph contains independent archive evidence for retrieval.")
    canvas.save()


def _ingest(settings, data_dir: Path, source_path: Path, collection: str, source_root: Path):
    """通过生产 IngestionPipeline 完成真实本地解析、切分、编码和存储。"""
    pipeline = IngestionPipeline(
        settings, collection, embedding=OfflineEmbedding(), data_dir=data_dir, source_root=source_root,
    )
    try:
        result = pipeline.run(source_path)
    finally:
        pipeline.close()
    assert result.success is True, result.error
    assert result.doc_id is not None
    assert result.chunk_count >= 1
    return result


def _write_child_runtime(tmp_path: Path) -> tuple[Path, Path]:
    """建立离线网络守卫和只使用注入 FakeEmbedding 的 stdio 子进程入口。"""
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import socket

            def _deny_network(*args, **kwargs):
                raise AssertionError("network access is disabled in this test subprocess")

            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            socket.create_connection = _deny_network
            """
        ).lstrip(),
        encoding="utf-8",
    )
    wrapper = tmp_path / "offline_mcp_server.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            import sys
            from src.core.settings import load_settings
            from src.libs.embedding.base_embedding import BaseEmbedding
            from src.mcp_server.server import create_default_protocol_handler, run_stdio_server

            class OfflineEmbedding(BaseEmbedding):
                def embed(self, texts, trace=None, **kwargs):
                    with open(sys.argv[3], "a", encoding="utf-8") as marker:
                        marker.write("\\n".join(texts) + "\\n")
                    if "backend failure probe" in texts:
                        raise RuntimeError("EMBEDDING_API_KEY_SENTINEL")
                    return [[1.0, 0.0] for _ in texts]

                def get_dimension(self):
                    return 2

            settings = load_settings(sys.argv[1])
            handler = create_default_protocol_handler(
                settings, data_dir=sys.argv[2], embedding_client=OfflineEmbedding(),
            )
            raise SystemExit(run_stdio_server(handler, log_level="ERROR"))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return wrapper, tmp_path / "embedding-calls.txt"


def _child_environment(tmp_path: Path) -> dict[str, str]:
    """子进程只继承启动所需环境，不读取用户凭据或工作目录配置。"""
    environment = {key: os.environ[key] for key in SAFE_INHERITED_ENV if key in os.environ}
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    assert MODEL_KEY_NAMES.isdisjoint(environment)
    return environment


def _text(result) -> str:
    return "\n".join(item.text for item in result.content if isinstance(item, types.TextContent))


def test_ingested_documents_are_searchable_and_summarizable_through_real_mcp_stdio(tmp_path, monkeypatch):
    """临时 MD/PDF 经摄取后由官方客户端查询、引用、回查摘要和读取关联图片。"""
    monkeypatch.chdir(tmp_path)

    def reject_network(*args, **kwargs):
        raise AssertionError("完整离线 RAG-MCP 验收不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    for name in MODEL_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)

    config_path = _settings_file(tmp_path)
    settings = load_settings(config_path)
    data_dir = tmp_path / "isolated-data"
    source_root = tmp_path / "documents"
    source_root.mkdir()

    png_path, jpeg_path, webp_path = source_root / "signal.png", source_root / "signal.jpg", source_root / "signal.webp"
    pdf_image_path = source_root / "pdf-figure.png"
    _save_image(png_path, "PNG", (220, 35, 45))
    _save_image(jpeg_path, "JPEG", (40, 210, 70))
    _save_image(webp_path, "WEBP", (35, 75, 225))
    _save_image(pdf_image_path, "PNG", (220, 180, 20))

    markdown_path = source_root / "alpha-handbook.md"
    markdown_path.write_text(
        """---
title: Alpha Markdown Handbook
tags: [alpha, offline]
---
# Alpha Markdown Handbook

Aurora lantern markdown evidence guides the alpha retrieval workflow.
![Red signal](signal.png)

Quartz notes explain how the alpha index keeps searchable local passages.
![Green signal](signal.jpg)

The handbook records a separate constellation path for offline knowledge search.
![Blue signal](signal.webp)

This final section preserves a second multi-block passage for the handbook archive.
""",
        encoding="utf-8",
    )
    pdf_path = source_root / "alpha-archive.pdf"
    _make_pdf(pdf_path, pdf_image_path)
    beta_path = source_root / "beta-manual.md"
    beta_path.write_text(
        """# Beta Manual

Beta cobalt manual evidence belongs only to the beta collection.

The beta archive keeps an independent searchable record for local retrieval.

Another beta paragraph makes this collection contain multiple chunks too.
""",
        encoding="utf-8",
    )

    markdown_result = _ingest(settings, data_dir, markdown_path, "alpha", source_root)
    pdf_result = _ingest(settings, data_dir, pdf_path, "alpha", source_root)
    beta_result = _ingest(settings, data_dir, beta_path, "beta", source_root)
    assert markdown_result.chunk_count >= 2
    assert markdown_result.image_count == 3
    assert pdf_result.image_count >= 1
    assert beta_result.chunk_count >= 2

    empty_store = ChromaStore(
        settings, persist_directory=data_dir / "db" / "chroma", collection_name="empty",
    )
    try:
        assert empty_store.collection.count() == 0
    finally:
        empty_store.client.close()

    wrapper, embedding_marker = _write_child_runtime(tmp_path)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(wrapper), str(config_path), str(data_dir), str(embedding_marker)],
        env=_child_environment(tmp_path),
        cwd=tmp_path,
    )
    captured_stderr = tmp_path / "mcp-server.stderr"

    async def exercise_protocol():
        with anyio.fail_after(45):
            with captured_stderr.open("w", encoding="utf-8") as stderr_log:
                async with stdio_client(parameters, errlog=stderr_log) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        initialized = await session.initialize()
                        assert initialized.serverInfo.name == "personal-knowledge-mcp"
                        assert initialized.serverInfo.version == "0.1.0"
                        listing = await session.list_tools()
                        assert {tool.name for tool in listing.tools} == {
                            "query_knowledge_hub", "list_collections", "get_document_summary",
                        }

                        collections = await session.call_tool("list_collections", {})
                        assert collections.isError is False
                        counts = {item["name"]: item["count"] for item in collections.structuredContent["collections"]}
                        assert counts == {
                            "alpha": markdown_result.chunk_count + pdf_result.chunk_count,
                            "beta": beta_result.chunk_count,
                            "empty": 0,
                        }
                        without_stats = await session.call_tool("list_collections", {"include_stats": False})
                        assert without_stats.isError is False
                        assert all("count" not in item for item in without_stats.structuredContent["collections"])

                        alpha_query = await session.call_tool(
                            "query_knowledge_hub", {"query": "aurora lantern markdown", "top_k": 20, "collection": "alpha"},
                        )
                        assert alpha_query.isError is False
                        citations = alpha_query.structuredContent["citations"]
                        markdown_citation = next(
                            citation for citation in citations if citation["source"] == str(markdown_path.resolve())
                        )
                        assert markdown_citation["metadata"]["source_ref"] == "doc_" + markdown_result.doc_id[:16]
                        assert markdown_citation["metadata"]["doc_hash"] == markdown_result.doc_id
                        assert markdown_citation["chunk_id"] in markdown_result.vector_ids
                        assert all(citation["source"] != str(beta_path.resolve()) for citation in citations)
                        assert "### [1] 结果 1" in _text(alpha_query)
                        assert "分数：" in _text(alpha_query)
                        assert "百分比" not in _text(alpha_query)
                        assert "Aurora lantern markdown evidence" in _text(alpha_query)
                        image_blocks = [item for item in alpha_query.content if isinstance(item, types.ImageContent)]
                        assert image_blocks
                        assert alpha_query.structuredContent["has_images"] is True
                        assert {item.mimeType for item in image_blocks} >= {"image/png", "image/jpeg", "image/webp"}
                        for image in image_blocks:
                            raw = base64.b64decode(image.data)
                            with Image.open(io.BytesIO(raw)) as decoded:
                                expected_mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[decoded.format]
                            assert image.mimeType == expected_mime

                        summary_by_source = await session.call_tool(
                            "get_document_summary",
                            {"doc_id": markdown_citation["metadata"]["source_ref"], "collection": "alpha"},
                        )
                        summary = summary_by_source.structuredContent
                        assert summary_by_source.isError is False
                        assert summary["title"] == "Alpha Markdown Handbook"
                        assert summary["source_path"] == str(markdown_path.resolve())
                        assert summary["chunk_count"] == markdown_result.chunk_count
                        assert summary["summary"].strip()
                        assert summary["tags"]
                        assert "MARKDOWN" in summary["tags"]
                        assert summary["metadata"]["doc_hash"] == markdown_result.doc_id
                        assert summary["metadata"]["summary_kind"] in {"stored_chunk_summary", "excerpt"}

                        summary_by_full_hash = await session.call_tool(
                            "get_document_summary", {"doc_id": markdown_result.doc_id, "collection": "alpha"},
                        )
                        summary_by_short_hash = await session.call_tool(
                            "get_document_summary", {"doc_id": markdown_result.doc_id[:16], "collection": "alpha"},
                        )
                        assert summary_by_full_hash.isError is False
                        assert summary_by_short_hash.isError is False
                        assert summary_by_full_hash.structuredContent["chunk_count"] == markdown_result.chunk_count
                        assert summary_by_short_hash.structuredContent["metadata"]["doc_hash"] == markdown_result.doc_id

                        pdf_query = await session.call_tool(
                            "query_knowledge_hub", {"query": "zircon chapter archive", "top_k": 20, "collection": "alpha"},
                        )
                        pdf_citation = next(
                            citation for citation in pdf_query.structuredContent["citations"]
                            if citation["metadata"].get("doc_hash") == pdf_result.doc_id
                        )
                        assert pdf_citation["source"] == str(pdf_path.resolve())
                        assert pdf_citation["chunk_id"] in pdf_result.vector_ids

                        # 通过临时 BM25 工件和外部 Embedding fake 制造双路故障，验证 MCP 错误脱敏。
                        bm25_path = data_dir / "db" / "bm25" / "alpha" / "alpha_bm25.json"
                        bm25_path.write_text("不是有效的 BM25 JSON", encoding="utf-8")
                        backend_failure = await session.call_tool(
                            "query_knowledge_hub", {"query": "backend failure probe", "collection": "alpha"},
                        )
                        assert backend_failure.isError is True
                        assert "RuntimeError" in _text(backend_failure)
                        assert "EMBEDDING_API_KEY_SENTINEL" not in _text(backend_failure)

                        beta_query = await session.call_tool(
                            "query_knowledge_hub", {"query": "beta cobalt manual", "top_k": 20, "collection": "beta"},
                        )
                        assert beta_query.isError is False
                        assert beta_query.structuredContent["citations"]
                        assert all(item["source"] == str(beta_path.resolve()) for item in beta_query.structuredContent["citations"])

                        empty_query = await session.call_tool(
                            "query_knowledge_hub", {"query": "empty collection probe", "collection": "empty"},
                        )
                        missing_collection = await session.call_tool(
                            "query_knowledge_hub", {"query": "missing collection probe", "collection": "not-created"},
                        )
                        missing_summary = await session.call_tool(
                            "get_document_summary", {"doc_id": "doc_ffffffffffffffff", "collection": "alpha"},
                        )
                        assert empty_query.isError is False
                        assert empty_query.structuredContent["isEmpty"] is True
                        assert missing_collection.isError is False
                        assert missing_collection.structuredContent["isEmpty"] is True
                        assert missing_summary.isError is True
                        assert "doc_ffffffffffffffff" not in _text(missing_summary)

                        invalid_schema = await session.call_tool(
                            "query_knowledge_hub", {"query": "valid", "top_k": 21, "collection": "alpha"},
                        )
                        unknown_tool = await session.call_tool("not_registered", {})
                        assert invalid_schema.isError is True
                        assert unknown_tool.isError is True

                        recovery = await session.call_tool("list_collections", {"include_stats": False})
                        assert recovery.isError is False

    asyncio.run(exercise_protocol())

    stderr_text = captured_stderr.read_text(encoding="utf-8")
    assert "Traceback" not in stderr_text
    assert "network access is disabled" not in stderr_text
    assert "API_KEY" not in stderr_text
    assert not (tmp_path / "unused-default").exists()
    assert not (tmp_path / "unused-trace.jsonl").exists()
    embedding_calls = embedding_marker.read_text(encoding="utf-8").splitlines()
    assert embedding_calls == [
        "aurora lantern markdown", "zircon chapter archive", "backend failure probe", "beta cobalt manual",
    ]
    assert "EMBEDDING_API_KEY_SENTINEL" not in stderr_text
