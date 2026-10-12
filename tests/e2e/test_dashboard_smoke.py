"""G1–G6/I2：从 Streamlit 真入口检查空库与离线真实数据页面。"""

import asyncio
from io import BytesIO
import os
from pathlib import Path
import socket
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from src.core.settings import load_settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubConfig, QueryKnowledgeHubTool
from src.observability.dashboard.services.data_service import DataService


pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"
STARTUP_PATH = REPO_ROOT / "scripts" / "start_dashboard.py"
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)
SAFE_ENV_NAMES = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
PAGES = (
    ("pages/overview.py", "系统总览"),
    ("pages/data_browser.py", "数据浏览"),
    ("pages/ingestion_manager.py", "摄取管理"),
    ("pages/ingestion_traces.py", "摄取追踪"),
    ("pages/query_traces.py", "查询追踪"),
    ("pages/evaluation_panel.py", "评估面板"),
)


class OfflineEmbedding(BaseEmbedding):
    """本地确定性向量，仅替代模型服务。"""

    def embed(self, texts, trace=None, **kwargs):
        return [[1.0, float(sum(map(ord, text)) % 101 + 1)] for text in texts]

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
        raise AssertionError("Dashboard AppTest must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def _png_bytes() -> bytes:
    image = BytesIO()
    Image.new("RGB", (18, 14), (45, 125, 210)).save(image, format="PNG")
    return image.getvalue()


def _prepare_services(tmp_path: Path, monkeypatch):
    settings_path = _settings_file(tmp_path)
    data_dir = tmp_path / "dashboard-data"
    monkeypatch.setenv("RAG_DASHBOARD_CONFIG", str(settings_path))
    monkeypatch.setenv("RAG_DASHBOARD_DATA_DIR", str(data_dir))
    settings = load_settings(settings_path)
    service = DataService(settings, data_dir=data_dir, embedding_client=OfflineEmbedding())
    return settings, data_dir, service


def _assert_clean_page(at, page_path: str, title: str):
    at.switch_page(page_path).run(timeout=20)
    assert len(at.exception) == 0, f"{page_path}: {at.exception}"
    assert len(at.error) == 0, f"{page_path}: {at.error}"
    assert at.title[0].value == title


def _visible_text(at) -> str:
    values = []
    for selector in ("title", "header", "subheader", "caption", "text", "code", "markdown",
                     "info", "success", "warning", "error"):
        for item in getattr(at, selector, []):
            value = getattr(item, "value", None)
            if value is not None:
                values.append(str(value))
    for item in at.json:
        values.append(str(item.value))
    return "\n".join(values)


def test_app_pages_render_without_errors_empty_and_with_real_offline_data(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)
    settings, data_dir, service = _prepare_services(tmp_path, monkeypatch)

    from streamlit.testing.v1 import AppTest

    empty = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert len(empty.exception) == 0
    assert len(empty.error) == 0
    for page_path, title in PAGES:
        _assert_clean_page(empty, page_path, title)
        if title == "评估面板":
            assert "已加载 3 道题" in _visible_text(empty)
    assert not data_dir.exists()

    image_bytes = _png_bytes()
    document_text = (
        "# Tundra River Field Note\n\n"
        "Amber river map evidence remains searchable in this local dashboard smoke test.\n\n"
        "![Blue river map](river-map.png)\n"
    ).encode("utf-8")
    outcomes = service.ingest_uploads(
        [("tundra-note.md", document_text), ("river-map.png", image_bytes)],
        "notes",
    )
    assert len(outcomes) == 1 and outcomes[0]["result"]["success"]

    query = "amber river map"
    query_tool = QueryKnowledgeHubTool(
        settings,
        QueryKnowledgeHubConfig(default_top_k=5, max_top_k=20, default_collection="notes", enable_rerank=False),
        data_dir=data_dir,
        embedding_client=service.embedding_client,
    )
    asyncio.run(query_tool.execute(query, top_k=5, collection="notes"))

    populated = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert len(populated.exception) == 0
    assert len(populated.error) == 0
    for page_path, title in PAGES:
        _assert_clean_page(populated, page_path, title)
        page_text = _visible_text(populated)
        if title == "数据浏览":
            assert "Amber river map evidence" in page_text
            assert "tundra-note.md" in page_text
        elif title == "摄取追踪":
            assert "处理前" in page_text and "处理后" in page_text
            assert "Amber river map evidence" in page_text
        elif title == "查询追踪":
            assert "Dense 召回" in page_text
            assert "BM25 召回" in page_text
            assert query in page_text
            assert "Amber river map evidence" in page_text
        elif title == "评估面板":
            assert any(item.label.startswith("确认执行检索/评分并使用所选模型 API") and item.value is False
                       for item in populated.checkbox)

    # 删除路径来自当前真实文档选择；只验证临时库中的单次提示与即时刷新。
    _assert_clean_page(populated, "pages/ingestion_manager.py", "摄取管理")
    source_path = service.list_documents("notes")[0]["source_path"]
    confirmation_key = f"delete_confirmation_notes:{source_path}"
    populated.text_input(confirmation_key).set_value(source_path)
    next(button for button in populated.button if button.label == "确认删除该文档").click()
    populated.run(timeout=20)
    assert len(populated.exception) == 0
    assert len(populated.error) == 0
    assert sum("已删除" in item.value for item in populated.success) == 1
    assert any(item.value == "该集合暂无文档。" for item in populated.info)
    assert Path(source_path).is_file()

    populated.run(timeout=20)
    assert not any("已删除" in item.value for item in populated.success)


def test_start_dashboard_help_exits_without_writing_or_starting_a_server(tmp_path):
    environment = {name: os.environ[name] for name in SAFE_ENV_NAMES if name in os.environ}
    assert set(MODEL_KEYS).isdisjoint(environment)
    data_dir = tmp_path / "must-not-be-created"
    config_path = tmp_path / "offline-settings.yaml"
    result = subprocess.run(
        [sys.executable, str(STARTUP_PATH), "--help", "--config", str(config_path), "--data-dir", str(data_dir)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout and "--data-dir" in result.stdout and "--port" in result.stdout
    assert not data_dir.exists()
