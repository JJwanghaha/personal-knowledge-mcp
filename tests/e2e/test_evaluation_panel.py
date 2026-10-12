"""H4：从 Streamlit 真入口检查评估页空态、确认门槛、逐题结果和历史报告。"""

from hashlib import sha256
import json
from pathlib import Path
import socket
import uuid

import jieba
import pytest
import yaml

from src.core.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


pytestmark = pytest.mark.e2e
REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "evaluation"
FIXTURE_NAMES = ("01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md")
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


class PanelEmbedding(BaseEmbedding):
    """通过公开 Embedding provider seam 提供确定性本地向量。"""

    def __init__(self, settings=None):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        vectors = []
        for text in texts:
            self.calls.append(text)
            vector = [0.0] * self.get_dimension()
            for token in jieba.lcut(text.casefold()):
                if token.strip():
                    slot = int.from_bytes(sha256(token.encode("utf-8")).digest()[:4], "big") % len(vector)
                    vector[slot] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def get_dimension(self):
        return 128


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("评估页 AppTest 必须保持离线")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def settings_values(tmp_path, provider):
    return {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": provider, "model": "panel-fixture", "dimensions": 128},
        "vector_store": {"provider": "chroma", "persist_directory": str(tmp_path / "unused-config-persist"),
                         "collection_name": "panel-eval"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": False, "provider": "custom", "metrics": ["hit_rate", "mrr"]},
        "observability": {"log_level": "ERROR", "trace_enabled": False,
                          "trace_file": str(tmp_path / "unused-traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 1200, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    }


def visible_text(at):
    values = []
    for selector in ("title", "header", "subheader", "caption", "text", "code", "markdown",
                     "info", "success", "warning", "error"):
        for element in getattr(at, selector, []):
            value = getattr(element, "value", None)
            if value is not None:
                values.append(str(value))
    for element in at.json:
        values.append(str(element.value))
    for element in at.dataframe:
        values.append(str(element.value))
    return "\n".join(values)


def click_button(at, label):
    next(button for button in at.button if button.label == label).click()


def test_evaluation_page_empty_state_confirmation_and_persisted_offline_run(tmp_path, monkeypatch):
    provider = f"panel-offline-{uuid.uuid4().hex}"
    EmbeddingFactory.register_provider(provider, PanelEmbedding)
    values = settings_values(tmp_path, provider)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
    data_root = tmp_path / "evaluation-data"
    monkeypatch.setenv("RAG_DASHBOARD_CONFIG", str(settings_path))
    monkeypatch.setenv("RAG_DASHBOARD_DATA_DIR", str(data_root))

    from streamlit.testing.v1 import AppTest

    empty = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    empty.switch_page("pages/evaluation_panel.py").run(timeout=30)
    assert len(empty.exception) == 0
    assert len(empty.error) == 0
    text = visible_text(empty)
    assert "评估面板" in text
    assert "已加载 3 道题" in text
    assert any(item.label == "预览固定试卷" for item in empty.expander)
    consent = next(item for item in empty.checkbox if item.key == "confirm_evaluation")
    assert consent.value is False
    click_button(empty, "运行固定测试集")
    empty.run(timeout=30)
    assert any("未确认模型调用，未运行评估" in item.value for item in empty.warning)
    assert "暂无历史报告" in visible_text(empty)
    assert not data_root.exists()

    click_button(empty, "生成未审核草稿")
    empty.run(timeout=30)
    assert any("请先选择资料并确认发送给模型" in item.value for item in empty.warning)
    assert not data_root.exists()

    settings = Settings.from_dict(values)
    pipeline = IngestionPipeline(settings, "panel-eval", embedding=PanelEmbedding(), data_dir=data_root,
                                 source_root=FIXTURE_ROOT)
    try:
        for name in FIXTURE_NAMES:
            result = pipeline.run(FIXTURE_ROOT / name)
            assert result.success, result.error
    finally:
        pipeline.close()

    populated = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    populated.switch_page("pages/evaluation_panel.py").run(timeout=30)
    assert len(populated.exception) == 0
    assert len(populated.error) == 0
    assert "evaluation-flow" not in visible_text(populated)
    assert "暂无历史报告" in visible_text(populated)

    source_picker = next(item for item in populated.multiselect if item.label.startswith("出题资料"))
    source_picker.set_value([str((FIXTURE_ROOT / FIXTURE_NAMES[0]).resolve())])
    populated.run(timeout=30)
    click_button(populated, "生成未审核草稿")
    populated.run(timeout=30)
    assert any("请先选择资料并确认发送给模型" in item.value for item in populated.warning)
    assert not (data_root / "evaluations" / "drafts").exists()

    next(item for item in populated.number_input if item.label == "TopK").set_value(1)
    next(item for item in populated.checkbox if item.key == "confirm_evaluation").set_value(True)
    click_button(populated, "运行固定测试集")
    populated.run(timeout=45)

    assert len(populated.exception) == 0
    assert len(populated.error) == 0
    rendered = visible_text(populated)
    report_labels = [item.label for item in populated.expander]
    assert all(any(label.startswith(f"{case_id} · success") for label in report_labels)
               for case_id in ("ingestion-force", "retrieval-rrf", "mcp-stdio"))
    assert "评分题数" in rendered and "总题数" in rendered
    assert "语料指纹" in rendered
    assert "不同试卷或语料指纹不能直接比较分数" in rendered
    assert "暂无历史报告" not in rendered

    reports = list((data_root / "evaluations" / "reports").glob("*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["query_count"] == 3
    assert saved["collection"] == "panel-eval"
    assert saved["top_k"] == 1
    assert saved["metric_counts"] == {"hit_rate": 3, "mrr": 3}
    assert saved["configuration"]["corpus_fingerprint"]
    assert saved["test_set_sha256"] == sha256(
        (REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json").read_bytes()
    ).hexdigest()
    assert all(len(row["retrieved_chunk_ids"]) <= 1 for row in saved["query_results"])
