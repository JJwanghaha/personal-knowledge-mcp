"""H3：通过实际摄取、评估服务和 CLI 验证离线评估闭环。"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import jieba
import pytest
import yaml

from src.core.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.evaluation_service import EvaluationService
from src.observability.evaluation.eval_runner import load_test_set


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "evaluation"
FIXTURE_NAMES = ("01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md")
GOLDEN_TEST_SET = REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
)
SAFE_ENV_NAMES = ("PATH", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")


class FixtureEmbedding(BaseEmbedding):
    """以 jieba 词项哈希替代真实 Embedding API，保留真实检索与存储算法。"""

    def __init__(self, settings=None):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        vectors = []
        for text in texts:
            self.calls.append(text)
            vector = [0.0] * self.get_dimension()
            for token in jieba.lcut(text.casefold()):
                if token.strip():
                    slot = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % len(vector)
                    vector[slot] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def get_dimension(self):
        return 128


class DraftLLM(BaseLLM):
    """返回基于调用方提供片段构造的结构化候选题。"""

    def __init__(self, content):
        self.content = content
        self.prompts = []

    def chat(self, messages, trace=None, **kwargs):
        self.prompts.append("\n".join(message.content for message in messages))
        return ChatResponse(self.content, "offline-draft-model")


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("评估集成测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def settings_values(tmp_path, *, persist_directory=None):
    return {
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline-test", "model": "fixture-hash", "dimensions": 128},
        "vector_store": {"provider": "chroma", "persist_directory": str(
            persist_directory or tmp_path / "config-persist-must-stay-unused"),
            "collection_name": "evaluation-flow"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": False, "provider": "custom", "metrics": ["hit_rate", "mrr"]},
        "observability": {"log_level": "ERROR", "trace_enabled": False,
                          "trace_file": str(tmp_path / "unused-traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 1200, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    }


def settings_for(tmp_path, *, persist_directory=None):
    return Settings.from_dict(settings_values(tmp_path, persist_directory=persist_directory))


def ingest_public_fixtures(settings, data_root, collection="evaluation-flow"):
    pipeline = IngestionPipeline(settings, collection, embedding=FixtureEmbedding(), data_dir=data_root,
                                 source_root=FIXTURE_ROOT)
    outcomes = {}
    try:
        for name in FIXTURE_NAMES:
            result = pipeline.run(FIXTURE_ROOT / name)
            assert result.success, result.error
            outcomes[name] = result
    finally:
        pipeline.close()
    return outcomes


def write_settings(path, values):
    path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_test_set(path, cases, *, reviewed=True):
    path.write_text(json.dumps({"version": "1.0", "reviewed": reviewed, "test_cases": cases},
                               ensure_ascii=False), encoding="utf-8")
    return path


def cli_environment(site_dir):
    environment = {name: os.environ[name] for name in SAFE_ENV_NAMES if name in os.environ}
    environment["PYTHONPATH"] = os.pathsep.join((str(site_dir), str(REPO_ROOT)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["RAGAS_DO_NOT_TRACK"] = "true"
    assert set(MODEL_KEYS).isdisjoint(environment)
    return environment


def write_offline_provider(site_dir):
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        """
import hashlib
import socket
import jieba
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory

def _deny_network(*args, **kwargs):
    raise AssertionError("CLI verification must stay offline")

socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
socket.create_connection = _deny_network

class FixtureEmbedding(BaseEmbedding):
    def __init__(self, settings=None):
        pass
    def embed(self, texts, trace=None, **kwargs):
        result = []
        for text in texts:
            vector = [0.0] * self.get_dimension()
            for token in jieba.lcut(text.casefold()):
                if token.strip():
                    slot = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % len(vector)
                    vector[slot] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            result.append([value / norm for value in vector])
        return result
    def get_dimension(self):
        return 128

EmbeddingFactory.register_provider("offline-test", FixtureEmbedding)
""".lstrip(),
        encoding="utf-8",
    )


def run_cli(environment, *args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), *map(str, args)],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=60, check=False,
    )


def test_service_runs_real_retrieval_writes_private_history_and_closes_storage(tmp_path):
    values = settings_values(tmp_path)
    settings = Settings.from_dict(values)
    data_root = tmp_path / "isolated-data"
    ingested = ingest_public_fixtures(settings, data_root)
    service = EvaluationService(settings, data_dir=data_root, embedding_client=FixtureEmbedding())

    report = service.run(GOLDEN_TEST_SET, "evaluation-flow", backend="custom", top_k=1, no_rerank=True)

    assert len(report.query_results) == 3
    assert report.collection == "evaluation-flow" and report.top_k == 1
    assert report.test_set_sha256 == hashlib.sha256(GOLDEN_TEST_SET.read_bytes()).hexdigest()
    assert report.configuration["corpus_fingerprint"]
    assert report.configuration["generate_answers"] is False
    assert report.metric_counts == {"hit_rate": 3, "mrr": 3}
    assert all(len(row.retrieved_chunk_ids) <= 1 for row in report.query_results)
    assert sum(item.chunk_count for item in ingested.values()) == 3
    assert not service.reports_root.exists()

    report_path = service.save_report(report)
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored["collection"] == "evaluation-flow"
    assert stored["top_k"] == 1
    assert stored["test_set_sha256"] == report.test_set_sha256
    history = service.list_reports()
    assert len(history) == 1 and history[0]["path"] == str(report_path)

    assert (data_root / "db" / "chroma" / "chroma.sqlite3").is_file()
    assert not Path(values["vector_store"]["persist_directory"]).exists()
    reopened = VectorStoreFactory.create(service.settings, collection_name="evaluation-flow", create_if_missing=False)
    try:
        assert reopened.collection.count() == sum(item.chunk_count for item in ingested.values())
    finally:
        reopened.client.close()


def test_generate_draft_without_data_dir_uses_custom_persist_and_only_selected_indexed_chunks(tmp_path, monkeypatch):
    import src.core.settings as settings_module

    monkeypatch.setattr(settings_module, "REPO_ROOT", tmp_path)
    custom_persist = tmp_path / "custom-chroma"
    settings = settings_for(tmp_path, persist_directory=custom_persist)
    pipeline = IngestionPipeline(settings, "drafts", embedding=FixtureEmbedding(), source_root=FIXTURE_ROOT)
    try:
        for name in FIXTURE_NAMES[:2]:
            outcome = pipeline.run(FIXTURE_ROOT / name)
            assert outcome.success, outcome.error
    finally:
        pipeline.close()

    service = EvaluationService(settings, llm=None)
    indexed = DataService(settings).list_documents("drafts")
    assert len(indexed) == 2
    selected = indexed[0]
    chunks = DataService(settings).get_chunks(selected["source_hash"], "drafts", source_path=selected["source_path"])
    assert chunks
    quote = chunks[0]["text"][:32].strip()
    llm = DraftLLM(json.dumps({"test_cases": [{
        "query": "选中资料中这段事实对应什么问题？", "source": "source-1",
        "evidence_quote": quote, "reference_answer": "由人工审核的参考答案。",
    }]}, ensure_ascii=False))
    service.llm = llm

    draft = service.generate_draft("drafts", [selected["source_path"]], count=1)

    assert draft["reviewed"] is False
    assert draft["test_cases"][0]["expected_sources"] == [selected["source_path"]]
    assert draft["test_cases"][0]["expected_chunk_ids"] == [
        item["id"] for item in chunks if quote in item["text"]
    ]
    prompt = "\n".join(llm.prompts)
    other = next(item for item in indexed if item["source_path"] != selected["source_path"])
    other_chunks = DataService(settings).get_chunks(other["source_hash"], "drafts", source_path=other["source_path"])
    assert all(item["text"] not in prompt for item in other_chunks)
    assert (custom_persist / "chroma.sqlite3").is_file()
    assert not (tmp_path / "data" / "db" / "chroma" / "chroma.sqlite3").exists()
    assert not (tmp_path / "data" / "evaluations" / "reports").exists()

    saved = service.save_draft(draft)
    assert saved.is_file()
    assert saved.is_relative_to(tmp_path / "data" / "evaluations" / "drafts")
    with pytest.raises(ValueError, match="Review"):
        load_test_set(saved)


def test_cli_help_check_threshold_validation_exit_codes_json_and_no_overwrite(tmp_path, monkeypatch):
    site_dir = tmp_path / "offline-provider"
    write_offline_provider(site_dir)
    for name in MODEL_KEYS:
        monkeypatch.setenv(name, "DO_NOT_FORWARD_API_KEY_SENTINEL")
    environment = cli_environment(site_dir)
    assert set(MODEL_KEYS).isdisjoint(environment)

    help_result = run_cli(environment, "--help")
    assert help_result.returncode == 0
    assert "--check" in help_result.stdout and "--data-dir" in help_result.stdout

    config_path = write_settings(tmp_path / "settings.yaml", settings_values(tmp_path))
    data_root = tmp_path / "must-not-be-created"
    no_write_output = tmp_path / "must-not-be-written.json"
    check_result = run_cli(environment, "--check", "--config", config_path, "--test-set", GOLDEN_TEST_SET,
                           "--collection", "evaluation-flow", "--data-dir", data_root, "--output", no_write_output)
    assert check_result.returncode == 0
    assert "未调用模型或创建数据" in check_result.stdout
    assert not data_root.exists() and not no_write_output.exists()

    invalid_threshold = run_cli(environment, "--config", config_path, "--test-set", GOLDEN_TEST_SET,
                                 "--collection", "evaluation-flow", "--data-dir", data_root,
                                 "--min-hit-rate", "0")
    assert invalid_threshold.returncode == 2
    assert not data_root.exists()
    assert "DO_NOT_FORWARD_API_KEY_SENTINEL" not in invalid_threshold.stdout + invalid_threshold.stderr

    protected_output = tmp_path / "already-exists.json"
    protected_output.write_text("preserve this file", encoding="utf-8")
    protected_root = tmp_path / "must-not-open-existing-output"
    output_conflict = run_cli(environment, "--config", config_path, "--test-set", GOLDEN_TEST_SET,
                              "--collection", "evaluation-flow", "--data-dir", protected_root,
                              "--output", protected_output)
    assert output_conflict.returncode == 2
    assert protected_output.read_text(encoding="utf-8") == "preserve this file"
    assert not protected_root.exists()

    data_root = tmp_path / "cli-data"
    ingested = ingest_public_fixtures(Settings.from_dict(settings_values(tmp_path)), data_root)
    assert sum(item.chunk_count for item in ingested.values()) == 3
    successful_output = tmp_path / "success-report.json"
    success = run_cli(environment, "--config", config_path, "--test-set", GOLDEN_TEST_SET,
                       "--collection", "evaluation-flow", "--data-dir", data_root,
                       "--backend", "custom", "--top-k", "1", "--no-rerank",
                       "--output", successful_output, "--json")
    assert success.returncode == 0, success.stderr
    payload = json.loads(success.stdout)
    assert payload["collection"] == "evaluation-flow"
    assert payload["top_k"] == 1
    assert payload["query_count"] == 3
    assert payload["quality_gate"] is None
    assert successful_output.is_file()
    assert (data_root / "db" / "chroma" / "chroma.sqlite3").is_file()
    assert not Path(settings_values(tmp_path)["vector_store"]["persist_directory"]).exists()

    bad_test_set = write_test_set(tmp_path / "bad-label.json", [{
        "case_id": "wrong-source", "query": "文件 SHA256 相同且已经成功处理，怎样强制重新摄取？",
        "expected_sources": ["missing-target.md"],
    }])
    failed = run_cli(environment, "--config", config_path, "--test-set", bad_test_set,
                      "--collection", "evaluation-flow", "--data-dir", data_root,
                      "--backend", "custom", "--top-k", "1", "--min-hit-rate", "1",
                      "--min-mrr", "1", "--json")
    assert failed.returncode == 1
    failure_payload = json.loads(failed.stdout)
    assert failure_payload["quality_gate"]["passed"] is False
    assert "below_threshold:hit_rate" in failure_payload["quality_gate"]["failures"]
    assert "DO_NOT_FORWARD_API_KEY_SENTINEL" not in failed.stdout + failed.stderr
