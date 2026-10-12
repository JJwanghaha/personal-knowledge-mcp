"""H5：用公开 Markdown 样例回归真实 Chroma/BM25/jieba 检索和质量门槛。"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket

import jieba
import pytest

from src.core.query_engine.query_service import create_query_components
from src.core.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.dashboard.services.data_service import DataService
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.quality_gate import check_quality_gate


pytestmark = pytest.mark.e2e
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "evaluation"
TEST_SET_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"
FIXTURE_NAMES = ("01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md")
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


class FixtureEmbedding(BaseEmbedding):
    """固定维度本地词项向量；Chroma、BM25 与 jieba 保持真实实现。"""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        if self.fail:
            raise RuntimeError("MODEL_API_KEY_SENTINEL")
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


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("离线召回回归不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def settings_for(data_root):
    base = Settings.from_dict({
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "fixture-hash", "dimensions": 128},
        "vector_store": {"provider": "chroma", "persist_directory": str(data_root / "db" / "chroma"),
                         "collection_name": "recall-e2e"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": True, "provider": "custom", "metrics": ["hit_rate", "mrr"]},
        "observability": {"log_level": "ERROR", "trace_enabled": False,
                          "trace_file": str(data_root / "unused-traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 1200, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    })
    return base


def ingest_fixtures(settings, data_root):
    pipeline = IngestionPipeline(settings, "recall-e2e", embedding=FixtureEmbedding(), data_dir=data_root,
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


def write_test_set(path, cases):
    path.write_text(json.dumps({"version": "1.0", "reviewed": True, "test_cases": cases},
                               ensure_ascii=False), encoding="utf-8")
    return path


def open_runner(settings, data_root, collection, embedding):
    components = create_query_components(settings, collection, data_root / "db" / "bm25",
                                         embedding_client=embedding)
    return components, EvalRunner(settings, components.hybrid_search, CustomEvaluator())


def test_public_fixture_recall_and_quality_gate_fail_after_evidence_loss_or_execution_degradation(tmp_path):
    data_root = tmp_path / "isolated-recall-data"
    settings = settings_for(data_root)
    ingested = ingest_fixtures(settings, data_root)
    corpus_size = sum(result.chunk_count for result in ingested.values())
    assert corpus_size == 3
    golden = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    assert len(golden["test_cases"]) == 3
    thresholds = golden["thresholds"]

    components, runner = open_runner(settings, data_root, "recall-e2e", FixtureEmbedding())
    try:
        assert components.vector_store.collection.count() == corpus_size
        report = runner.run(TEST_SET_PATH, top_k=1, collection="recall-e2e")
    finally:
        components.close()

    assert report.top_k == 1
    assert len(report.query_results) == 3
    assert all(len(row.retrieved_chunk_ids) == 1 for row in report.query_results)
    assert all(len(row.retrieved_chunk_ids) < corpus_size for row in report.query_results)
    assert report.metric_counts == {"hit_rate": 3, "mrr": 3}
    assert check_quality_gate(report, thresholds).passed

    data = DataService(settings, data_dir=data_root, embedding_client=FixtureEmbedding())
    target = next(item for item in data.list_documents("recall-e2e")
                  if item["source_path"].endswith(FIXTURE_NAMES[0]))
    deletion = data.delete_document(target["source_path"], "recall-e2e", target["source_hash"])
    assert deletion.success

    damaged_set = write_test_set(tmp_path / "target-deleted.json", [golden["test_cases"][0]])
    after_deletion, deleted_runner = open_runner(settings, data_root, "recall-e2e", FixtureEmbedding())
    try:
        missed = deleted_runner.run(damaged_set, top_k=1, collection="recall-e2e")
    finally:
        after_deletion.close()
    assert missed.query_results[0].retrieved_chunk_ids
    assert missed.query_results[0].metrics == {"hit_rate": 0.0, "mrr": 0.0}
    lost_evidence_gate = check_quality_gate(missed, {"hit_rate": 1.0, "mrr": 1.0})
    assert not lost_evidence_gate.passed
    assert "below_threshold:hit_rate" in lost_evidence_gate.failures

    empty_store = VectorStoreFactory.create(settings, collection_name="empty-recall", create_if_missing=True)
    empty_store.client.close()
    empty_components, empty_runner = open_runner(settings, data_root, "empty-recall", FixtureEmbedding())
    labelled_empty = write_test_set(tmp_path / "labelled-empty.json", [golden["test_cases"][0]])
    unlabelled_empty = write_test_set(tmp_path / "unlabelled-empty.json", [{
        "case_id": "no-label", "query": golden["test_cases"][0]["query"],
    }])
    try:
        empty_result = empty_runner.run(labelled_empty, top_k=1, collection="empty-recall")
        assert empty_result.query_results[0].retrieved_chunks == []
        assert empty_result.query_results[0].metrics == {"hit_rate": 0.0, "mrr": 0.0}
        assert not check_quality_gate(empty_result, {"hit_rate": 0.01}).passed

        unlabelled_result = empty_runner.run(unlabelled_empty, top_k=1, collection="empty-recall")
        assert unlabelled_result.query_results[0].status == "skipped"
        assert unlabelled_result.query_results[0].metrics == {}
        assert not check_quality_gate(unlabelled_result, {"hit_rate": 0.01}).passed
    finally:
        empty_components.close()

    degraded_components, degraded_runner = open_runner(
        settings, data_root, "recall-e2e", FixtureEmbedding(fail=True),
    )
    try:
        degraded = degraded_runner.run(
            write_test_set(tmp_path / "fallback.json", [golden["test_cases"][1]]),
            top_k=1, collection="recall-e2e",
        )
    finally:
        degraded_components.close()
    assert degraded.query_results[0].errors == {"retrieval": "fallback"}
    assert degraded.query_results[0].metrics
    fallback_gate = check_quality_gate(degraded, {"hit_rate": 0.01})
    assert not fallback_gate.passed
    assert "execution_errors" in fallback_gate.failures
