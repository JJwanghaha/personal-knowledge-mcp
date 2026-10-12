"""H3：通过真实本地检索边界检查逐题评估、标签口径和失败分类。"""

import builtins
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket

import pytest

from src.core.query_engine.query_service import create_query_components
from src.core.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.evaluator.base_evaluator import NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.eval_runner import EvalRunner, load_test_set, write_report
from src.observability.evaluation.quality_gate import check_quality_gate
from src.observability.evaluation.ragas_evaluator import RagasEvaluator
from src.observability.evaluation.testset_generator import LLMAnswerGenerator


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "evaluation"
FIXTURE_NAMES = ("01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md")
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


class FixtureEmbedding(BaseEmbedding):
    """在 BaseEmbedding seam 上用 jieba 分词生成稳定的本地向量。"""

    def embed(self, texts, trace=None, **kwargs):
        import jieba

        vectors = []
        for text in texts:
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


def typed_ragas_llm():
    """按 Ragas 0.4.3 的现代类型接口提供不联网替身。"""
    from ragas.llms.base import InstructorBaseRagasLLM

    class TypedRagasLLM(InstructorBaseRagasLLM):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, response_model):
            self.calls += 1
            raise AssertionError("此用例不应请求 Ragas 模型")

        async def agenerate(self, prompt, response_model):
            return self.generate(prompt, response_model)

        def close(self):
            pass

    return TypedRagasLLM()


class RecordingLLM(BaseLLM):
    """只替代 BaseLLM 外部 seam，记录提示词并按指定问题返回模型错误。"""

    def __init__(self):
        self.prompts = []

    def chat(self, messages, trace=None, **kwargs):
        self.prompts.append("\n".join(message.content for message in messages))
        user_text = messages[-1].content
        if "答题异常探针" in user_text:
            raise RuntimeError("MODEL_API_KEY_SENTINEL")
        return ChatResponse("模型生成的待评答案", "offline-test-model")


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("评估单元测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def settings_for(tmp_path):
    return Settings.from_dict({
        "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "offline", "model": "fixture-hash", "dimensions": 128},
        "vector_store": {"provider": "chroma", "persist_directory": str(tmp_path / "unused-store"),
                         "collection_name": "runner"},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 5},
        "evaluation": {"enabled": True, "provider": "custom", "metrics": ["hit_rate", "mrr"]},
        "observability": {"log_level": "ERROR", "trace_enabled": False,
                          "trace_file": str(tmp_path / "unused-traces.jsonl"), "structured_logging": False},
        "ingestion": {"chunk_size": 1200, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "offline", "model": "offline", "max_image_size": 512},
    })


def indexed_workspace(tmp_path, collection="runner"):
    settings = settings_for(tmp_path)
    data_root = tmp_path / "isolated-data"
    pipeline = IngestionPipeline(settings, collection, embedding=FixtureEmbedding(), data_dir=data_root,
                                 source_root=FIXTURE_ROOT)
    ingested = {}
    try:
        for name in FIXTURE_NAMES:
            result = pipeline.run(FIXTURE_ROOT / name)
            assert result.success, result.error
            ingested[name] = result
    finally:
        pipeline.close()

    runtime_settings = replace(
        settings,
        vector_store=replace(settings.vector_store, persist_directory=str(data_root / "db" / "chroma")),
    )
    components = create_query_components(
        runtime_settings, collection, data_root / "db" / "bm25", embedding_client=FixtureEmbedding(),
    )
    return runtime_settings, data_root, components, ingested


def write_test_set(path, cases, *, reviewed=True):
    path.write_text(json.dumps({"version": "1.0", "reviewed": reviewed, "test_cases": cases},
                               ensure_ascii=False), encoding="utf-8")
    return path


def test_report_preserves_case_labels_hash_settings_and_scored_case_averages(tmp_path):
    settings, _, components, ingested = indexed_workspace(tmp_path)
    target_id = ingested[FIXTURE_NAMES[0]].vector_ids[0]
    cases = [
        {"case_id": "id-wins", "query": "文件 SHA256 相同且已经成功处理，怎样强制重新摄取？",
         "expected_chunk_ids": ["chunk-that-does-not-exist"], "expected_sources": [FIXTURE_NAMES[0]]},
        {"case_id": "path-boundary", "query": "文件 SHA256 相同且已经成功处理，怎样强制重新摄取？",
         "expected_sources": ["1-摄取流程.md"]},
        {"case_id": "source-hit", "query": "文件 SHA256 相同且已经成功处理，怎样强制重新摄取？",
         "expected_sources": [FIXTURE_NAMES[0]]},
    ]
    test_path = write_test_set(tmp_path / "labels.json", cases)
    raw = test_path.read_bytes()
    try:
        report = EvalRunner(settings, components.hybrid_search, CustomEvaluator()).run(
            test_path, top_k=3, collection="runner",
        )
    finally:
        components.close()

    assert [row.case_id for row in report.query_results] == ["id-wins", "path-boundary", "source-hit"]
    assert report.query_results[0].label_scope == "chunk_id"
    assert report.query_results[0].metrics == {"hit_rate": 0.0, "mrr": 0.0}
    assert report.query_results[1].label_scope == "source"
    assert report.query_results[1].metrics == {"hit_rate": 0.0, "mrr": 0.0}
    assert report.query_results[2].metrics == {"hit_rate": 1.0, "mrr": 1.0}
    assert target_id in report.query_results[2].retrieved_chunk_ids
    assert report.metric_counts == {"hit_rate": 3, "mrr": 3}
    assert report.aggregate_metrics == {"hit_rate": pytest.approx(1 / 3), "mrr": pytest.approx(1 / 3)}
    assert report.test_set_sha256 == hashlib.sha256(raw).hexdigest()
    assert report.test_set_path == str(test_path.resolve())
    assert report.collection == "runner" and report.top_k == 3
    assert report.configuration["retrieval"]["fusion_top_k"] == 10

    assert check_quality_gate(report, {"hit_rate": 0.3, "mrr": 0.3}).passed
    for threshold in (0, -0.1, 1.01, float("inf"), True):
        with pytest.raises(ValueError):
            check_quality_gate(report, {"hit_rate": threshold})

    saved = write_report(report, tmp_path / "reports" / "fixed.json")
    original = saved.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(report, saved)
    assert saved.read_bytes() == original


def test_empty_retrieval_is_zero_only_when_labelled_and_duplicate_metric_is_not_overwritten(tmp_path):
    settings, data_root, components, _ = indexed_workspace(tmp_path)
    components.close()
    empty_store = None
    empty_components = None
    try:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        empty_store = VectorStoreFactory.create(settings, collection_name="empty", create_if_missing=True)
        empty_store.client.close()
        empty_components = create_query_components(
            settings, "empty", data_root / "db" / "bm25", embedding_client=FixtureEmbedding(),
        )
        path = write_test_set(tmp_path / "empty.json", [
            {"case_id": "no-label", "query": "完全不存在的检索问题"},
            {"case_id": "labelled-empty", "query": "MCP stdio 通信为什么将日志写到 stderr？",
             "expected_sources": [FIXTURE_NAMES[2]]},
        ])
        report = EvalRunner(settings, empty_components.hybrid_search, CustomEvaluator()).run(
            path, top_k=1, collection="empty",
        )
        assert report.query_results[0].status == "skipped"
        assert report.query_results[0].skipped == {"CustomEvaluator:0": "missing_expected_evidence"}
        assert report.query_results[0].metrics == {}
        assert report.query_results[1].status == "success"
        assert report.query_results[1].retrieved_chunks == []
        assert report.query_results[1].retrieved_chunk_ids == []
        assert report.query_results[1].metrics == {"hit_rate": 0.0, "mrr": 0.0}
        assert report.metric_counts == {"hit_rate": 1, "mrr": 1}
        gate = check_quality_gate(report, {"hit_rate": 0.01, "mrr": 0.01})
        assert not gate.passed
        assert "incomplete_coverage:hit_rate" in gate.failures

        duplicate_case = write_test_set(tmp_path / "duplicate.json", [
            {"case_id": "duplicate", "query": "RRF 如何融合 Dense 与 BM25？",
             "expected_sources": [FIXTURE_NAMES[1]]},
        ])
        duplicate_evaluator = CompositeEvaluator(evaluators=[
            CustomEvaluator(metrics=["hit_rate"]), CustomEvaluator(metrics=["hit_rate"]),
        ])
        duplicate = EvalRunner(settings, empty_components.hybrid_search, duplicate_evaluator).run(
            duplicate_case, top_k=1, collection="empty",
        )
        row = duplicate.query_results[0]
        assert row.retrieved_chunks == []
        assert row.metrics == {"hit_rate": 0.0}
        assert row.errors == {"CustomEvaluator:1": "duplicate_metric"}
        assert duplicate.metric_counts == {"hit_rate": 1}
        assert not check_quality_gate(duplicate, {"hit_rate": 0.01}).passed
    finally:
        if empty_components is not None:
            empty_components.close()


def test_reference_answer_never_becomes_generated_answer_and_ragas_is_skipped(tmp_path):
    assert importlib.metadata.version("ragas") == "0.4.3"
    settings, _, components, _ = indexed_workspace(tmp_path)
    ragas_llm = typed_ragas_llm()
    composite_settings = replace(
        settings,
        evaluation=replace(settings.evaluation, provider="composite",
                           metrics=["hit_rate", "mrr", "faithfulness"], backends=["custom", "ragas"]),
    )
    evaluator = EvaluatorFactory.create(composite_settings, llm=ragas_llm)
    path = write_test_set(tmp_path / "reference-only.json", [{
        "case_id": "reference-only", "query": "MCP stdio 通信为什么将日志写到 stderr？",
        "expected_sources": [FIXTURE_NAMES[2]], "reference_answer": "REF_MUST_NOT_BE_THE_GENERATED_ANSWER",
    }])
    try:
        report = EvalRunner(composite_settings, components.hybrid_search, evaluator).run(
            path, top_k=1, collection="runner",
        )
    finally:
        components.close()

    row = report.query_results[0]
    assert row.reference_answer == "REF_MUST_NOT_BE_THE_GENERATED_ANSWER"
    assert row.generated_answer is None and row.answer_source == "none"
    assert row.skipped == {"RagasEvaluator:1": "missing_generated_answer"}
    assert "faithfulness" not in row.metrics
    assert ragas_llm.calls == 0
    assert report.metric_counts == {"hit_rate": 1, "mrr": 1}


def test_composite_initialization_errors_survive_while_custom_scores_only_eligible_rows(tmp_path, monkeypatch):
    assert importlib.metadata.version("ragas") == "0.4.3"
    settings, _, components, _ = indexed_workspace(tmp_path)
    ragas_llm = typed_ragas_llm()
    composite_settings = replace(
        settings,
        evaluation=replace(settings.evaluation, provider="composite",
                           metrics=["hit_rate", "faithfulness"], backends=["custom", "ragas"]),
    )
    original_import = builtins.__import__

    def import_without_ragas(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("simulated optional dependency initialization failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_ragas)
    evaluator = EvaluatorFactory.create(composite_settings, llm=ragas_llm)
    path = write_test_set(tmp_path / "partial-composite.json", [
        {"case_id": "eligible", "query": "MCP stdio 通信为什么将日志写到 stderr？",
         "expected_sources": [FIXTURE_NAMES[2]]},
        {"case_id": "unlabelled", "query": "Dense 和 BM25 的区别是什么？"},
    ])
    try:
        report = EvalRunner(composite_settings, components.hybrid_search, evaluator).run(
            path, top_k=1, collection="runner",
        )
    finally:
        components.close()

    assert evaluator.initialization_errors == {"ragas": "ImportError"}
    assert report.query_results[0].metrics["hit_rate"] == 1.0
    assert report.query_results[0].errors["ragas"] == "ImportError"
    assert report.query_results[1].status == "error"
    assert report.query_results[1].skipped == {"CustomEvaluator:0": "missing_expected_evidence"}
    assert report.query_results[1].errors["ragas"] == "ImportError"
    assert report.metric_counts == {"hit_rate": 1}
    gate = check_quality_gate(report, {"hit_rate": 0.01})
    assert not gate.passed
    assert "execution_errors" in gate.failures
    assert "incomplete_coverage:hit_rate" in gate.failures
    assert ragas_llm.calls == 0


def test_model_errors_are_reported_per_case_without_exposing_api_sentinels(tmp_path, caplog):
    settings, _, components, _ = indexed_workspace(tmp_path)
    llm = RecordingLLM()
    runner = EvalRunner(settings, components.hybrid_search, NoneEvaluator(),
                        answer_generator=LLMAnswerGenerator(settings, llm))
    path = write_test_set(tmp_path / "answer-generation.json", [
        {"case_id": "generated", "query": "MCP stdio 通信为什么将日志写到 stderr？",
         "reference_answer": "REFERENCE_SECRET_NOT_FOR_PROMPT"},
        {"case_id": "model-error", "query": "答题异常探针和检索证据",
         "reference_answer": "ANOTHER_REFERENCE_SECRET"},
    ])
    try:
        report = runner.run(path, top_k=1, collection="runner")
    finally:
        components.close()

    generated, failed = report.query_results
    assert generated.generated_answer == "模型生成的待评答案"
    assert generated.answer_source == "generated"
    assert failed.status == "error"
    assert failed.errors == {"answer_generation": "RuntimeError"}
    assert failed.generated_answer is None and failed.answer_source == "none"
    assert all("REFERENCE_SECRET" not in prompt for prompt in llm.prompts)
    assert "MODEL_API_KEY_SENTINEL" not in json.dumps(report.to_dict(), ensure_ascii=False)
    assert "MODEL_API_KEY_SENTINEL" not in caplog.text
