"""Composite evaluator 的公开配置、隔离与诊断行为。"""

import builtins

import pytest

from src.core.settings import Settings
from src.core.trace import TraceContext
from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.composite_evaluator import CompositeEvaluationError, CompositeEvaluator


pytestmark = pytest.mark.unit


def settings_for(
    tmp_path,
    *,
    provider="composite",
    enabled=True,
    backends=None,
    metrics=None,
):
    """从离线字典构造 Settings，不读取项目配置或外部数据。"""
    evaluation = {"enabled": enabled, "provider": provider, "metrics": list(metrics or [])}
    if backends is not None:
        evaluation["backends"] = list(backends)
    return Settings.from_dict(
        {
            "llm": {"provider": "offline", "model": "offline", "temperature": 0.0, "max_tokens": 32},
            "embedding": {"provider": "offline", "model": "offline", "dimensions": 2},
            "vector_store": {
                "provider": "chroma",
                "persist_directory": str(tmp_path / "unused-store"),
                "collection_name": "unused-test-collection",
            },
            "retrieval": {"dense_top_k": 3, "sparse_top_k": 3, "fusion_top_k": 3, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none", "model": "offline", "top_k": 3},
            "evaluation": evaluation,
            "observability": {
                "log_level": "INFO",
                "trace_enabled": False,
                "trace_file": str(tmp_path / "unused-trace.jsonl"),
                "structured_logging": False,
            },
        }
    )


class ScriptedEvaluator(BaseEvaluator):
    """可控的外部策略替身，只通过 BaseEvaluator.evaluate 接入。"""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    def evaluate(self, query, retrieved_chunks, generated_answer=None, ground_truth=None, trace=None, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def block_ragas_import(monkeypatch):
    """在可选依赖边界模拟 Ragas 缺失，不改评估器内部实现。"""
    original_import = builtins.__import__

    def import_without_ragas(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("offline test: optional dependency unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_ragas)


def test_settings_from_dict_keeps_backends_optional_and_custom_defaults(tmp_path):
    settings = settings_for(tmp_path, provider="custom")

    assert settings.evaluation.backends is None
    evaluator = EvaluatorFactory.create(settings)
    assert evaluator.evaluate("问题", ["先前片段", "命中片段"], ground_truth="命中片段") == {
        "hit_rate": 1.0,
        "mrr": 0.5,
    }


def test_disabled_factory_does_not_import_or_construct_ragas(monkeypatch, tmp_path):
    block_ragas_import(monkeypatch)
    settings = settings_for(tmp_path, provider="ragas", enabled=False)

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, NoneEvaluator)
    assert evaluator.evaluate("问题", ["候选片段"]) == {}


@pytest.mark.parametrize(
    "backends,metrics",
    [
        (["composite"], ["mrr"]),
        (["none"], ["mrr"]),
        (["disabled"], ["mrr"]),
        (["custom", "custom"], ["mrr"]),
        (["not-registered"], ["mrr"]),
        (["custom"], ["faithfulness"]),
        (["custom", "ragas"], ["not-a-metric"]),
    ],
)
def test_factory_rejects_invalid_composite_backend_configuration(tmp_path, backends, metrics):
    settings = settings_for(tmp_path, backends=backends, metrics=metrics)

    with pytest.raises(RuntimeError, match="Failed to instantiate Evaluator provider 'composite': ValueError"):
        EvaluatorFactory.create(settings)


def test_missing_ragas_child_keeps_custom_metrics_and_reports_only_error_type(monkeypatch, tmp_path):
    block_ragas_import(monkeypatch)
    settings = settings_for(tmp_path, backends=["custom", "ragas"], metrics=["hit_rate", "faithfulness"])

    evaluator = EvaluatorFactory.create(settings)
    result = evaluator.evaluate("问题", ["不相关", "目标"], ground_truth="目标")

    assert result == {"hit_rate": 1.0}
    assert evaluator.last_errors == {"ragas": "ImportError"}


def test_all_backend_initialization_failures_raise_without_zero_metrics(monkeypatch, tmp_path):
    block_ragas_import(monkeypatch)
    settings = settings_for(tmp_path, backends=["ragas"], metrics=["faithfulness"])
    evaluator = EvaluatorFactory.create(settings)

    with pytest.raises(CompositeEvaluationError, match="No evaluator produced valid metrics"):
        evaluator.evaluate("问题", ["证据"], generated_answer="答案")

    assert evaluator.last_errors == {"ragas": "ImportError"}


@pytest.mark.parametrize(
    "bad_output",
    [pytest.param({}, id="empty-dict"), pytest.param(None, id="none"),
     pytest.param({"new": float("nan")}, id="nan"), pytest.param({"new": True}, id="bool"),
     pytest.param({"new": "0.5"}, id="string")],
)
def test_invalid_backend_output_is_discarded_without_partial_commit(bad_output):
    evaluator = CompositeEvaluator(
        evaluators=[ScriptedEvaluator({"stable": 0.25}), ScriptedEvaluator(bad_output)]
    )

    result = evaluator.evaluate("问题", ["候选"])

    assert result == {"stable": 0.25}
    assert evaluator.last_errors == {"ScriptedEvaluator:1": "ValueError"}


def test_duplicate_metric_does_not_overwrite_or_commit_sibling_values():
    evaluator = CompositeEvaluator(
        evaluators=[
            ScriptedEvaluator({"shared": 0.25}),
            ScriptedEvaluator({"new": 0.75, "shared": 0.9}),
        ]
    )

    result = evaluator.evaluate("问题", ["候选"])

    assert result == {"shared": 0.25}
    assert evaluator.last_errors == {"ScriptedEvaluator:1": "ValueError"}


def test_runtime_failure_is_partial_type_only_and_last_errors_reset_on_reuse():
    sentinel = "test-sdk-secret-sentinel"
    intermittent = ScriptedEvaluator(RuntimeError(f"request included {sentinel}"), {"later": 0.5})
    stable = ScriptedEvaluator({"stable": 0.75}, {"stable": 0.8})
    evaluator = CompositeEvaluator(evaluators=[intermittent, stable])
    first_trace = TraceContext()

    assert evaluator.evaluate("问题", ["候选"], trace=first_trace) == {"stable": 0.75}
    assert evaluator.last_errors == {"ScriptedEvaluator:0": "RuntimeError"}
    composite_stage = first_trace.get_stage_data("evaluation_composite")
    assert composite_stage["status"] == "partial"
    assert composite_stage["errors"] == {"ScriptedEvaluator:0": "RuntimeError"}
    backend_stages = [item for item in first_trace.stages if item["stage"] == "evaluation_backend"]
    assert [item["data"]["status"] for item in backend_stages] == ["error", "success"]
    assert all(item["elapsed_ms"] >= 0 for item in backend_stages)
    assert first_trace.get_stage_data("evaluation_backend")["status"] == "success"
    assert sentinel not in repr(first_trace.stages)
    assert first_trace.finished_at is None
    assert "collect" not in repr(first_trace.stages).lower()

    second_trace = TraceContext()
    assert evaluator.evaluate("问题", ["候选"], trace=second_trace) == {"later": 0.5, "stable": 0.8}
    assert evaluator.last_errors == {}
    assert second_trace.get_stage_data("evaluation_composite")["status"] == "success"
    assert second_trace.finished_at is None


def test_all_runtime_failures_raise_and_keep_trace_error_types_only():
    sentinel = "private-sdk-detail-sentinel"
    evaluator = CompositeEvaluator(evaluators=[ScriptedEvaluator(RuntimeError(sentinel))])
    trace = TraceContext()

    with pytest.raises(CompositeEvaluationError) as error:
        evaluator.evaluate("问题", ["候选"], trace=trace)

    assert str(error.value) == "No evaluator produced valid metrics"
    assert evaluator.last_errors == {"ScriptedEvaluator:0": "RuntimeError"}
    assert trace.get_stage_data("evaluation_composite")["status"] == "error"
    assert trace.get_stage_data("evaluation_composite")["errors"] == evaluator.last_errors
    assert sentinel not in repr(trace.stages)
    assert trace.finished_at is None
