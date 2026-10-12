"""H2：顺序执行多个评估器并合并指标，不平均为总分。"""

from dataclasses import replace
import logging
from math import isfinite
import time

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS

logger = logging.getLogger(__name__)


class CompositeEvaluationError(RuntimeError):
    """没有任何评估器成功产出指标。"""


class CompositeEvaluator(BaseEvaluator):
    """调用方串行复用；last_errors 是最近一次调用诊断，不是多请求共享状态。"""

    def __init__(self, evaluators=None, settings=None, **kwargs):
        self.settings = settings
        self.initialization_errors = {}
        self.last_errors = {}
        if evaluators is not None:
            if not isinstance(evaluators, (list, tuple)) or not evaluators:
                raise ValueError("Composite requires a nonempty evaluator list")
            if any(not isinstance(evaluator, BaseEvaluator) for evaluator in evaluators):
                raise ValueError("Composite children must implement BaseEvaluator")
            self._children = [(f"{type(evaluator).__name__}:{index}", evaluator) for index, evaluator in enumerate(evaluators)]
        else:
            self._children = self._build_from_settings(settings, **kwargs)

    @property
    def evaluators(self):
        return [evaluator for _, evaluator in self._children]

    def _build_from_settings(self, settings, **kwargs):
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        if settings is None or not settings.evaluation.backends:
            raise ValueError("Composite requires evaluation.backends or injected evaluators")
        backends = settings.evaluation.backends
        if not isinstance(backends, (list, tuple)) or any(not isinstance(name, str) or not name.strip() for name in backends):
            raise ValueError("Evaluation backends must be nonempty names")
        names = [name.strip().lower() for name in backends]
        if len(names) != len(set(names)) or any(name in {"composite", "none", "disabled"} for name in names):
            raise ValueError("Composite backends must be unique concrete providers")
        if any(name not in EvaluatorFactory.list_providers() for name in names):
            raise ValueError("Unknown composite backend")
        metrics = kwargs.pop("metrics", settings.evaluation.metrics)
        if not isinstance(metrics, (list, tuple)) or any(not isinstance(metric, str) for metric in metrics):
            raise ValueError("Composite metrics must be a list of names")
        metrics = list(dict.fromkeys(metric.strip().lower() for metric in metrics))
        supported = {"custom": CustomEvaluator.SUPPORTED_METRICS, "ragas": SUPPORTED_METRICS}
        if all(name in supported for name in names):
            available = set().union(*(supported[name] for name in names))
            if any(metric not in available for metric in metrics):
                raise ValueError("Unsupported composite metrics")
        children = []
        for name in names:
            selected = [metric for metric in metrics if metric in supported[name]] if name in supported else metrics
            # 有明确指标清单时，没有匹配项的后端不应默认计算额外指标。
            if metrics and not selected:
                continue
            sub_settings = replace(settings, evaluation=replace(settings.evaluation, enabled=True, provider=name,
                                                                metrics=selected, backends=None))
            try:
                child = EvaluatorFactory.create(sub_settings, **kwargs)
                children.append((name, child))
            except Exception as exc:
                self.initialization_errors[name] = type(exc).__name__
        return children

    def evaluate(self, query, retrieved_chunks, generated_answer=None, ground_truth=None, trace=None, **kwargs):
        self.last_errors = dict(self.initialization_errors)
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        started = time.monotonic()
        merged = {}
        for name, evaluator in self._children:
            child_started = time.monotonic()
            try:
                values = evaluator.evaluate(query=query, retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer, ground_truth=ground_truth, trace=trace, **kwargs)
                if not isinstance(values, dict) or not values:
                    raise ValueError("Evaluator returned no metrics")
                scores = {}
                for key, value in values.items():
                    if not isinstance(key, str) or not key or isinstance(value, (str, bool)):
                        raise ValueError("Evaluator metrics need names and numeric scores")
                    score = float(value)
                    if not isfinite(score) or key in merged:
                        raise ValueError("Nonfinite or duplicate metric")
                    scores[key] = score
            except Exception as exc:
                self.last_errors[name] = type(exc).__name__
                logger.warning("Evaluation backend %s failed: %s", name, type(exc).__name__)
                if trace is not None:
                    trace.record_stage("evaluation_backend", {"method": name, "status": "error", "error_type": type(exc).__name__},
                                       elapsed_ms=(time.monotonic() - child_started) * 1000)
                continue
            merged.update(scores)
            if trace is not None:
                trace.record_stage("evaluation_backend", {"method": name, "status": "success", "metrics": dict(scores)},
                                   elapsed_ms=(time.monotonic() - child_started) * 1000)
        if trace is not None:
            trace.record_stage("evaluation_composite", {"method": "composite", "status": "error" if not merged else "partial" if self.last_errors else "success",
                "metrics": dict(merged), "errors": dict(self.last_errors)}, elapsed_ms=(time.monotonic() - started) * 1000)
        if not merged:
            raise CompositeEvaluationError("No evaluator produced valid metrics") from None
        return merged
