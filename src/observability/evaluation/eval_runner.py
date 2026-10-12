"""H3：固定试卷→检索/可选回答→评估→逐题与汇总报告。"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

from src.core.trace.snapshots import snapshot_results
from src.libs.evaluator.base_evaluator import NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


@dataclass
class GoldenTestCase:
    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    generated_answer: str | None = None
    case_id: str = ""
    evidence_quote: str | None = None

    @classmethod
    def from_dict(cls, data, index=0):
        if not isinstance(data, dict) or not isinstance(data.get("query"), str) or not data["query"].strip():
            raise ValueError("Each test case needs a nonblank query")
        for name in ("expected_chunk_ids", "expected_sources"):
            values = data.get(name, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError("Expected evidence must be a list of nonblank strings")
        for name in ("reference_answer", "generated_answer", "evidence_quote"):
            if data.get(name) is not None and (not isinstance(data[name], str) or not data[name].strip()):
                raise ValueError("Answer and quote fields must be nonblank text or null")
        case_id = data.get("case_id", f"case-{index + 1}")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be nonblank text")
        return cls(data["query"], list(data.get("expected_chunk_ids", [])), list(data.get("expected_sources", [])),
                   data.get("reference_answer"), data.get("generated_answer"), case_id, data.get("evidence_quote"))


def load_test_set(path) -> list[GoldenTestCase]:
    """LLM 草稿 reviewed=false 时禁止评分；兼容上游未声明 reviewed 的手工文件。"""
    return _parse_test_set(json.loads(Path(path).read_text(encoding="utf-8")))


def _parse_test_set(data):
    if not isinstance(data, dict) or data.get("reviewed", True) is not True:
        raise ValueError("Review the draft before setting reviewed=true and evaluating")
    raw = data.get("test_cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError("test_cases must be a nonempty list")
    cases = [GoldenTestCase.from_dict(item, index) for index, item in enumerate(raw)]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Test case IDs must be unique")
    return cases


@dataclass
class QueryResult:
    query: str
    case_id: str = ""
    status: str = "skipped"
    label_scope: str = "unlabelled"
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    generated_answer: str | None = None
    reference_answer: str | None = None
    answer_source: str = "none"
    metrics: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    query_results: list[QueryResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""
    test_set_sha256: str = ""
    collection: str | None = None
    top_k: int = 5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    configuration: dict = field(default_factory=dict)

    def to_dict(self):
        result = asdict(self)
        result["query_count"] = len(self.query_results)
        result["status_counts"] = {name: sum(row.status == name for row in self.query_results)
                                   for name in ("success", "partial", "skipped", "error")}
        return result


def write_report(report: EvalReport, path):
    """显式保存私人报告，拒绝覆盖已有文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path


class EvalRunner:
    def __init__(self, settings=None, hybrid_search=None, evaluator=None, answer_generator=None,
                 answer_overrides=None, reranker=None):
        self.settings, self.hybrid_search, self.evaluator = settings, hybrid_search, evaluator
        self.answer_generator, self.answer_overrides, self.reranker = answer_generator, answer_overrides or {}, reranker

    def run(self, test_set_path, top_k=5, collection=None) -> EvalReport:
        if self.hybrid_search is None or self.evaluator is None:
            raise ValueError("EvalRunner requires a retriever and evaluator")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        raw = Path(test_set_path).read_bytes()
        cases = _parse_test_set(json.loads(raw))
        report = EvalReport(evaluator_name=type(self.evaluator).__name__, test_set_path=str(Path(test_set_path).resolve()),
                            test_set_sha256=sha256(raw).hexdigest(), collection=collection, top_k=top_k)
        if self.settings is not None:
            s = self.settings
            report.configuration = {"llm": {"provider": s.llm.provider, "model": s.llm.model},
                "embedding": {"provider": s.embedding.provider, "model": s.embedding.model, "dimensions": s.embedding.dimensions},
                "retrieval": asdict(s.retrieval), "rerank": asdict(s.rerank), "evaluation": asdict(s.evaluation),
                "ingestion": {"chunk_size": s.ingestion.chunk_size, "chunk_overlap": s.ingestion.chunk_overlap} if s.ingestion else None}
        started = time.monotonic()
        for index, case in enumerate(cases):
            report.query_results.append(self._evaluate_single(case, index, top_k, collection))
        for row in report.query_results:
            for metric, value in row.metrics.items():
                report.aggregate_metrics[metric] = report.aggregate_metrics.get(metric, 0.0) + value
                report.metric_counts[metric] = report.metric_counts.get(metric, 0) + 1
        report.aggregate_metrics = {metric: total / report.metric_counts[metric] for metric, total in report.aggregate_metrics.items()}
        report.total_elapsed_ms = (time.monotonic() - started) * 1000
        return report

    @staticmethod
    def _source_matches(source, expected):
        source, expected = str(source).replace("\\", "/"), expected.replace("\\", "/")
        return source == expected if Path(expected).is_absolute() else source == expected or source.endswith("/" + expected)

    def _evaluate_single(self, case, index, top_k, collection):
        row = QueryResult(case.query, case.case_id, reference_answer=case.reference_answer)
        started = time.monotonic()
        try:
            candidate_k = max(top_k, getattr(getattr(self.hybrid_search, "config", None), "fusion_top_k", top_k))
            retrieved = self.hybrid_search.search(case.query, top_k=candidate_k,
                filters={"collection": collection} if collection is not None else None, return_details=True)
            if getattr(retrieved, "used_fallback", False):
                row.errors["retrieval"] = "fallback"
            results = retrieved if isinstance(retrieved, list) else retrieved.results
            if self.reranker is not None:
                ranked = self.reranker.rerank(case.query, results, top_k=top_k)
                results = ranked.results
                if ranked.used_fallback:
                    row.errors["rerank"] = "fallback"
            results = results[:top_k]
            row.retrieved_chunks = snapshot_results(results)
            row.retrieved_chunk_ids = [item.chunk_id for item in results]
            row.retrieved_sources = [str(item.metadata.get("source_path", "")) for item in results]
            if case.expected_chunk_ids:
                expected_ids, row.label_scope = case.expected_chunk_ids, "chunk_id"
            elif case.expected_sources:
                expected_ids = [item.chunk_id for item in results if any(self._source_matches(item.metadata.get("source_path", ""),
                                source) for source in case.expected_sources)]
                row.label_scope = "source"
            else:
                expected_ids = []
            override = self.answer_overrides.get(index)
            row.generated_answer = override if override is not None else case.generated_answer
            if row.generated_answer is not None:
                if not isinstance(row.generated_answer, str) or not row.generated_answer.strip():
                    raise ValueError("Provided answer must be nonblank text")
                row.answer_source = "provided"
            elif self.answer_generator is not None and results:
                try:
                    answer = self.answer_generator(case.query, results)
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Answer generator returned no answer")
                    row.generated_answer, row.answer_source = answer, "generated"
                except Exception as exc:
                    row.errors["answer_generation"] = type(exc).__name__
            children = self.evaluator.evaluators if isinstance(self.evaluator, CompositeEvaluator) else [self.evaluator]
            if isinstance(self.evaluator, CompositeEvaluator):
                row.errors.update(self.evaluator.initialization_errors)
            eligible = []
            for position, evaluator in enumerate(children):
                name = f"{type(evaluator).__name__}:{position}"
                if isinstance(evaluator, NoneEvaluator):
                    row.skipped[name] = "evaluation_disabled"
                elif isinstance(evaluator, CustomEvaluator) and row.label_scope == "unlabelled":
                    row.skipped[name] = "missing_expected_evidence"
                elif isinstance(evaluator, CustomEvaluator) and not results:
                    # 真实空召回且有标准证据，是未命中，而非系统异常；不伪造候选。
                    if set(evaluator.metrics).intersection(row.metrics):
                        row.errors[name] = "duplicate_metric"
                    else:
                        row.metrics.update({metric: 0.0 for metric in evaluator.metrics})
                elif isinstance(evaluator, RagasEvaluator) and (not results or row.generated_answer is None):
                    row.skipped[name] = "missing_contexts" if not results else "missing_generated_answer"
                else:
                    eligible.append(evaluator)
            if eligible:
                active = CompositeEvaluator(evaluators=eligible)
                try:
                    values = active.evaluate(case.query, results, generated_answer=row.generated_answer,
                                             ground_truth={"ids": expected_ids})
                    if set(values).intersection(row.metrics):
                        raise ValueError("Duplicate batch metric")
                    row.metrics.update(values)
                except Exception as exc:
                    row.errors["evaluation"] = type(exc).__name__
                row.errors.update(active.last_errors)
            row.status = ("partial" if row.metrics and (row.errors or row.skipped) else "success" if row.metrics
                          else "error" if row.errors else "skipped")
        except Exception as exc:
            row.errors["retrieval_or_input"] = type(exc).__name__
            row.status = "error"
        row.elapsed_ms = (time.monotonic() - started) * 1000
        return row
