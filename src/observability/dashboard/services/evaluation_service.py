"""H3/H4：CLI 和后台共用评估入口、报告保存与显式候选题生成。"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import uuid

from src.core.query_engine.query_service import create_query_components
from src.core.settings import resolve_path
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner, load_test_set, write_report
from src.observability.evaluation.ragas_evaluator import DEFAULT_METRICS
from src.observability.evaluation.testset_generator import LLMAnswerGenerator, LLMTestSetGenerator, save_draft
from src.observability.dashboard.services.data_service import DataService
from src.mcp_server.tools._storage import configured_storage, validate_collection

DEFAULT_TEST_SET = resolve_path("tests/fixtures/golden_test_set.json")


class EvaluationService:
    def __init__(self, settings=None, *, data_dir=None, embedding_client=None, llm=None):
        self.settings, self.data_root = configured_storage(settings, data_dir)
        self._data_dir_override = self.data_root if data_dir is not None else None
        self.embedding_client, self.llm = embedding_client, llm
        self.reports_root = self.data_root / "evaluations" / "reports"

    def run(self, test_set_path=DEFAULT_TEST_SET, collection="default", *, backend="custom", top_k=5,
            generate_answers=False, no_rerank=False, evaluator=None):
        validate_collection(collection)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        load_test_set(test_set_path)  # 草稿/格式检查先于模型和存储创建。
        if backend not in {"custom", "ragas", "composite"}:
            raise ValueError("Unsupported evaluation backend")
        metrics = ["hit_rate", "mrr"] if backend == "custom" else list(DEFAULT_METRICS)
        if backend == "composite":
            metrics = ["hit_rate", "mrr", *DEFAULT_METRICS]
        settings = replace(self.settings, evaluation=replace(self.settings.evaluation, enabled=True, provider=backend,
                            metrics=metrics, backends=["custom", "ragas"] if backend == "composite" else None))
        if no_rerank:
            settings = replace(settings, rerank=replace(settings.rerank, enabled=False, provider="none"))
        components = create_query_components(settings, collection, self.data_root / "db" / "bm25", embedding_client=self.embedding_client)
        try:
            selected = evaluator if evaluator is not None else EvaluatorFactory.create(settings)
            answer_generator = LLMAnswerGenerator(settings, self.llm) if generate_answers else None
            report = EvalRunner(settings, components.hybrid_search, selected, answer_generator=answer_generator,
                                reranker=components.reranker).run(test_set_path, top_k, collection)
            corpus = components.vector_store.get_by_metadata()
            report.configuration["corpus_fingerprint"] = sha256(json.dumps(sorted((c["id"], c["metadata"].get("doc_hash", ""))
                for c in corpus), ensure_ascii=False).encode("utf-8")).hexdigest()
            report.configuration["generate_answers"] = generate_answers
            return report
        finally:
            components.close()

    def save_report(self, report):
        return write_report(report, self.reports_root / f"{uuid.uuid4().hex}.json")

    def list_reports(self, limit=30):
        if not self.reports_root.is_dir():
            return []
        reports = []
        for path in sorted(self.reports_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            if not path.resolve().is_relative_to(self.reports_root.resolve()):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and "aggregate_metrics" in value and "test_set_sha256" in value:
                    reports.append({"path": str(path), "report": value})
            except (OSError, ValueError):
                continue
        return reports

    def generate_draft(self, collection, source_paths, *, count=10):
        data = DataService(self.settings, data_dir=self._data_dir_override)
        indexed = {doc["source_path"]: doc for doc in data.list_documents(collection)}
        if not source_paths or any(source not in indexed for source in source_paths):
            raise ValueError("Select existing indexed documents in this collection")
        documents = [{"source_path": source, "chunks": data.get_chunks(indexed[source]["source_hash"], collection, source_path=source)}
                     for source in source_paths]
        return LLMTestSetGenerator(self.settings, self.llm).generate(documents, count)

    def save_draft(self, draft):
        return save_draft(draft, self.data_root / "evaluations" / "drafts" / f"{uuid.uuid4().hex}.json")
