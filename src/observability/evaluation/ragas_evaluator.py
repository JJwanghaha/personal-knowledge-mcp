"""H1：沿用上游 Ragas collections 指标，通过现有 BaseEvaluator interface 调用。"""

from contextlib import contextmanager, ExitStack
from math import isfinite
import os
import time

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.embedding.embedding_factory import EmbeddingFactory

SUPPORTED_METRICS = frozenset({"faithfulness", "answer_relevancy", "context_precision"})
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
SCORE_RANGES = {"faithfulness": (0.0, 1.0), "context_precision": (0.0, 1.0),
                "answer_relevancy": (-1.0, 1.0)}


class RagasEvaluationError(RuntimeError):
    """评估运行失败，不把缺失或无效分数伪装为零分。"""


def _import_ragas():
    """仅选中后端才加载依赖，禁止 Ragas 使用统计。"""
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    try:
        import ragas
        from ragas._analytics import do_not_track
    except ImportError:
        raise ImportError("Ragas dependencies are missing or incompatible; install .[evaluation]") from None
    # 已导入的库可能缓存过旧值；本项目明确不发送评估使用统计。
    do_not_track.cache_clear()
    return ragas


class RagasEvaluator(BaseEvaluator):
    """本批使用无参考版本上下文精度；不生成答案，不把 ground_truth IDs 当文本。"""

    def __init__(self, settings=None, metrics=None, *, llm=None, embeddings=None):
        requested = metrics if metrics is not None else settings.evaluation.metrics if settings is not None else None
        if requested is not None and not isinstance(requested, (list, tuple)):
            raise ValueError("Ragas metrics must be a list or tuple")
        names = requested or DEFAULT_METRICS
        if any(not isinstance(name, str) for name in names):
            raise ValueError("Metric names must be strings")
        self.metric_names = tuple(dict.fromkeys(name.strip().lower() for name in names))
        if any(name not in SUPPORTED_METRICS for name in self.metric_names):
            raise ValueError("Ragas supports faithfulness, answer_relevancy, context_precision")
        _import_ragas()
        self.settings, self._llm, self._embeddings = settings, llm, embeddings

    def evaluate(self, query, retrieved_chunks, generated_answer=None, ground_truth=None, trace=None, **kwargs):
        """调用者提供真实待评答案；配置和构造不会因此自动调用模型。"""
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        if not isinstance(generated_answer, str) or not generated_answer.strip():
            raise ValueError("Ragas requires a nonempty generated_answer")
        contexts = self._extract_texts(retrieved_chunks)
        started = time.monotonic()
        try:
            from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
            with self._build_wrappers() as (llm, embeddings):
                metrics = {}
                for name in self.metric_names:
                    if name == "faithfulness":
                        result = Faithfulness(llm=llm).score(user_input=query, response=generated_answer, retrieved_contexts=contexts)
                    elif name == "answer_relevancy":
                        result = AnswerRelevancy(llm=llm, embeddings=embeddings).score(user_input=query, response=generated_answer)
                    else:
                        result = ContextPrecisionWithoutReference(llm=llm).score(user_input=query, response=generated_answer, retrieved_contexts=contexts)
                    if result.value is None or isinstance(result.value, (str, bool)):
                        raise ValueError("Ragas returned a missing or nonnumeric score")
                    value = float(result.value)
                    if not isfinite(value):
                        raise ValueError("Ragas returned a nonfinite score")
                    lower, upper = SCORE_RANGES[name]
                    if not lower <= value <= upper:
                        raise ValueError("Ragas returned a score outside the metric range")
                    metrics[name] = value
        except Exception as exc:
            if trace is not None:
                trace.record_stage("evaluation", {"method": "ragas", "status": "error", "error_type": type(exc).__name__},
                                   elapsed_ms=(time.monotonic() - started) * 1000)
            raise RagasEvaluationError(f"Ragas evaluation failed: {type(exc).__name__}") from None
        if trace is not None:
            trace.record_stage("evaluation", {"method": "ragas", "status": "success", "metrics": dict(metrics),
                "context_precision_variant": "without_reference", "context_count": len(contexts)},
                elapsed_ms=(time.monotonic() - started) * 1000)
        return metrics

    @staticmethod
    def _extract_texts(chunks):
        texts = []
        for chunk in chunks:
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict):
                text = next((chunk[key] for key in ("text", "content", "page_content") if chunk.get(key)), None)
            else:
                text = getattr(chunk, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every Ragas context must contain nonempty text, not just an ID")
            texts.append(text)
        return texts

    @contextmanager
    def _build_wrappers(self):
        """一次评估持有自建客户端；注入 Ragas llm/embeddings 由调用方负责生命周期。"""
        from src.observability.evaluation.ragas_models import GeminiRagasLLM, ProjectRagasEmbedding, ThreadedRagasLLM
        with ExitStack() as stack:
            llm, embeddings = self._llm, self._embeddings
            if llm is None:
                if self.settings is None:
                    raise ValueError("Settings or an injected Ragas LLM is required")
                config = self.settings.llm
                provider = config.provider.lower()
                if provider == "gemini":
                    from google import genai
                    from google.genai import types
                    key = os.environ.get("GEMINI_API_KEY")
                    if not key:
                        raise ValueError("Set GEMINI_API_KEY before real evaluation")
                    options = types.HttpOptions(base_url=config.base_url, timeout=int(config.timeout * 1000),
                                                retry_options=types.HttpRetryOptions(attempts=1))
                    client = stack.enter_context(genai.Client(api_key=key, vertexai=False, http_options=options))
                    llm = GeminiRagasLLM(config, client)
                elif provider in {"openai", "azure"}:
                    from openai import OpenAI, AzureOpenAI
                    from ragas.llms import llm_factory
                    key = os.environ.get("AZURE_OPENAI_API_KEY" if provider == "azure" else "OPENAI_API_KEY")
                    if not key:
                        raise ValueError("Set the selected provider's API key before evaluation")
                    if provider == "azure":
                        endpoint = config.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
                        version = config.api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
                        if not endpoint or not version:
                            raise ValueError("Azure requires an endpoint and explicit API version")
                        client = stack.enter_context(AzureOpenAI(api_key=key, azure_endpoint=endpoint,
                            api_version=version, timeout=config.timeout, max_retries=0))
                    else:
                        client = stack.enter_context(OpenAI(api_key=key, base_url=config.base_url, timeout=config.timeout, max_retries=0))
                    model = (config.deployment_name or config.model) if provider == "azure" else config.model
                    wrapped = llm_factory(model,
                                      provider="openai", adapter="instructor", client=client,
                                      temperature=config.temperature, max_tokens=config.max_tokens, max_retries=1)
                    llm = ThreadedRagasLLM(wrapped)
                else:
                    raise ValueError("H1 Ragas LLM supports gemini, openai, azure")
            if "answer_relevancy" in self.metric_names and embeddings is None:
                if self.settings is None:
                    raise ValueError("Answer relevancy requires Settings or injected Ragas embeddings")
                embeddings = ProjectRagasEmbedding(EmbeddingFactory.create(self.settings))
            yield llm, embeddings
