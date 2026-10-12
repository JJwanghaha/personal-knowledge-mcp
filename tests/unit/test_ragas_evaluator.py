"""H1 Ragas evaluator 与模型适配器的离线公开接口测试。"""

import asyncio
import builtins
import json
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from src.core.settings import load_settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.ragas_evaluator import RagasEvaluationError, RagasEvaluator


pytestmark = pytest.mark.unit

_SECRET_ENV_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    """每个测试移除真实凭据并阻止出站 socket。"""
    for name in _SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")

    def deny_network(*args, **kwargs):
        raise AssertionError("Ragas tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)


def settings_for(*, provider="ragas", enabled=True, metrics=None, llm_provider=None, **llm_changes):
    settings = load_settings()
    evaluation = replace(
        settings.evaluation,
        provider=provider,
        enabled=enabled,
        metrics=list(metrics if metrics is not None else settings.evaluation.metrics),
    )
    llm = settings.llm
    if llm_provider is not None:
        llm = replace(llm, provider=llm_provider)
    if llm_changes:
        llm = replace(llm, **llm_changes)
    return replace(settings, evaluation=evaluation, llm=llm)


class NeverCalledLLM:
    """用于确认输入校验在调用外部 LLM 之前完成。"""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, response_model):
        self.calls += 1
        raise AssertionError("LLM must not be called")

    async def agenerate(self, prompt, response_model):
        self.calls += 1
        raise AssertionError("LLM must not be called")


def test_disabled_factory_never_imports_ragas_or_model_sdks(monkeypatch):
    """关闭评估时不加载可选 Ragas、Google 或 OpenAI 模块。"""
    real_import = builtins.__import__

    def reject_optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"ragas", "google", "openai"}:
            raise AssertionError(f"unexpected optional import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_optional_import)
    evaluator = EvaluatorFactory.create(settings_for(enabled=False))

    assert evaluator.evaluate("问题", ["候选片段"]) == {}


def test_missing_ragas_import_is_reported_by_enabled_factory(monkeypatch):
    """模拟可选包导入失败，工厂保留清楚的安装提示且不泄漏底层异常。"""
    real_import = builtins.__import__

    def missing_ragas(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("simulated broken optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_ragas)
    with pytest.raises(ImportError, match=r"install \.\[evaluation\]") as exc_info:
        EvaluatorFactory.create(settings_for(metrics=["faithfulness"]))

    assert exc_info.value.__cause__ is None


def test_evaluator_construction_loads_ragas_without_creating_sdk_clients(monkeypatch):
    """选中可选实现只加载依赖；密钥和 SDK client 留到 evaluate。"""
    from google import genai
    import openai

    created = []

    def unexpected_client(*args, **kwargs):
        created.append((args, kwargs))
        raise AssertionError("construction must not create an SDK client")

    monkeypatch.setattr(genai, "Client", unexpected_client)
    monkeypatch.setattr(openai, "OpenAI", unexpected_client)
    monkeypatch.setattr(openai, "AzureOpenAI", unexpected_client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key-sentinel")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key-sentinel")
    evaluator_settings = settings_for(metrics=["faithfulness"], llm_provider="gemini")

    evaluator = RagasEvaluator(settings=evaluator_settings, metrics=["faithfulness"])

    assert evaluator.settings is evaluator_settings
    assert created == []


@pytest.mark.parametrize("unsupported", ["hit_rate", "mrr"])
def test_settings_metrics_are_not_silently_filtered_for_ragas(unsupported):
    """Ragas 不把 custom evaluator 指标当作可静默忽略的配置。"""
    configured = settings_for(metrics=["faithfulness", unsupported])

    with pytest.raises(ValueError, match="Ragas supports"):
        RagasEvaluator(settings=configured)


def test_factory_can_register_a_new_evaluator_provider():
    """新 provider 通过公开 Registry 接口进入统一工厂。"""

    class IsolatedFactory(EvaluatorFactory):
        _PROVIDERS = {}

    class RegisteredEvaluator(BaseEvaluator):
        def __init__(self, settings=None, **kwargs):
            self.settings = settings
            self.options = kwargs

        def evaluate(self, query, retrieved_chunks, generated_answer=None, ground_truth=None, trace=None, **kwargs):
            return {"registered": 1.0}

    IsolatedFactory.register_provider("  Local-Ragas-Test  ", RegisteredEvaluator)
    configured = settings_for(provider="local-ragas-test")
    evaluator = IsolatedFactory.create(configured, option="forwarded")

    assert isinstance(evaluator, RegisteredEvaluator)
    assert evaluator.settings is configured
    assert evaluator.options == {"option": "forwarded"}
    assert evaluator.evaluate("问题", ["证据"]) == {"registered": 1.0}
    assert IsolatedFactory.list_providers() == ["local-ragas-test"]


@pytest.mark.parametrize(
    "query,chunks,answer",
    [
        ("问题", ["证据"], None),
        ("问题", ["证据"], ""),
        ("问题", ["证据"], "   "),
        ("问题", [{}], "答案"),
        ("问题", [{"text": 42}], "答案"),
        ("问题", [object()], "答案"),
        ("问题", ["证据", "  "], "答案"),
        ("问题", "证据", "答案"),
        ("问题", [], "答案"),
    ],
)
def test_invalid_evaluation_input_is_rejected_before_model_call(query, chunks, answer):
    """缺答案、坏文本和非法候选在模型调用前明确失败。"""
    llm = NeverCalledLLM()
    evaluator = RagasEvaluator(metrics=["faithfulness"], llm=llm)

    with pytest.raises(ValueError):
        evaluator.evaluate(query, chunks, generated_answer=answer)

    assert llm.calls == 0


class ProjectEmbedding(BaseEmbedding):
    """返回可控向量的项目级 embedding 假实现。"""

    def __init__(self, dimension=3, vectors=None):
        self.dimension = dimension
        self.vectors = vectors
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.validate_texts(texts)
        self.calls.append((list(texts), dict(kwargs)))
        if self.vectors is not None:
            return self.vectors
        return [[float(index == 0) for index in range(self.dimension)] for _ in texts]

    def get_dimension(self):
        return self.dimension


def test_project_ragas_embedding_public_sync_and_async_methods_preserve_batches():
    """项目 embedding 的单条、批量及 async 入口统一保留批量契约。"""
    from src.observability.evaluation.ragas_models import ProjectRagasEmbedding

    project_embedding = ProjectEmbedding(dimension=3)
    adapter = ProjectRagasEmbedding(project_embedding)

    assert adapter.embed_text("单条") == [1.0, 0.0, 0.0]
    assert adapter.embed_texts(["甲", "乙"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert asyncio.run(adapter.aembed_text("异步单条")) == [1.0, 0.0, 0.0]
    assert asyncio.run(adapter.aembed_texts(["异步甲", "异步乙"])) == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert [call[0] for call in project_embedding.calls] == [
        ["单条"], ["甲", "乙"], ["异步单条"], ["异步甲", "异步乙"]
    ]


@pytest.mark.parametrize(
    "vectors,error",
    [
        ([], "output count does not match"),
        ([[1.0, 0.0]], "dimension does not match"),
        ([[True, 0.0, 0.0]], "finite numbers"),
        ([[float("nan"), 0.0, 0.0]], "finite numbers"),
    ],
)
def test_project_ragas_embedding_rejects_invalid_backend_vectors(vectors, error):
    """批量向量数量、维度、类型和有限数值遵守项目 embedding 校验。"""
    from src.observability.evaluation.ragas_models import ProjectRagasEmbedding

    adapter = ProjectRagasEmbedding(ProjectEmbedding(dimension=3, vectors=vectors))

    with pytest.raises(ValueError, match=error):
        adapter.embed_texts(["输入"])


@pytest.mark.parametrize("texts,error", [([], "cannot be empty"), (["有效", 7], "not a string"), ([" "], "empty or whitespace-only")])
def test_project_ragas_embedding_rejects_invalid_batch_inputs(texts, error):
    """Embedding 批次必须是非空且每项均为有效字符串。"""
    from src.observability.evaluation.ragas_models import ProjectRagasEmbedding

    with pytest.raises(ValueError, match=error):
        ProjectRagasEmbedding(ProjectEmbedding()).embed_texts(texts)


class TypedJudgement(BaseModel):
    verdict: int
    reason: str


class SyncStructuredLLM:
    """模拟同步 Instructor 的公开 generate 接口并记录运行线程。"""

    def __init__(self):
        self.thread_ids = []

    def generate(self, prompt, response_model):
        self.thread_ids.append((prompt, threading.get_ident()))
        return response_model.model_validate({"verdict": 1, "reason": "本地结构化结果"})


def test_threaded_ragas_llm_sync_and_async_public_methods_use_sync_generate_in_worker():
    """同步响应保持类型化，async 路径把同步 Instructor 调用放到工作线程。"""
    from src.observability.evaluation.ragas_models import ThreadedRagasLLM

    wrapped = SyncStructuredLLM()
    adapter = ThreadedRagasLLM(wrapped)
    sync_result = adapter.generate("同步 prompt", TypedJudgement)
    caller_thread = threading.get_ident()

    async_result = asyncio.run(adapter.agenerate("异步 prompt", TypedJudgement))

    assert isinstance(sync_result, TypedJudgement)
    assert isinstance(async_result, TypedJudgement)
    assert sync_result.verdict == async_result.verdict == 1
    assert wrapped.thread_ids[0][0] == "同步 prompt"
    assert wrapped.thread_ids[1][0] == "异步 prompt"
    assert wrapped.thread_ids[0][1] == caller_thread
    assert wrapped.thread_ids[1][1] != caller_thread


def test_gemini_ragas_llm_uses_native_json_schema_and_parses_typed_response():
    """Gemini 原生 generateContent 接收 Pydantic JSON Schema 并返回类型化对象。"""
    from src.observability.evaluation.ragas_models import GeminiRagasLLM

    class FakeModels:
        def __init__(self):
            self.call = None

        def generate_content(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(text=json.dumps({"verdict": 1, "reason": "SDK fake"}))

    from google.genai import types

    models = FakeModels()
    settings = settings_for(llm_provider="gemini", model="gemini-offline", timeout=60.0)
    result = GeminiRagasLLM(settings.llm, SimpleNamespace(models=models)).generate(
        "prompt 全文", TypedJudgement
    )

    assert isinstance(result, TypedJudgement)
    assert result.verdict == 1
    assert models.call["model"] == "gemini-offline"
    assert models.call["contents"] == "prompt 全文"
    assert isinstance(models.call["config"], types.GenerateContentConfig)
    assert models.call["config"].response_mime_type == "application/json"
    assert models.call["config"].response_json_schema == TypedJudgement.model_json_schema()


def test_gemini_ragas_llm_async_method_uses_sync_sdk_on_worker_thread():
    """Gemini 的 async Ragas 接口将同步原生 SDK 调用移入工作线程。"""
    from src.observability.evaluation.ragas_models import GeminiRagasLLM

    class FakeModels:
        def __init__(self):
            self.thread_ids = []

        def generate_content(self, **kwargs):
            self.thread_ids.append(threading.get_ident())
            return SimpleNamespace(text=json.dumps({"verdict": 1, "reason": "async SDK fake"}))

    models = FakeModels()
    adapter = GeminiRagasLLM(
        settings_for(llm_provider="gemini").llm,
        SimpleNamespace(models=models),
    )
    caller_thread = threading.get_ident()

    result = asyncio.run(adapter.agenerate("async prompt", TypedJudgement))

    assert isinstance(result, TypedJudgement)
    assert result.reason == "async SDK fake"
    assert len(models.thread_ids) == 1
    assert models.thread_ids[0] != caller_thread


@pytest.mark.parametrize("failure", ["timeout", "no_response", "non_json"])
def test_gemini_ragas_llm_rejects_timeout_missing_or_unstructured_response(failure):
    """超时、空响应和无效 JSON 不伪装成类型化判定。"""
    from src.observability.evaluation.ragas_models import GeminiRagasLLM

    class FakeModels:
        def generate_content(self, **kwargs):
            if failure == "timeout":
                raise TimeoutError("synthetic timeout")
            if failure == "no_response":
                return SimpleNamespace(text=None)
            return SimpleNamespace(text="not-json")

    settings = settings_for(llm_provider="gemini")
    adapter = GeminiRagasLLM(settings.llm, SimpleNamespace(models=FakeModels()))
    expected_error = TimeoutError if failure == "timeout" else ValueError if failure == "no_response" else ValidationError

    with pytest.raises(expected_error):
        adapter.generate("offline prompt", TypedJudgement)


@pytest.mark.parametrize(
    "provider,environment,changes",
    [
        ("openai", {"AZURE_OPENAI_API_KEY": "azure-only-sentinel"}, {}),
        ("azure", {"OPENAI_API_KEY": "openai-only-sentinel"}, {}),
        (
            "azure",
            {
                "AZURE_OPENAI_API_KEY": "azure-test-sentinel",
                "OPENAI_BASE_URL": "https://unrelated.invalid",
                "OPENAI_API_VERSION": "unrelated-version",
            },
            {"base_url": None, "api_version": None},
        ),
        (
            "azure",
            {
                "AZURE_OPENAI_API_KEY": "azure-test-sentinel",
                "AZURE_OPENAI_ENDPOINT": "https://azure.invalid",
                "OPENAI_API_VERSION": "unrelated-version",
            },
            {"base_url": None, "api_version": None},
        ),
    ],
)
def test_openai_and_azure_missing_credentials_or_endpoint_never_fallback(
    monkeypatch, provider, environment, changes
):
    """OpenAI/Azure 只读取所选供应商的凭据和 Azure 显式端点、版本。"""
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    configured = settings_for(
        metrics=["faithfulness"], llm_provider=provider, **changes
    )
    evaluator = RagasEvaluator(settings=configured, metrics=["faithfulness"])

    with pytest.raises(RagasEvaluationError, match="Ragas evaluation failed: ValueError") as exc_info:
        evaluator.evaluate("问题", ["事实证据"], generated_answer="给定答案")

    assert exc_info.value.__cause__ is None
