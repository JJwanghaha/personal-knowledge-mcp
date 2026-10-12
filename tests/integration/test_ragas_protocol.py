"""通过 Ragas 0.4.3 collections 与公开模型协议验证 H1 离线链路。"""

import importlib.metadata
import json
import os
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
for _key_name in (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
):
    os.environ.pop(_key_name, None)

import ragas
from ragas._analytics import do_not_track
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from src.core.settings import load_settings
from src.core.trace import TraceContext
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.observability.evaluation.ragas_evaluator import RagasEvaluationError, RagasEvaluator
from src.observability.evaluation.ragas_models import ThreadedRagasLLM


pytestmark = pytest.mark.integration

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
    """移除真实凭据、关闭 Ragas telemetry 并拒绝真实网络连接。"""
    for name in _SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    do_not_track.cache_clear()

    def deny_network(*args, **kwargs):
        raise AssertionError("Ragas integration tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)


def settings_for(*, metrics, llm_provider="gemini", **llm_changes):
    settings = load_settings()
    evaluation = replace(
        settings.evaluation,
        enabled=True,
        provider="ragas",
        metrics=list(metrics),
    )
    llm = replace(settings.llm, provider=llm_provider, **llm_changes)
    return replace(settings, evaluation=evaluation, llm=llm)


def metric_responses(*, context_verdicts=(1, 0, 1), statement_verdicts=(1, 0)):
    return {
        "StatementGeneratorOutput": [
            {"statements": ["结构化事实甲", "结构化事实乙"]}
        ],
        "NLIStatementOutput": [
            {
                "statements": [
                    {
                        "statement": "结构化事实甲",
                        "reason": "证据支持",
                        "verdict": statement_verdicts[0],
                    },
                    {
                        "statement": "结构化事实乙",
                        "reason": "证据不足",
                        "verdict": statement_verdicts[1],
                    },
                ]
            }
        ],
        "AnswerRelevanceOutput": [
            {"question": "生成问题甲", "noncommittal": 0},
            {"question": "生成问题乙", "noncommittal": 0},
            {"question": "生成问题丙", "noncommittal": 0},
        ],
        "ContextPrecisionOutput": [
            {"reason": f"本地判定 {index}", "verdict": verdict}
            for index, verdict in enumerate(context_verdicts)
        ],
    }


class ScriptedRagasLLM(InstructorBaseRagasLLM):
    """实现 Ragas 现代类型化接口，模拟外部 LLM 的结构化响应。"""

    def __init__(self, responses=None, error=None):
        self.responses = responses if responses is not None else metric_responses()
        self.error = error
        self.prompts = []
        self.thread_ids = []
        self.close_count = 0

    def _respond(self, prompt, response_model):
        self.prompts.append((prompt, response_model.__name__))
        self.thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        choices = self.responses.get(response_model.__name__, [])
        if not choices:
            raise AssertionError(f"No scripted response for {response_model.__name__}")
        return response_model.model_validate(choices.pop(0))

    def generate(self, prompt, response_model):
        return self._respond(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return self._respond(prompt, response_model)

    def close(self):
        self.close_count += 1


class ScriptedSyncInstructor:
    """模拟同步 Instructor SDK 的 generate 边界。"""

    def __init__(self, responses=None):
        self.responses = responses if responses is not None else metric_responses()
        self.thread_ids = []
        self.prompts = []

    def generate(self, prompt, response_model):
        self.thread_ids.append(threading.get_ident())
        self.prompts.append((prompt, response_model.__name__))
        choices = self.responses.get(response_model.__name__, [])
        if not choices:
            raise AssertionError(f"No scripted response for {response_model.__name__}")
        return response_model.model_validate(choices.pop(0))


class ScriptedRagasEmbedding(BaseRagasEmbedding):
    """向 Ragas 返回可手算余弦相似度的向量。"""

    def __init__(self, vectors=None):
        super().__init__()
        self.vectors = vectors or {
            "问题全文_北方城市的首都": [1.0, 0.0],
            "生成问题甲": [1.0, 0.0],
            "生成问题乙": [0.0, 1.0],
            "生成问题丙": [1.0, 1.0],
        }
        self.calls = []
        self.close_count = 0

    def embed_text(self, text, **kwargs):
        self.calls.append(("sync", text))
        return self.vectors[text]

    async def aembed_text(self, text, **kwargs):
        self.calls.append(("async", text))
        return self.vectors[text]

    def close(self):
        self.close_count += 1


def test_installed_ragas_043_exposes_the_real_collection_metrics():
    """焦点测试首先确认真实安装版及现代 collections 导入成功。"""
    assert importlib.metadata.version("ragas") == "0.4.3"
    assert Faithfulness.__name__ == "Faithfulness"
    assert AnswerRelevancy.__name__ == "AnswerRelevancy"
    assert ContextPrecisionWithoutReference.__name__ == "ContextPrecisionWithoutReference"


def test_ragas_constructor_disables_cached_telemetry_state(monkeypatch):
    """构造时将 telemetry 环境变量设为 true，并清除 Ragas 已缓存的旧值。"""
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "false")
    do_not_track.cache_clear()
    assert do_not_track() is False

    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "false")
    RagasEvaluator(metrics=["faithfulness"], llm=ScriptedRagasLLM())

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert do_not_track() is True


def test_three_real_collections_metrics_use_full_inputs_and_no_reference():
    """三项真实 Ragas 算法收到全文，context precision 采用无参考实现。"""
    llm = ScriptedRagasLLM()
    embeddings = ScriptedRagasEmbedding()
    trace = TraceContext()
    query = "问题全文_北方城市的首都"
    answer = "答案全文_这座城市承载了完整的待评回答内容"
    contexts = [
        "证据全文甲_城市和国家的关系在这里完整陈述",
        "证据全文乙_这条独立材料只提供旁支背景",
        "证据全文丙_另一段完整事实支持答案的第二部分",
    ]
    chunks = [
        {"text": contexts[0]},
        SimpleNamespace(text=contexts[1]),
        {"page_content": contexts[2]},
    ]
    evaluator = RagasEvaluator(
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        llm=llm,
        embeddings=embeddings,
    )

    result = evaluator.evaluate(query, chunks, generated_answer=answer, trace=trace)

    # Faithfulness: 1 of 2 claims supported. Answer relevancy: mean of 1, 0, and sqrt(1/2).
    # Context precision: useful ranks [1, 3] yield average precision 5/6.
    assert result["faithfulness"] == pytest.approx(0.5)
    assert result["answer_relevancy"] == pytest.approx((1 + 0 + 2**-0.5) / 3)
    assert result["context_precision"] == pytest.approx(5 / 6, abs=1e-9)
    prompt_text = "\n".join(prompt for prompt, _ in llm.prompts)
    for expected_text in [query, answer, *contexts]:
        assert expected_text in prompt_text
    assert {model_name for _, model_name in llm.prompts} == {
        "StatementGeneratorOutput",
        "NLIStatementOutput",
        "AnswerRelevanceOutput",
        "ContextPrecisionOutput",
    }
    assert {text for _, text in embeddings.calls} == {
        query,
        "生成问题甲",
        "生成问题乙",
        "生成问题丙",
    }
    evaluation_trace = trace.get_stage_data("evaluation")
    assert evaluation_trace["status"] == "success"
    assert evaluation_trace["metrics"] == pytest.approx(result)
    assert evaluation_trace["context_precision_variant"] == "without_reference"
    assert evaluation_trace["context_count"] == 3
    assert llm.close_count == 0
    assert embeddings.close_count == 0


def test_metrics_without_answer_relevancy_do_not_require_project_embeddings():
    """Faithfulness 与无参考 Context Precision 不创建或要求 Embedding。"""
    base = settings_for(metrics=["faithfulness", "context_precision"])
    settings = replace(
        base,
        embedding=replace(base.embedding, provider="not-registered-for-this-test"),
    )
    llm = ScriptedRagasLLM()

    result = RagasEvaluator(settings=settings, llm=llm).evaluate(
        "问题", ["第一段证据", "第二段证据", "第三段证据"], generated_answer="给定答案"
    )

    assert set(result) == {"faithfulness", "context_precision"}


def test_answer_relevancy_uses_dimensions_from_project_settings(monkeypatch):
    """AnswerRelevancy 才创建项目 Embedding，并沿用配置的 768 维度。"""
    provider = "offline-ragas-embedding-768"
    created = []

    class SettingsEmbedding(BaseEmbedding):
        def __init__(self, settings=None, **kwargs):
            self.settings = settings
            self.dimension = settings.embedding.dimensions
            self.seen_texts = []
            created.append(self)

        def embed(self, texts, trace=None, **kwargs):
            self.validate_texts(texts)
            self.seen_texts.extend(texts)
            return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]

        def get_dimension(self):
            return self.dimension

    monkeypatch.setitem(EmbeddingFactory._PROVIDERS, provider, SettingsEmbedding)
    EmbeddingFactory.register_provider(provider, SettingsEmbedding)
    base = settings_for(metrics=["answer_relevancy"])
    settings = replace(
        base,
        embedding=replace(base.embedding, provider=provider, dimensions=768),
    )
    llm = ScriptedRagasLLM(metric_responses(context_verdicts=()))

    result = RagasEvaluator(settings=settings, llm=llm).evaluate(
        "问题全文_北方城市的首都", ["证据"], generated_answer="给定答案"
    )

    assert result["answer_relevancy"] == pytest.approx(1.0)
    assert created[0].settings is settings
    assert created[0].dimension == 768
    assert len(created[0].seen_texts) == 4


def test_real_faithfulness_score_uses_threaded_sync_instructor():
    """Ragas.score 的内部 async 生成由 ThreadedRagasLLM 调度同步客户端线程。"""
    sync_client = ScriptedSyncInstructor()
    caller_thread = threading.get_ident()
    evaluator = RagasEvaluator(
        metrics=["faithfulness"], llm=ThreadedRagasLLM(sync_client)
    )

    result = evaluator.evaluate(
        "问题", ["原始证据全文"], generated_answer="使用者提供的完整答案"
    )

    assert result == {"faithfulness": 0.5}
    assert len(sync_client.thread_ids) == 2
    assert all(thread_id != caller_thread for thread_id in sync_client.thread_ids)
    assert {name for _, name in sync_client.prompts} == {
        "StatementGeneratorOutput",
        "NLIStatementOutput",
    }


def test_empty_faithfulness_judgment_fails_on_nonfinite_ragas_score():
    """真实 Ragas 算法产生 NaN 时显式失败并留下不含分数的错误 trace。"""
    llm = ScriptedRagasLLM(
        {"StatementGeneratorOutput": [{"statements": []}]}
    )
    trace = TraceContext()

    with pytest.raises(RagasEvaluationError, match="Ragas evaluation failed: ValueError"):
        RagasEvaluator(metrics=["faithfulness"], llm=llm).evaluate(
            "问题", ["证据"], generated_answer="给定答案", trace=trace
        )

    failure = trace.get_stage_data("evaluation")
    assert failure["status"] == "error"
    assert failure["error_type"] == "ValueError"
    assert "nan" not in repr(failure).lower()


def test_out_of_range_real_context_precision_score_is_rejected():
    """真实 collection 算法遇到非法 verdict 得出的越界分数必须显式失败。"""
    llm = ScriptedRagasLLM(metric_responses(context_verdicts=(2,)))

    try:
        result = RagasEvaluator(metrics=["context_precision"], llm=llm).evaluate(
            "问题", ["证据"], generated_answer="给定答案"
        )
    except RagasEvaluationError as exc:
        assert str(exc) == "Ragas evaluation failed: ValueError"
    else:
        pytest.fail(f"越界分数未被拒绝，实际返回：{result!r}")


def test_raw_llm_exception_and_key_sentinel_do_not_reach_error_or_trace():
    """原始 SDK 错误与合成 key sentinel 只留下异常类型，且异常不设 cause。"""
    sentinel = "sk-test-only-api-key-sentinel"
    llm = ScriptedRagasLLM(error=RuntimeError(f"request failed with api key {sentinel}"))
    trace = TraceContext()

    with pytest.raises(RagasEvaluationError) as exc_info:
        RagasEvaluator(metrics=["faithfulness"], llm=llm).evaluate(
            "问题", ["证据"], generated_answer="给定答案", trace=trace
        )

    assert str(exc_info.value) == "Ragas evaluation failed: RuntimeError"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert sentinel not in str(exc_info.value)
    trace_data = repr(trace.get_stage_data("evaluation"))
    assert sentinel not in trace_data
    assert "request failed" not in trace_data
    assert trace.get_stage_data("evaluation")["error_type"] == "RuntimeError"


def test_gemini_evaluate_uses_settings_and_closes_its_owned_client(monkeypatch):
    """Ragas 自建 Gemini client 使用 Settings 模型/60 秒超时并由 evaluate 关闭。"""
    from google import genai

    class FakeModels:
        def __init__(self):
            self.models = []

        def generate_content(self, *, model, contents, config):
            self.models.append(model)
            name = config.response_json_schema["title"]
            if name == "StatementGeneratorOutput":
                value = {"statements": ["由给定答案抽取的事实"]}
            elif name == "NLIStatementOutput":
                value = {
                    "statements": [
                        {
                            "statement": "由给定答案抽取的事实",
                            "reason": "本地 SDK fake 的结构化结果",
                            "verdict": 1,
                        }
                    ]
                }
            else:
                raise AssertionError(f"Unexpected schema: {name}")
            return SimpleNamespace(text=json.dumps(value))

    class FakeClient:
        def __init__(self, *, api_key, vertexai, http_options):
            self.api_key = api_key
            self.vertexai = vertexai
            self.http_options = http_options
            self.models = FakeModels()
            self.exit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.exit_count += 1

    clients = []

    def create_client(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(genai, "Client", create_client)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-only-sentinel")
    base = settings_for(
        metrics=["faithfulness"],
        llm_provider="gemini",
        model="gemini-offline-model",
        timeout=60.0,
        base_url="https://gemini.invalid",
    )
    settings = replace(
        base,
        embedding=replace(base.embedding, provider="not-used-for-faithfulness"),
    )

    result = RagasEvaluator(settings=settings).evaluate(
        "问题", ["完整证据"], generated_answer="由调用者提供的回答"
    )

    assert result == {"faithfulness": 1.0}
    assert len(clients) == 1
    assert clients[0].api_key == "gemini-test-only-sentinel"
    assert clients[0].vertexai is False
    assert clients[0].http_options.base_url == "https://gemini.invalid"
    assert clients[0].http_options.timeout == 60_000
    assert clients[0].models.models == ["gemini-offline-model"] * 2
    assert clients[0].exit_count == 1


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_openai_and_azure_upstream_factory_work_offline_through_thread_adapter(
    monkeypatch, provider
):
    """OpenAI/Azure 正常走 Ragas upstream Instructor 与同步 Threaded adapter。"""
    import httpx
    import openai

    requests = []

    def mock_server(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if len(requests) % 2 == 1:
            response_content = json.dumps({"statements": ["已知结构化事实"]})
        else:
            response_content = json.dumps(
                {
                    "statements": [
                        {
                            "statement": "已知结构化事实",
                            "reason": "本地 HTTP mock 的证据判断",
                            "verdict": 1,
                        }
                    ]
                }
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 1,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original_openai = openai.OpenAI
    original_azure = openai.AzureOpenAI
    sdk_clients = []

    with httpx.Client(transport=httpx.MockTransport(mock_server)) as mock_http_client:
        if provider == "openai":
            class OfflineOpenAI(original_openai):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, http_client=mock_http_client)
                    sdk_clients.append(self)

            monkeypatch.setattr(openai, "OpenAI", OfflineOpenAI)
            monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-only")
            llm_settings = {
                "model": "offline-model",
                "base_url": "https://openai.invalid/v1",
            }
        else:
            class OfflineAzureOpenAI(original_azure):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, http_client=mock_http_client)
                    sdk_clients.append(self)

            monkeypatch.setattr(openai, "AzureOpenAI", OfflineAzureOpenAI)
            monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-only")
            llm_settings = {
                "model": "unused-base-model",
                "deployment_name": "offline-deployment",
                "base_url": "https://azure.invalid",
                "api_version": "2025-01-01-preview",
            }
        configured = settings_for(
            metrics=["faithfulness"], llm_provider=provider, **llm_settings
        )

        result = RagasEvaluator(settings=configured).evaluate(
            "问题", ["真实评估证据"], generated_answer="真实待评答案"
        )

    assert result == {"faithfulness": 1.0}
    assert len(requests) == 2
    assert all("chat/completions" in request.url.path for request, _ in requests)
    assert len(sdk_clients) == 1
    assert sdk_clients[0].is_closed
    if provider == "azure":
        assert all("offline-deployment" in request.url.path for request, _ in requests)
        assert all(
            request.url.params.get("api-version") == "2025-01-01-preview"
            for request, _ in requests
        )
