"""Composite 通过公开工厂组合真实 Ragas collections 与 custom 指标。"""

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import socket

import pytest
from ragas._analytics import do_not_track
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from src.core.settings import Settings
from src.core.trace import TraceContext
from src.core.types import RetrievalResult
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


pytestmark = pytest.mark.integration

_MODEL_KEY_NAMES = (
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
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    do_not_track.cache_clear()
    for name in _MODEL_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)

    def deny_network(*args, **kwargs):
        raise AssertionError("Composite integration tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)


def settings_for(tmp_path, *, metrics, backends=("custom", "ragas")):
    """所有 provider、模型与存储路径均为离线测试值。"""
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
            "evaluation": {
                "enabled": True,
                "provider": "composite",
                "metrics": list(metrics),
                "backends": list(backends),
            },
            "observability": {
                "log_level": "INFO",
                "trace_enabled": False,
                "trace_file": str(tmp_path / "unused-trace.jsonl"),
                "structured_logging": False,
            },
        }
    )


def metric_responses(*, context_verdicts=(1, 0, 1), statement_verdicts=(1, 0)):
    return {
        "StatementGeneratorOutput": [{"statements": ["结构化事实甲", "结构化事实乙"]}],
        "NLIStatementOutput": [
            {
                "statements": [
                    {"statement": "结构化事实甲", "reason": "证据支持", "verdict": statement_verdicts[0]},
                    {"statement": "结构化事实乙", "reason": "证据不足", "verdict": statement_verdicts[1]},
                ]
            }
        ],
        "AnswerRelevanceOutput": [
            {"question": "候选问题甲", "noncommittal": 0},
            {"question": "候选问题乙", "noncommittal": 0},
            {"question": "候选问题丙", "noncommittal": 0},
        ],
        "ContextPrecisionOutput": [
            {"reason": f"本地判定 {index}", "verdict": verdict}
            for index, verdict in enumerate(context_verdicts)
        ],
    }


class ScriptedRagasLLM(InstructorBaseRagasLLM):
    """在 Ragas 的结构化 LLM 边界返回离线判定。"""

    def __init__(self, responses=None):
        self.responses = responses if responses is not None else metric_responses()
        self.prompts = []

    def _respond(self, prompt, response_model):
        self.prompts.append((str(prompt), response_model.__name__))
        choices = self.responses.get(response_model.__name__, [])
        if not choices:
            raise AssertionError(f"No scripted response for {response_model.__name__}")
        return response_model.model_validate(choices.pop(0))

    def generate(self, prompt, response_model):
        return self._respond(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return self._respond(prompt, response_model)

    def close(self):
        pass


class ScriptedRagasEmbedding(BaseRagasEmbedding):
    """返回可手算 cosine 的向量，不加载或调用 embedding SDK。"""

    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors

    def embed_text(self, text, **kwargs):
        return self.vectors[text]

    async def aembed_text(self, text, **kwargs):
        return self.vectors[text]

    def close(self):
        pass


def test_factory_routes_shared_inputs_to_real_custom_and_ragas_without_averaging(tmp_path):
    question = "问题全文_城市的中心在哪里"
    answer = "答案全文_中心位于河流交汇处"
    first_context = "证据全文甲_河流交汇处形成城市中心"
    second_context = "证据全文乙_附近还有一座远郊车站"
    chunks = [
        RetrievalResult("other", -0.25, first_context),
        RetrievalResult("target", -0.75, second_context),
    ]
    settings = settings_for(tmp_path, metrics=["hit_rate", "mrr", "faithfulness"])
    llm = ScriptedRagasLLM()
    trace = TraceContext()
    evaluator = EvaluatorFactory.create(settings, llm=llm)

    result = evaluator.evaluate(
        question,
        chunks,
        generated_answer=answer,
        ground_truth="target",
        trace=trace,
    )

    assert result == {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.5}
    prompt_text = "\n".join(prompt for prompt, _ in llm.prompts)
    for expected in (question, answer, first_context, second_context):
        assert expected in prompt_text
    backend_stages = [item for item in trace.stages if item["stage"] == "evaluation_backend"]
    assert [item["data"]["method"] for item in backend_stages] == ["custom", "ragas"]
    assert [item["data"]["status"] for item in backend_stages] == ["success", "success"]
    assert all(item["elapsed_ms"] >= 0 for item in backend_stages)
    composite_stage = trace.get_stage_data("evaluation_composite")
    assert composite_stage["status"] == "success"
    assert composite_stage["metrics"] == result
    assert trace.stages
    assert trace.finished_at is None


def test_empty_metrics_use_each_backend_default_without_cross_provider_extras(tmp_path):
    question = "默认问题_河流汇入何处"
    answer = "默认答案_河流汇入城市中心"
    chunks = [
        RetrievalResult("first", -0.1, "证据甲_河流进入城市中心"),
        RetrievalResult("target", -0.4, "证据乙_地理事实由此得到支持"),
        RetrievalResult("third", -0.8, "证据丙_提供额外背景信息"),
    ]
    vectors = {
        question: [1.0, 0.0],
        "候选问题甲": [1.0, 0.0],
        "候选问题乙": [0.0, 1.0],
        "候选问题丙": [1.0, 1.0],
    }
    settings = settings_for(tmp_path, metrics=[])
    evaluator = EvaluatorFactory.create(
        settings,
        llm=ScriptedRagasLLM(),
        embeddings=ScriptedRagasEmbedding(vectors),
    )

    result = evaluator.evaluate(question, chunks, generated_answer=answer, ground_truth="target")

    assert result == pytest.approx(
        {
            "hit_rate": 1.0,
            "mrr": 0.5,
            "faithfulness": 0.5,
            "answer_relevancy": (1.0 + 0.0 + 2**-0.5) / 3,
            "context_precision": 5 / 6,
        }
    )


def test_real_answer_relevancy_preserves_finite_negative_cosine(tmp_path):
    question = "负相关问题"
    vectors = {
        question: [1.0, 0.0],
        "候选问题甲": [-1.0, 0.0],
        "候选问题乙": [-1.0, 0.0],
        "候选问题丙": [-1.0, 0.0],
    }
    settings = settings_for(tmp_path, metrics=["answer_relevancy"], backends=("ragas",))
    evaluator = EvaluatorFactory.create(
        settings,
        llm=ScriptedRagasLLM(),
        embeddings=ScriptedRagasEmbedding(vectors),
    )

    result = evaluator.evaluate(question, ["证据"], generated_answer="答案")

    assert result == {"answer_relevancy": -1.0}
