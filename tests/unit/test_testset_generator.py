"""H3：检查候选题输入边界、证据校验和未审核草稿约束。"""

import json
import socket

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.observability.evaluation.eval_runner import load_test_set
from src.observability.evaluation.testset_generator import LLMTestSetGenerator, save_draft


pytestmark = pytest.mark.unit
MODEL_KEYS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
)


class ScriptedLLM(BaseLLM):
    """仅替代文本 LLM 的系统边界，返回调用方指定的 JSON。"""

    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, messages, trace=None, **kwargs):
        self.calls.append(messages)
        return ChatResponse(self.content, "offline-question-model")


@pytest.fixture(autouse=True)
def isolate_external_access(monkeypatch):
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("候选题生成测试不得访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def selected_documents():
    return [
        {"source_path": "/tmp/已索引/摄取.md", "chunks": [
            {"id": "chunk-ingestion-1", "text": "已索引片段：SHA256 相同的文件默认跳过，--force 可以强制重新摄取。"},
            {"id": "chunk-ingestion-2", "text": "另一个片段记录完整性数据库中的成功处理状态。"},
        ]},
        {"source_path": "/tmp/已索引/检索.md", "chunks": [
            {"id": "chunk-retrieval-1", "text": "未选择的来源：RRF 按排名融合 Dense 和 BM25。"},
        ]},
    ]


def scripted_response(cases):
    return json.dumps({"test_cases": cases}, ensure_ascii=False)


def test_generator_sends_only_selected_chunk_text_and_derives_ids_from_literal_evidence():
    documents = selected_documents()
    llm = ScriptedLLM(scripted_response([{
        "query": "文件内容相同后如何强制重新摄取？",
        "source": "source-1",
        "evidence_quote": "--force 可以强制重新摄取",
        "reference_answer": "使用 --force。",
    }]))

    draft = LLMTestSetGenerator(settings=None, llm=llm).generate([documents[0]], count=1)

    assert draft["reviewed"] is False
    assert draft["generator_model"] == "offline-question-model"
    assert len(draft["test_cases"]) == 1
    case = draft["test_cases"][0]
    assert case["expected_chunk_ids"] == ["chunk-ingestion-1"]
    assert case["expected_sources"] == [documents[0]["source_path"]]
    assert case["evidence_quote"] == "--force 可以强制重新摄取"
    prompt = "\n".join(message.content for message in llm.calls[0])
    assert documents[0]["chunks"][0]["text"] in prompt
    assert documents[1]["chunks"][0]["text"] not in prompt
    assert "generated_answer" not in case
    assert "score" not in case


@pytest.mark.parametrize("count", [0, 51, True])
def test_question_count_is_rejected_before_constructing_or_calling_llm(count):
    llm = ScriptedLLM("{}")

    with pytest.raises(ValueError, match="between 1 and 50"):
        LLMTestSetGenerator(settings=None, llm=llm).generate(selected_documents(), count=count)

    assert llm.calls == []


@pytest.mark.parametrize("documents", [
    [],
    [None],
    [{"chunks": [{"id": "chunk", "text": "有效正文"}]}],
    [{"source_path": 12, "chunks": [{"id": "chunk", "text": "有效正文"}]}],
    [{"source_path": "/tmp/source.md", "chunks": "not-a-list"}],
    [{"source_path": "/tmp/source.md", "chunks": [{"text": "缺少真实 chunk id"}]}],
    [{"source_path": "/tmp/source.md", "chunks": [{"id": 7, "text": "chunk id 类型错误"}]}],
    [{"source_path": "/tmp/source.md", "chunks": [{"id": "chunk", "text": "  "}]}],
])
def test_malformed_selected_source_or_chunk_is_rejected_before_llm(documents):
    llm = ScriptedLLM("{}")

    with pytest.raises(ValueError):
        LLMTestSetGenerator(settings=None, llm=llm).generate(documents, count=1)

    assert llm.calls == []


def test_generation_input_over_40000_characters_is_rejected_before_llm():
    llm = ScriptedLLM("{}")
    documents = [{"source_path": "/tmp/large.md", "chunks": [
        {"id": "large-chunk", "text": "证" * 40001},
    ]}]

    with pytest.raises(ValueError, match="40000"):
        LLMTestSetGenerator(settings=None, llm=llm).generate(documents, count=1)

    assert llm.calls == []


@pytest.mark.parametrize(("cases", "count"), [
    ([], 1),
    ([{"query": "未知资料", "source": "source-9", "evidence_quote": "逐字证据", "reference_answer": "答案"}], 1),
    ([{"query": "错误证据", "source": "source-1", "evidence_quote": "不在片段中的引文", "reference_answer": "答案"}], 1),
    ([
        {"query": "重复问题", "source": "source-1", "evidence_quote": "SHA256 相同", "reference_answer": "答案甲"},
        {"query": " 重复问题 ", "source": "source-1", "evidence_quote": "SHA256 相同", "reference_answer": "答案乙"},
    ], 2),
])
def test_untrusted_llm_json_must_match_count_alias_uniqueness_and_literal_evidence(cases, count):
    llm = ScriptedLLM(scripted_response(cases))

    with pytest.raises(ValueError):
        LLMTestSetGenerator(settings=None, llm=llm).generate(selected_documents(), count=count)

    assert len(llm.calls) == 1


def test_saved_draft_cannot_overwrite_and_runner_rejects_reviewed_false(tmp_path):
    llm = ScriptedLLM(scripted_response([{
        "query": "怎样强制重新摄取？", "source": "source-1",
        "evidence_quote": "--force 可以强制重新摄取", "reference_answer": "使用强制选项。",
    }]))
    draft = LLMTestSetGenerator(settings=None, llm=llm).generate([selected_documents()[0]], count=1)

    path = save_draft(draft, tmp_path / "draft.json")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_draft(draft, path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Review"):
        load_test_set(path)
