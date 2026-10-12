"""I4：补齐真实存储和评分适配器的公开契约边界。"""

from contextlib import contextmanager
from copy import deepcopy
import socket
from types import SimpleNamespace

import pytest

from src.core.query_engine.reranker import CoreReranker, RerankConfig
from src.core.types import RetrievalResult
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.vector_store.chroma_store import ChromaStore


@pytest.fixture(autouse=True)
def offline_external_boundaries(monkeypatch):
    """不继承本机凭据，并在测试期间拒绝 Python socket 网络访问。"""
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")

    def reject_network(*args, **kwargs):
        raise AssertionError("Public contract tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)


@contextmanager
def _open_chroma(persist_directory, collection_name):
    settings = SimpleNamespace(
        vector_store=SimpleNamespace(
            persist_directory=str(persist_directory),
            collection_name=collection_name,
        )
    )
    store = ChromaStore(
        settings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    try:
        yield store
    finally:
        store.client.close()


@pytest.mark.integration
def test_chroma_metadata_read_and_delete_stay_within_the_selected_target(tmp_path):
    """复合 metadata 条件精确读取，删除来源路径时不影响其他来源或集合。"""
    persist_directory = tmp_path / "chroma"
    source_path = "/private/docs/guide.md"
    notes_records = [
        {
            "id": "guide-old",
            "vector": [1.0, 0.0],
            "metadata": {
                "text": "旧版本正文",
                "source_path": source_path,
                "doc_hash": "old-hash",
            },
        },
        {
            "id": "guide-current",
            "vector": [0.0, 1.0],
            "metadata": {
                "text": "当前版本正文",
                "source_path": source_path,
                "doc_hash": "current-hash",
            },
        },
        {
            "id": "another-source",
            "vector": [-1.0, 0.0],
            "metadata": {
                "text": "另一来源正文",
                "source_path": "/private/docs/another.md",
                "doc_hash": "current-hash",
            },
        },
    ]
    archive_record = {
        "id": "archive-guide",
        "vector": [1.0, 0.0],
        "metadata": {
            "text": "归档集合正文",
            "source_path": source_path,
            "doc_hash": "current-hash",
        },
    }

    with _open_chroma(persist_directory, "notes") as store:
        store.upsert(notes_records)
    with _open_chroma(persist_directory, "archive") as store:
        store.upsert([archive_record])

    with _open_chroma(persist_directory, "notes") as store:
        assert store.get_by_metadata(
            {"source_path": source_path, "doc_hash": "current-hash"}
        ) == [
            {
                "id": "guide-current",
                "text": "当前版本正文",
                "metadata": {
                    "text": "当前版本正文",
                    "source_path": source_path,
                    "doc_hash": "current-hash",
                },
            }
        ]

        with pytest.raises(ValueError, match="explicit metadata filters"):
            store.delete_by_metadata({})
        assert {item["id"] for item in store.get_by_metadata()} == {
            "guide-old",
            "guide-current",
            "another-source",
        }

        assert store.delete_by_metadata({"source_path": source_path}) == 2
        assert store.get_by_metadata({"source_path": source_path}) == []
        assert [item["id"] for item in store.get_by_metadata({"doc_hash": "current-hash"})] == [
            "another-source"
        ]

    with _open_chroma(persist_directory, "archive") as store:
        assert store.get_by_metadata({"source_path": source_path}) == [
            {
                "id": "archive-guide",
                "text": "归档集合正文",
                "metadata": archive_record["metadata"],
            }
        ]


class _FixedCrossEncoder:
    """替代真实模型 predict 接口，返回固定评分且不下载权重。"""

    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


@pytest.mark.unit
def test_cross_encoder_scores_convert_to_core_results_and_keep_original_scores():
    """Core 将 Cross-Encoder 输出转换为统一结果，并保留原始分数和来源。"""
    settings = SimpleNamespace(
        rerank=SimpleNamespace(enabled=True, provider="cross_encoder", top_k=2)
    )
    model = _FixedCrossEncoder([0.1, 0.9, 0.6])
    backend = CrossEncoderReranker(settings, model=model)
    core = CoreReranker(
        settings,
        backend,
        RerankConfig(enabled=True, top_k=2),
    )
    source = [
        RetrievalResult("a", 0.95, "候选甲正文", {"source_path": "a.md"}),
        RetrievalResult("b", 0.2, "候选乙正文", {"source_path": "b.md"}),
        RetrievalResult("c", 0.7, "候选丙正文", {"source_path": "c.md"}),
    ]
    original = deepcopy(source)

    result = core.rerank("目标问题", source)

    assert model.pairs == [
        ("目标问题", "候选甲正文"),
        ("目标问题", "候选乙正文"),
        ("目标问题", "候选丙正文"),
    ]
    assert result.reranker_type == "cross_encoder"
    assert result.used_fallback is False
    assert [item.chunk_id for item in result.results] == ["b", "c"]
    assert [item.score for item in result.results] == [0.9, 0.6]
    assert result.results[0] == RetrievalResult(
        "b",
        0.9,
        "候选乙正文",
        {
            "source_path": "b.md",
            "original_score": 0.2,
            "rerank_score": 0.9,
            "reranked": True,
        },
    )
    assert result.original_order == source
    assert source == original
