"""验证本地向量服务协议，但不需要安装 Ollama 或模型。"""

import json
from dataclasses import replace

import httpx
import pytest

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory

pytestmark = pytest.mark.unit


def test_ollama_batches_texts_without_silent_truncation():
    """新版 /api/embed 一次发送多个文本，超长文本不得静默截断。"""
    settings = load_settings()
    settings = replace(settings, embedding=replace(settings.embedding, provider="ollama", model="test-embed", dimensions=2, base_url=None))

    def server(request):
        assert request.url.path == "/api/embed"
        assert json.loads(request.content) == {
            "model": "test-embed", "input": ["甲", "乙"], "truncate": False,
        }
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        embedding = EmbeddingFactory.create(settings, client=client)
        assert embedding.embed(["甲", "乙"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert embedding.get_dimension() == 2
