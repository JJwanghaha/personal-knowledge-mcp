"""在外部 HTTP 边界验证真实 LLM adapter，不请求模型服务。"""

import json
from dataclasses import replace

import httpx
import pytest

from src.core.settings import load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory

pytestmark = pytest.mark.unit


def test_openai_factory_sends_messages_and_returns_unified_response():
    """工厂选出的真实实现发出正确请求并转换响应。"""
    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider="openai", model="test-model", base_url=None))

    def server(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "test-model", "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.0, "max_tokens": 4096,
        }
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "你好，学习者"}}],
            "model": "test-model", "usage": {"total_tokens": 7},
        })

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create(settings, api_key="test-key", client=client)
        response = llm.chat([Message("user", "你好")])
    assert response.content == "你好，学习者"
    assert response.model == "test-model"
    assert response.usage == {"total_tokens": 7}


@pytest.mark.parametrize("provider", ["deepseek", "azure", "ollama"])
def test_other_llm_providers_route_to_their_own_protocol(provider):
    """不同后端保持公开接口，但地址、认证、请求体按协议变化。"""
    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider=provider, model="test-model", base_url=None))

    def server(request):
        body = json.loads(request.content)
        assert body["messages"] == [{"role": "user", "content": "你好"}]
        if provider == "azure":
            assert request.url.path == "/openai/deployments/test-deployment/chat/completions"
            assert request.url.params["api-version"] == "test-version"
            assert request.headers["api-key"] == "test-key"
            assert "authorization" not in request.headers
        elif provider == "ollama":
            assert request.url.path == "/api/chat"
            assert body["stream"] is False
            assert body["options"] == {"temperature": 0.0, "num_predict": 4096}
            return httpx.Response(200, json={
                "message": {"content": "本地回答"}, "model": "test-model",
                "prompt_eval_count": 3, "eval_count": 4,
            })
        else:
            assert request.url.host == "api.deepseek.com"
            assert request.url.path == "/chat/completions"
            assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "云端回答"}}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create(
            settings, api_key="test-key", client=client,
            endpoint="https://azure.example", api_version="test-version",
            deployment_name="test-deployment",
        )
        response = llm.chat([Message("user", "你好")])
    assert response.content == ("本地回答" if provider == "ollama" else "云端回答")
    assert response.model == "test-model"
    if provider == "ollama":
        assert response.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
