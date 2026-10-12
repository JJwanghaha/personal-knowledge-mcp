"""通过真实 Google SDK 与模拟 HTTP 验证 Gemini 图文协议。"""

import base64
import io
import json
from dataclasses import replace

import httpx
import pytest
from google import genai
from google.genai import types
from PIL import Image

from src.core.settings import VisionLLMSettings, load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory

pytestmark = pytest.mark.unit


def picture_bytes(size=(40, 20)):
    """只生成固定颜色的测试图，不使用真实知识库图片。"""
    output = io.BytesIO()
    with Image.new("RGB", size, "blue") as image:
        image.save(output, format="PNG")
    return output.getvalue()


def vision_settings():
    """固定 Gemini 测试配置，不继承用户当前 Gateway 的地址。"""
    settings = load_settings()
    return replace(settings, llm=replace(settings.llm, provider="gemini", base_url=None), vision_llm=VisionLLMSettings(
        enabled=False, provider="gemini", model="test-gemini-vision", max_image_size=100,
    ))


def test_gemini_vision_encodes_image_and_preserves_history():
    data = picture_bytes()
    received = []

    def server(request):
        assert request.url.path == "/v1beta/models/test-gemini-vision:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        received.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "模拟图片描述"}]}}],
            "modelVersion": "test-gemini-vision",
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
        })

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(httpx_client=http_client)) as client:
            llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
            result = llm.chat_with_image("描述图片", ImageInput(data=data), messages=[
                Message("system", "用中文"), Message("user", "上一轮问题"), Message("assistant", "上一轮回答"),
            ], temperature=0.2, max_tokens=80)

    body = received[0]
    assert body["systemInstruction"]["parts"] == [{"text": "用中文"}]
    assert body["contents"][:2] == [
        {"role": "user", "parts": [{"text": "上一轮问题"}]},
        {"role": "model", "parts": [{"text": "上一轮回答"}]},
    ]
    current = body["contents"][2]
    assert current["role"] == "user"
    assert len(current["parts"]) == 2
    assert current["parts"][0] == {"text": "描述图片"}
    inline = current["parts"][1]["inlineData"]
    assert inline["mime_type"] == "image/png"
    assert base64.urlsafe_b64decode(inline["data"]) == data
    assert body["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 80}
    assert result.content == "模拟图片描述"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert result.model == "test-gemini-vision"


def call_with_server(server, image=None, **kwargs):
    """只替换外部 HTTP，真实 SDK 序列化和响应解析继续运行。"""
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(
            httpx_client=http_client, retry_options=types.HttpRetryOptions(attempts=1),
        )) as client:
            llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
            result = llm.chat_with_image("描述图片", image or ImageInput(data=picture_bytes()), **kwargs)
            assert not http_client.is_closed
            return result


@pytest.mark.parametrize("carrier", ["path", "data", "base64"])
def test_gemini_image_carriers_resize_only_path_and_bytes(tmp_path, carrier):
    data = picture_bytes((400, 200))
    path = tmp_path / "large.png"
    path.write_bytes(data)
    inputs = {"path": path, "data": data, "base64": base64.b64encode(data).decode("ascii")}

    def server(request):
        blob = json.loads(request.content)["contents"][-1]["parts"][1]["inlineData"]
        assert blob["mime_type"] == "image/png"
        with Image.open(io.BytesIO(base64.urlsafe_b64decode(blob["data"]))) as image:
            assert image.size == ((400, 200) if carrier == "base64" else (100, 50))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "描述"}]}}]})

    assert call_with_server(server, ImageInput(**{carrier: inputs[carrier]})).content == "描述"
    assert path.read_bytes() == data


@pytest.mark.parametrize("status", [401, 429, 503])
def test_gemini_reports_http_failure_without_raw_details(status):
    from src.libs.llm.gemini_vision_llm import GeminiVisionLLMError

    def server(request):
        return httpx.Response(status, json={"error": {"code": status, "message": "private-detail"}})

    with pytest.raises(GeminiVisionLLMError, match=f"HTTP {status}") as error:
        call_with_server(server)
    assert "private-detail" not in str(error.value)


@pytest.mark.parametrize("exception", [httpx.ReadTimeout, httpx.ConnectError])
def test_gemini_reports_transport_failure_without_raw_details(exception):
    from src.libs.llm.gemini_vision_llm import GeminiVisionLLMError

    def server(request):
        raise exception("private-detail", request=request)

    with pytest.raises(GeminiVisionLLMError, match=exception.__name__) as error:
        call_with_server(server)
    assert "private-detail" not in str(error.value)


@pytest.mark.parametrize("body", [
    {}, {"candidates": []}, {"candidates": [{"content": {"parts": []}}]},
    {"candidates": [{"content": {"parts": [{"text": " "}]}}]},
    {"promptFeedback": {"blockReason": "SAFETY"}},
])
def test_gemini_rejects_empty_or_blocked_answer(body):
    with pytest.raises(RuntimeError, match="Invalid or blocked text response"):
        call_with_server(lambda request: httpx.Response(200, json=body))


@pytest.mark.parametrize("image", [
    ImageInput(path="not-created.png"), ImageInput(data=b"invalid-image"),
    ImageInput(base64="not-valid-base64!"),
])
def test_gemini_invalid_image_does_not_request_model(image):
    def no_request(request):
        raise AssertionError("invalid image must not request server")

    with pytest.raises(RuntimeError, match="Image preprocessing failed|Image encoding failed"):
        call_with_server(no_request, image)


@pytest.mark.parametrize("text,image,history", [
    (" ", ImageInput(data=b"fixture"), None),
    ("描述", "not-an-ImageInput", None),
    ("描述", ImageInput(data=b"fixture"), [Message("tool", "不支持")]),
])
def test_gemini_invalid_input_does_not_request_model(text, image, history):
    def no_request(request):
        raise AssertionError("invalid input must not request server")

    with httpx.Client(transport=httpx.MockTransport(no_request)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(httpx_client=http_client)) as client:
            llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
            with pytest.raises(ValueError):
                llm.chat_with_image(text, image, messages=history)


@pytest.mark.parametrize("field,value", [
    ("max_image_size", 0), ("max_image_size", True), ("max_image_size", 1.5),
    ("timeout", 0), ("timeout", float("inf")), ("timeout", True),
])
def test_gemini_constructor_rejects_invalid_limits(field, value):
    with pytest.raises(RuntimeError, match=field):
        LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", **{field: value})


def test_gemini_supports_model_override_without_changing_provider():
    def server(request):
        assert request.url.path == "/v1beta/models/override-vision:generateContent"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "描述"}]}}]})

    assert call_with_server(server, model="override-vision").model == "override-vision"


def test_gemini_preprocessing_handles_thin_image_without_zero_dimension():
    llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key")
    processed = llm.preprocess_image(ImageInput(data=picture_bytes((400, 1))), max_size=(1, 1))
    with Image.open(io.BytesIO(processed.data)) as image:
        assert image.size == (1, 1)


def test_gemini_self_created_client_uses_timeout_and_no_retry(monkeypatch):
    """外部 SDK 构造 seam 校验自有 client 配置，真实 SDK 仍处理请求。"""
    original_client = genai.Client
    clients = []

    class ObservedClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    def server(request):
        assert request.url.path == "/v1beta/models/test-gemini-vision:generateContent"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "描述"}]}}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        def create_client(**kwargs):
            assert kwargs["api_key"] == "test-key"
            assert kwargs["vertexai"] is False
            options = kwargs["http_options"]
            assert options.timeout == 60000
            assert options.retry_options.attempts == 1
            options.httpx_client = http_client
            client = ObservedClient(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(genai, "Client", create_client)
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key")
        assert llm.chat_with_image("描述", ImageInput(data=picture_bytes())).content == "描述"
        assert clients[0].was_closed
        assert not http_client.is_closed
