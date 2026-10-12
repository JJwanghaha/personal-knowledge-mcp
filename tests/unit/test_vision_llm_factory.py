"""通过公开视觉接口验证图片输入和工厂，不调用真实模型。"""

import pytest
from dataclasses import replace

from src.core.settings import load_settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput

pytestmark = pytest.mark.unit


class ValidatingFakeVisionLLM(BaseVisionLLM):
    """替代外部模型，调用真实的公共输入校验，不模拟图片理解。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
        self.validate_text(text)
        self.validate_image(image)
        return ChatResponse("测试描述，不代表图片识别", "fake-vision")


def test_image_input_accepts_one_local_path_without_reading_it():
    """图片输入只表示来源；B8 不读取文件或检查图片内容。"""
    from src.libs.llm.base_vision_llm import ImageInput

    image = ImageInput(path="not-created-diagram.png")

    assert image.path == "not-created-diagram.png"
    assert image.data is None
    assert image.base64 is None
    assert image.mime_type == "image/png"


@pytest.mark.parametrize("inputs", [
    {},
    {"path": "diagram.png", "data": b"image"},
    {"path": "diagram.png", "base64": "aW1hZ2U="},
    {"data": b"image", "base64": "aW1hZ2U="},
    {"path": "diagram.png", "data": b"image", "base64": "aW1hZ2U="},
])
def test_image_input_requires_exactly_one_source(inputs):
    """零个来源或多个来源均不能表达明确的图片输入。"""
    from src.libs.llm.base_vision_llm import ImageInput

    with pytest.raises(ValueError, match="one of|exactly one"):
        ImageInput(**inputs)


def test_factory_creates_vision_adapter_using_text_provider_when_vision_is_absent():
    """沿用上游的配置选名规则，但使用独立视觉注册表。"""
    from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput

    class FakeVisionLLM(BaseVisionLLM):
        def __init__(self, settings, **kwargs):
            self.settings = settings
            self.kwargs = kwargs

        def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
            return ChatResponse("测试描述，不代表图片识别", "fake-vision")

    class IsolatedFactory(LLMFactory):
        _PROVIDERS = {}
        _VISION_PROVIDERS = {}

    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider="fake"))
    IsolatedFactory.register_vision_provider("fake", FakeVisionLLM)
    llm = IsolatedFactory.create_vision_llm(settings, label="test")
    response = llm.chat_with_image("描述图片", ImageInput(data=b"fixture"))

    assert llm.settings is settings
    assert llm.kwargs == {"label": "test"}
    assert response == ChatResponse("测试描述，不代表图片识别", "fake-vision")
    assert IsolatedFactory.list_vision_providers() == ["fake"]
    assert IsolatedFactory.list_providers() == []


def test_default_preprocessing_is_an_identity_extension_point():
    """上游基类不执行压缩或格式转换，不能据此声称已处理大图片。"""
    llm = ValidatingFakeVisionLLM()
    image = ImageInput(path="not-created.png")

    processed = llm.preprocess_image(image, max_size=(2048, 2048))

    assert processed is image
    assert llm.preprocess_image(processed) is image


def test_vision_config_loads_without_changing_text_model(tmp_path):
    """视觉配置是可选独立配置，不能覆盖文本模型参数。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data["vision_llm"] = {
        "enabled": False, "provider": "fake", "model": "fake-vision",
        "max_image_size": 2048, "azure_endpoint": "https://vision.example",
        "api_version": "test-version", "deployment_name": "vision-deployment",
        "base_url": "https://vision.example/v1",
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_settings(path)

    assert settings.vision_llm.enabled is False
    assert settings.vision_llm.provider == "fake"
    assert settings.vision_llm.model == "fake-vision"
    assert settings.vision_llm.max_image_size == 2048
    assert settings.vision_llm.azure_endpoint == "https://vision.example"
    assert settings.vision_llm.api_version == "test-version"
    assert settings.vision_llm.deployment_name == "vision-deployment"
    assert settings.vision_llm.base_url == "https://vision.example/v1"
    assert settings.llm.model == data["llm"]["model"]


def test_vision_adapter_accepts_text_and_image_with_shared_response():
    """统一视觉 interface 与文本调用复用 ChatResponse。"""
    llm = ValidatingFakeVisionLLM()
    response = llm.chat_with_image("描述图片", ImageInput(data=b"fixture"))

    assert response == ChatResponse("测试描述，不代表图片识别", "fake-vision")


@pytest.mark.parametrize("value", [0, -1, True])
def test_vision_config_rejects_invalid_image_size(tmp_path, value):
    """图片边长应为非布尔正整数；字段合法不代表已经执行压缩。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH, SettingsError

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data["vision_llm"] = {
        "enabled": False, "provider": "fake", "model": "fake-vision",
        "max_image_size": value,
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(SettingsError, match="vision_llm.max_image_size"):
        load_settings(path)


def _isolated_factory():
    """每次独立维护两张注册表，不污染正式 provider。"""
    class IsolatedFactory(LLMFactory):
        _PROVIDERS = {}
        _VISION_PROVIDERS = {}

    return IsolatedFactory


@pytest.mark.parametrize("inputs,mime", [
    ({"data": b"fixture"}, "image/jpeg"),
    ({"base64": "aW1hZ2U="}, "image/webp"),
])
def test_image_input_accepts_bytes_and_base64_without_decoding(inputs, mime):
    """编码数据和 MIME 原样保存，不把输入载体测试当作真实图片验收。"""
    image = ImageInput(**inputs, mime_type=mime)

    assert image.mime_type == mime
    assert image.data == inputs.get("data")
    assert image.base64 == inputs.get("base64")


def test_image_input_accepts_path_object():
    """调用方可使用 Path，无须转换成字符串。"""
    from pathlib import Path

    image = ImageInput(path=Path("not-created.png"))

    assert image.path == Path("not-created.png")


def test_vision_interface_cannot_be_instantiated_without_implementation():
    """抽象基类不能替代真实或 Fake 视觉 adapter。"""
    with pytest.raises(TypeError, match="abstract"):
        BaseVisionLLM()


@pytest.mark.parametrize("text", [None, 42, "", "  \n"])
def test_vision_adapter_rejects_invalid_prompt(text):
    """公共校验阻止空提示词和非字符串提示词进入模型。"""
    with pytest.raises(ValueError, match="Text must|Text prompt"):
        ValidatingFakeVisionLLM().chat_with_image(text, ImageInput(data=b"fixture"))


@pytest.mark.parametrize("image", [None, "diagram.png", {"path": "diagram.png"}])
def test_vision_adapter_rejects_non_image_input(image):
    """视觉接口要求显式 ImageInput，不猜测字符串是哪一种图片来源。"""
    with pytest.raises(ValueError, match="Image must be an ImageInput"):
        ValidatingFakeVisionLLM().chat_with_image("描述图片", image)


def test_default_config_keeps_vision_unconfigured():
    """加入可选配置不要求旧配置提供视觉字段，也不启用模型。"""
    assert load_settings().vision_llm is None


def test_vision_provider_takes_priority_and_normalizes_name():
    """视觉与文本可选不同 provider；enabled 的使用由调用方负责。"""
    from src.core.settings import VisionLLMSettings

    factory = _isolated_factory()
    factory.register_vision_provider(" VISION ", ValidatingFakeVisionLLM)
    settings = replace(load_settings(), vision_llm=VisionLLMSettings(
        enabled=False, provider=" ViSiOn ", model="vision-model", max_image_size=2048,
    ))

    llm = factory.create_vision_llm(settings, test_option="forwarded")

    assert isinstance(llm, ValidatingFakeVisionLLM)
    assert llm.settings is settings
    assert llm.kwargs == {"test_option": "forwarded"}
    assert factory.list_vision_providers() == ["vision"]
    assert factory.list_providers() == []


def test_unknown_vision_provider_does_not_fall_back_to_text_provider():
    """配置存在但选错后端时明确报错，不偷偷切换模型供应商。"""
    from src.core.settings import VisionLLMSettings

    factory = _isolated_factory()
    factory.register_vision_provider("gemini", ValidatingFakeVisionLLM)
    settings = replace(load_settings(), vision_llm=VisionLLMSettings(
        enabled=True, provider="unknown", model="vision-model", max_image_size=2048,
    ))

    with pytest.raises(ValueError, match="Unsupported Vision LLM provider: 'unknown'.*gemini"):
        factory.create_vision_llm(settings)


@pytest.mark.parametrize("provider_class", [object, object(), BaseLLM])
def test_vision_registry_rejects_non_vision_adapters(provider_class):
    """视觉注册表不接受普通类或非类对象。"""
    with pytest.raises(ValueError, match="must inherit from BaseVisionLLM"):
        _isolated_factory().register_vision_provider("invalid", provider_class)


def test_text_provider_registration_does_not_enable_vision():
    """文本注册表不会被视觉工厂当作替代后端。"""
    from src.libs.llm.base_llm import BaseLLM

    class TextOnlyLLM(BaseLLM):
        def chat(self, messages, trace=None, **kwargs):
            return ChatResponse("文本回答", "text-only")

    factory = _isolated_factory()
    factory.register_provider("gemini", TextOnlyLLM)

    with pytest.raises(ValueError, match="Available Vision LLM providers: none"):
        factory.create_vision_llm(load_settings())


def test_vision_factory_reports_missing_provider_configuration():
    """配置缺失时指出完整字段，不向调用方泄漏 AttributeError。"""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="settings.vision_llm.provider or settings.llm.provider"):
        _isolated_factory().create_vision_llm(SimpleNamespace())


def test_vision_factory_reports_construction_failure():
    """未知后端与已注册后端构造失败是不同错误。"""
    class BrokenVisionLLM(ValidatingFakeVisionLLM):
        def __init__(self, settings, **kwargs):
            raise ValueError("测试构造失败")

    factory = _isolated_factory()
    factory.register_vision_provider("broken", BrokenVisionLLM)
    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider="broken"))

    with pytest.raises(RuntimeError, match="Failed to instantiate Vision LLM provider 'broken': 测试构造失败"):
        factory.create_vision_llm(settings)


def test_vision_provider_list_is_sorted():
    """调用方得到稳定的可选视觉 provider 列表。"""
    factory = _isolated_factory()
    factory.register_vision_provider("zebra", ValidatingFakeVisionLLM)
    factory.register_vision_provider("alpha", ValidatingFakeVisionLLM)

    assert factory.list_vision_providers() == ["alpha", "zebra"]


@pytest.mark.parametrize("changes,error", [
    ({"enabled": "false"}, "vision_llm.enabled"),
    ({"provider": ""}, "vision_llm.provider"),
    ({"model": " "}, "vision_llm.model"),
    ({"max_image_size": 2048.5}, "vision_llm.max_image_size"),
    ({"base_url": 42}, "vision_llm.base_url"),
])
def test_vision_config_reports_invalid_fields(tmp_path, changes, error):
    """非敏感连接参数仍遵守 Settings 的类型化约定。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH, SettingsError

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    vision = {"enabled": False, "provider": "fake", "model": "fake", "max_image_size": 2048}
    vision.update(changes)
    data["vision_llm"] = vision
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(SettingsError, match=error):
        load_settings(path)


@pytest.mark.parametrize("section", [None, "vision", []])
def test_present_vision_config_must_be_a_mapping(tmp_path, section):
    """省略可选配置合法，显式给出 null 或错误结构则指出字段。"""
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH, SettingsError

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data["vision_llm"] = section
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(SettingsError, match="settings.vision_llm"):
        load_settings(path)


def test_vision_interface_and_factory_import_without_sdk_key_or_network():
    """B9 注册后仍可离线导入和构造，后端创建不初始化 SDK 或读图。"""
    import os
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT

    script = '''
import importlib.abc
import socket
import sys
from dataclasses import replace

class NoModelSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"google", "openai", "PIL", "sentence_transformers"}:
            raise ImportError("optional model and image SDK disabled")

def no_network(*args, **kwargs):
    raise AssertionError("B8 must not request external systems")

sys.meta_path.insert(0, NoModelSDK())
socket.socket.connect = no_network
from src.core.settings import load_settings
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory
assert LLMFactory.list_vision_providers() == ["azure", "gemini"]
assert load_settings().vision_llm is None
ImageInput(path="not-created.png")
settings = load_settings()
settings = replace(settings, llm=replace(settings.llm, provider="gemini", base_url=None))
llm = LLMFactory.create_vision_llm(settings, api_key="test-key")
image = ImageInput(data=b"fixture")
assert llm.preprocess_image(image, max_size=(1, 1)) is image
try:
    llm.chat_with_image("描述", image)
except RuntimeError as error:
    assert "Missing SDK" in str(error)
else:
    raise AssertionError("missing SDK must prevent real vision calls")
print("vision interface boundary passed")
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    }}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=environment,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "vision interface boundary passed"
