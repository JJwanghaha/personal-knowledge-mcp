"""项目配置的数据结构与加载入口。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


class SettingsError(ValueError):
    """配置文件不存在或内容不合法。"""


def resolve_path(relative: str | Path) -> Path:
    """把项目相对路径解析为不依赖当前工作目录的绝对路径。"""
    path = Path(relative)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require_mapping(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = data.get(key)
    field_path = f"{path}.{key}"
    if value is None:
        raise SettingsError(f"Missing required field: {field_path}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {field_path}")
    return value


def _require_value(data: dict[str, Any], key: str, path: str) -> Any:
    field_path = f"{path}.{key}"
    if key not in data or data[key] is None:
        raise SettingsError(f"Missing required field: {field_path}")
    return data[key]


def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: dict[str, Any], key: str, path: str) -> list[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


def _optional_str(data: dict[str, Any], key: str, path: str) -> str | None:
    """连接字段允许省略，提供时仍须是非空字符串。"""
    if data.get(key) is None:
        return None
    return _require_str(data, key, path)


@dataclass(frozen=True)
class LLMSettings:
    """LLM 运行参数。"""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    base_url: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
    timeout: float = 60.0


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding 运行参数。"""

    provider: str
    model: str
    dimensions: int
    base_url: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
    timeout: float = 60.0


@dataclass(frozen=True)
class VisionLLMSettings:
    """视觉调用独立配置；max_image_size 供后续 adapter 使用。"""

    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_version: str | None = None
    azure_endpoint: str | None = None
    deployment_name: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class VectorStoreSettings:
    """向量存储参数。"""

    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    """候选召回与融合参数。"""

    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    """重排序参数。"""

    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    """评估参数。"""

    enabled: bool
    provider: str
    metrics: list[str]
    backends: list[str] | None = None


@dataclass(frozen=True)
class ObservabilitySettings:
    """日志与链路追踪参数。"""

    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class IngestionSettings:
    """离线摄取参数。"""

    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: dict[str, Any] | None = None
    metadata_enricher: dict[str, Any] | None = None


@dataclass(frozen=True)
class Settings:
    """所有模块共享的类型化配置。"""

    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: IngestionSettings | None = None
    vision_llm: VisionLLMSettings | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """把已解析的 YAML mapping 转换为 Settings。"""
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")
        ingestion = data.get("ingestion")

        ingestion_settings = None
        if ingestion is not None:
            if not isinstance(ingestion, dict):
                raise SettingsError("Expected mapping for field: settings.ingestion")
            for name in ("chunk_refiner", "metadata_enricher"):
                option = ingestion.get(name)
                if option is not None:
                    if not isinstance(option, dict):
                        raise SettingsError(f"Expected mapping for field: ingestion.{name}")
                    if "use_llm" in option and not isinstance(option["use_llm"], bool):
                        raise SettingsError(f"Expected boolean for field: ingestion.{name}.use_llm")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),
                metadata_enricher=ingestion.get("metadata_enricher"),
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision, "enabled", "vision_llm"),
                provider=_require_str(vision, "provider", "vision_llm"),
                model=_require_str(vision, "model", "vision_llm"),
                max_image_size=_require_int(vision, "max_image_size", "vision_llm"),
                api_version=_optional_str(vision, "api_version", "vision_llm"),
                azure_endpoint=_optional_str(vision, "azure_endpoint", "vision_llm"),
                deployment_name=_optional_str(vision, "deployment_name", "vision_llm"),
                base_url=_optional_str(vision, "base_url", "vision_llm"),
            )

        return cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                base_url=_optional_str(llm, "base_url", "llm"),
                deployment_name=_optional_str(llm, "deployment_name", "llm"),
                api_version=_optional_str(llm, "api_version", "llm"),
                timeout=_require_number(llm, "timeout", "llm") if "timeout" in llm else 60.0,
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                base_url=_optional_str(embedding, "base_url", "embedding"),
                deployment_name=_optional_str(embedding, "deployment_name", "embedding"),
                api_version=_optional_str(embedding, "api_version", "embedding"),
                timeout=_require_number(embedding, "timeout", "embedding") if "timeout" in embedding else 60.0,
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(
                    vector_store, "persist_directory", "vector_store"
                ),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
                backends=[str(item) for item in _require_list(evaluation, "backends", "evaluation")]
                         if "backends" in evaluation else None,
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(
                    observability, "structured_logging", "observability"
                ),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
        )


def validate_settings(settings: Settings) -> None:
    """校验构造后仍需满足的关键配置约束。"""
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if isinstance(settings.embedding.dimensions, bool) or settings.embedding.dimensions <= 0:
        raise SettingsError("Expected positive integer for field: embedding.dimensions")
    for section in ("llm", "embedding"):
        timeout = getattr(settings, section).timeout
        if not isfinite(timeout) or timeout <= 0:
            raise SettingsError(f"Expected positive finite number for field: {section}.timeout")
    if settings.vision_llm is not None:
        size = settings.vision_llm.max_image_size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SettingsError("Expected positive integer for field: vision_llm.max_image_size")


def load_settings(path: str | Path | None = None) -> Settings:
    """从 YAML 文件加载类型化配置。"""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
