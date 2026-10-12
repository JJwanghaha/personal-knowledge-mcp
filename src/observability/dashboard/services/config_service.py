"""G1：只读取配置供总览显示，不创建模型或展示凭据。"""

from dataclasses import dataclass

from src.core.settings import load_settings


@dataclass
class ComponentInfo:
    name: str
    provider: str
    model: str
    extra: dict


class ConfigService:
    def __init__(self, settings_path=None):
        self.settings_path = settings_path
        self._settings = None

    @property
    def settings(self):
        if self._settings is None:
            self._settings = load_settings(self.settings_path)
        return self._settings

    def reload(self):
        self._settings = None

    def get_component_cards(self):
        s = self.settings
        cards = [ComponentInfo("生成模型", s.llm.provider, s.llm.model, {"temperature": s.llm.temperature}),
                 ComponentInfo("向量模型", s.embedding.provider, s.embedding.model, {"dimensions": s.embedding.dimensions}),
                 ComponentInfo("向量存储", s.vector_store.provider, s.vector_store.collection_name, {"path": s.vector_store.persist_directory}),
                 ComponentInfo("混合检索", "dense + bm25", "RRF", {"dense_top_k": s.retrieval.dense_top_k,
                     "sparse_top_k": s.retrieval.sparse_top_k, "fusion_top_k": s.retrieval.fusion_top_k, "rrf_k": s.retrieval.rrf_k}),
                 ComponentInfo("重排", s.rerank.provider if s.rerank.enabled else "关闭", s.rerank.model, {"enabled": s.rerank.enabled}),
                 ComponentInfo("Trace", "JSONL", "开启" if s.observability.trace_enabled else "关闭", {"path": s.observability.trace_file})]
        if s.ingestion is not None:
            cards.append(ComponentInfo("文档摄取", s.ingestion.splitter, "PDF / Markdown", {"chunk_size": s.ingestion.chunk_size,
                "chunk_overlap": s.ingestion.chunk_overlap, "batch_size": s.ingestion.batch_size}))
        if s.vision_llm is not None:
            cards.append(ComponentInfo("图片描述", s.vision_llm.provider, s.vision_llm.model, {"enabled": s.vision_llm.enabled}))
        return cards
