"""H1：Ragas 现代模型 interface 与现有 Gemini/Embedding 的小适配器。"""

import asyncio

from src.observability.evaluation.ragas_evaluator import _import_ragas

_import_ragas()

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding


class ThreadedRagasLLM(InstructorBaseRagasLLM):
    """Ragas 指标内部走 async，把同步 Instructor 客户端放到工作线程使用。"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt, response_model):
        return self.llm.generate(prompt, response_model)

    async def agenerate(self, prompt, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)


class GeminiRagasLLM(InstructorBaseRagasLLM):
    """原生 generateContent + JSON Schema；不通过旧 Google SDK 或公司网关。"""

    def __init__(self, config, client):
        self.config, self.client = config, client

    def generate(self, prompt, response_model):
        from google.genai import types
        config = types.GenerateContentConfig(temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens, response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema())
        response = self.client.models.generate_content(model=self.config.model, contents=prompt, config=config)
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("Gemini returned no structured judgment")
        return response_model.model_validate_json(response.text)

    async def agenerate(self, prompt, response_model):
        # 使用已持有的同步客户端；不留下跨 asyncio.run 的异步连接。
        return await asyncio.to_thread(self.generate, prompt, response_model)


class ProjectRagasEmbedding(BaseRagasEmbedding):
    """复用 BaseEmbedding 的批量、维度、厂商协议，不让 Ragas 自动选择模型。"""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def embed_text(self, text, **kwargs):
        return self.embed_texts([text], **kwargs)[0]

    def embed_texts(self, texts, **kwargs):
        vectors = self.embedding.embed(texts, **kwargs)
        self.embedding.validate_vectors(vectors, len(texts))
        return vectors

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    async def aembed_texts(self, texts, **kwargs):
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)
