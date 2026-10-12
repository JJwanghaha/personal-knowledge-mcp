"""I1/I2/I5：无 Key 的可复现离线验收；外部模型替换，业务模块保持真实。"""

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_ENV = {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
             "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "OPENAI_BASE_URL"}
COLLECTION = "offline-acceptance"
PROVIDER = "acceptance_offline"

from src.libs.embedding.base_embedding import BaseEmbedding


class DemoEmbedding(BaseEmbedding):
    """固定维度词项哈希，不是 Gemini，也不代表语义效果。"""

    def __init__(self, settings=None, **kwargs):
        self.dimension = 128

    def get_dimension(self):
        return self.dimension

    def embed(self, texts, trace=None, **kwargs):
        import jieba
        self.validate_texts(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in jieba.lcut(text.casefold()):
                if token.strip():
                    slot = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % self.dimension
                    vector[slot] += 1
            norm = sum(value * value for value in vector) ** 0.5 or 1
            vectors.append([value / norm for value in vector])
        return vectors


@contextmanager
def offline_environment():
    """清理模型凭据、拒绝 Python socket 连接；不等同操作系统断网沙箱。"""
    for key in MODEL_ENV:
        os.environ.pop(key, None)
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    original_connect, original_create = socket.socket.connect, socket.create_connection

    def reject_network(*args, **kwargs):
        raise RuntimeError("Offline acceptance forbids network access")

    socket.socket.connect, socket.create_connection = reject_network, reject_network
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = original_connect, original_create


def require(condition, stage):
    if not condition:
        raise RuntimeError(f"Acceptance check failed: {stage}")


def prepare(root):
    """仅写入全新的验收目录，公开样例不覆盖工作资料。"""
    import yaml
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    documents, data = root / "documents", root / "data"
    documents.mkdir(parents=True)
    fixtures = REPO_ROOT / "tests/fixtures/sample_documents/evaluation"
    for source in fixtures.glob("*.md"):
        shutil.copy2(source, documents / source.name)
    (documents / "assets").mkdir()
    image_path = documents / "assets" / "acceptance.png"
    with Image.new("RGB", (64, 32), "teal") as image:
        image.save(image_path)
    (documents / "04-图片验收.md").write_text(
        "# 独立图片验收\n\n本页包含用于本地验收的蓝绿色图片示意。\n\n![示意图](assets/acceptance.png)\n", encoding="utf-8")
    canvas = Canvas(str(documents / "05-PDF验收.pdf"))
    canvas.drawString(50, 760, "Quartz PDF acceptance record with a local image.")
    canvas.drawImage(str(image_path), 50, 680, width=128, height=64)
    canvas.save()
    values = {
        "llm": {"provider": "unused_offline", "model": "not-called", "temperature": 0.0, "max_tokens": 64},
        "embedding": {"provider": PROVIDER, "model": "hash-128-not-semantic", "dimensions": 128},
        "vector_store": {"provider": "chroma", "persist_directory": str(data / "db/chroma"), "collection_name": COLLECTION},
        "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 10, "rrf_k": 30},
        "rerank": {"enabled": False, "provider": "none", "model": "not-called", "top_k": 5},
        "evaluation": {"enabled": True, "provider": "custom", "metrics": ["hit_rate", "mrr"]},
        "observability": {"log_level": "ERROR", "trace_enabled": True,
                          "trace_file": str(root / "logs/traces.jsonl"), "structured_logging": True},
        "ingestion": {"chunk_size": 1200, "chunk_overlap": 0, "splitter": "recursive", "batch_size": 8,
                      "chunk_refiner": {"use_llm": False}, "metadata_enricher": {"use_llm": False}},
        "vision_llm": {"enabled": False, "provider": "unused_offline", "model": "not-called", "max_image_size": 1024},
    }
    config = root / "settings.offline.yaml"
    config.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config, documents, data


async def verify_mcp(config, data, root):
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from PIL import Image
    import base64
    import io

    environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL") if key in os.environ}
    environment.update({"RAGAS_DO_NOT_TRACK": "true", "PYTHONUNBUFFERED": "1"})
    parameters = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve()), "--serve-mcp",
        "--config", str(config), "--data-dir", str(data)], env=environment, cwd=str(root))
    with (root / "mcp.stderr.log").open("w", encoding="utf-8") as stderr:
        with anyio.fail_after(90):
            async with stdio_client(parameters, errlog=stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    require({tool.name for tool in tools.tools} == {"query_knowledge_hub", "list_collections", "get_document_summary"}, "mcp_tools")
                    collections = await session.call_tool("list_collections", {})
                    require(not collections.isError and COLLECTION in {c["name"] for c in collections.structuredContent["collections"]}, "mcp_collection")
                    citations = []
                    golden = json.loads((REPO_ROOT / "tests/fixtures/golden_test_set.json").read_text(encoding="utf-8"))
                    for case in golden["test_cases"]:
                        result = await session.call_tool("query_knowledge_hub", {"query": case["query"], "collection": COLLECTION, "top_k": 1})
                        require(not result.isError and not result.structuredContent["isEmpty"], "mcp_query")
                        cite = result.structuredContent["citations"][0]
                        require(Path(cite["source"]).name in case["expected_sources"], "mcp_source")
                        citations.append(cite)
                        summary = await session.call_tool("get_document_summary", {"doc_id": cite["metadata"]["source_ref"], "collection": COLLECTION})
                        require(not summary.isError and summary.structuredContent["chunk_count"] > 0, "mcp_summary")
                    picture = await session.call_tool("query_knowledge_hub", {"query": "独立图片验收 蓝绿色图片示意", "collection": COLLECTION, "top_k": 1})
                    images = [block for block in picture.content if block.type == "image"]
                    require(bool(images), "mcp_image")
                    with Image.open(io.BytesIO(base64.b64decode(images[0].data))) as image:
                        image.verify()
                    missing = await session.call_tool("query_knowledge_hub", {"query": "没有资料", "collection": "missing-collection"})
                    require(not missing.isError and missing.structuredContent["isEmpty"], "mcp_missing_collection")
                    invalid = await session.call_tool("query_knowledge_hub", {"query": 42})
                    require(invalid.isError, "mcp_schema_error")
                    recovered = await session.call_tool("list_collections", {})
                    require(not recovered.isError, "mcp_connection_recovery")
    return {"tool_count": 3, "queries_checked": len(citations), "citations": citations,
            "image_checked": True, "summary_checked": True, "error_recovery_checked": True}


def verify_dashboard(config, data):
    from streamlit.testing.v1 import AppTest
    names = ("overview", "data_browser", "ingestion_manager", "ingestion_traces", "query_traces", "evaluation_panel")
    os.environ["RAG_DASHBOARD_CONFIG"], os.environ["RAG_DASHBOARD_DATA_DIR"] = str(config), str(data)
    app = AppTest.from_file(str(REPO_ROOT / "src/observability/dashboard/app.py")).run(timeout=30)
    for name in names:
        app.switch_page(f"pages/{name}.py").run(timeout=30)
        require(not app.exception and not app.error, f"dashboard_{name}")
    return {"pages_checked": list(names), "mode": "AppTest; not an actual Agent host"}


def run_demo(root):
    from scripts import ingest, query, evaluate
    from src.core.settings import load_settings
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.observability.dashboard.services.data_service import DataService
    from src.observability.dashboard.services.evaluation_service import EvaluationService
    from src.observability.dashboard.services.trace_service import TraceService
    from src.observability.evaluation.quality_gate import check_quality_gate

    config, documents, data = prepare(root)
    EmbeddingFactory.register_provider(PROVIDER, DemoEmbedding)
    require(ingest.main(["--path", str(documents), "--collection", COLLECTION, "--config", str(config), "--data-dir", str(data)]) == 0, "ingestion_cli")
    require(ingest.main(["--path", str(documents), "--collection", COLLECTION, "--config", str(config), "--data-dir", str(data)]) == 0, "duplicate_skip_cli")
    settings = load_settings(config)
    browser = DataService(settings, data_dir=data)
    docs = browser.list_documents(COLLECTION)
    require(len(docs) == 5 and sum(doc["image_count"] for doc in docs) >= 2, "stored_documents_and_images")
    require(query.main(["--query", "RRF 怎样融合 BM25 候选", "--collection", COLLECTION, "--config", str(config),
                        "--data-dir", str(data), "--top-k", "1", "--no-rerank"]) == 0, "query_cli")
    mcp = asyncio.run(verify_mcp(config, data, root))
    service = EvaluationService(settings, data_dir=data, embedding_client=DemoEmbedding())
    report = service.run(collection=COLLECTION, backend="custom", top_k=1, no_rerank=True)
    gate = check_quality_gate(report, {"hit_rate": 1.0, "mrr": 1.0})
    require(gate.passed and report.metric_counts == {"hit_rate": 3, "mrr": 3}, "evaluation_gate")
    report_path = service.save_report(report)
    require(evaluate.main(["--collection", COLLECTION, "--config", str(config), "--data-dir", str(data), "--top-k", "1",
                           "--backend", "custom", "--min-hit-rate", "1", "--min-mrr", "1", "--output", str(root / "cli-report.json")]) == 0, "evaluation_cli")
    dashboard = verify_dashboard(config, data)
    traces = TraceService(settings.observability.trace_file)
    ingestion_traces, query_traces = traces.list_traces("ingestion"), traces.list_traces("query")
    require(len(ingestion_traces) >= 10 and len(query_traces) >= 5 and all(t["finished_at"] for t in ingestion_traces + query_traces), "completed_traces")
    return {"success": True, "offline_only": True, "model": "hash embedding; not Gemini semantic acceptance",
            "documents_checked": len(docs), "mcp": mcp, "dashboard": dashboard,
            "evaluation": {"metrics": report.aggregate_metrics, "metric_counts": report.metric_counts, "gate_passed": gate.passed},
            "artifacts": {"config": str(config), "documents": str(documents), "data": str(data),
                          "trace": str(traces.traces_path), "report": str(report_path)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行完整离线验收，不需要 Key，不读取真实知识库；结果保存在新的独立目录。")
    parser.add_argument("--output-dir", help="必须是尚不存在的目录，默认 data/acceptance/随机ID")
    parser.add_argument("--serve-mcp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    with offline_environment():
        if args.serve_mcp:
            from src.core.settings import load_settings
            from src.mcp_server.server import create_default_protocol_handler, run_stdio_server
            settings = load_settings(args.config)
            handler = create_default_protocol_handler(settings, data_dir=args.data_dir, embedding_client=DemoEmbedding())
            return run_stdio_server(handler, log_level="ERROR")
        root = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "data/acceptance" / uuid.uuid4().hex
        if root.exists():
            print("验收目录已存在，拒绝覆盖。", file=sys.stderr)
            return 2
        root.mkdir(parents=True)
        try:
            result = run_demo(root)
        except Exception as exc:
            result = {"success": False, "offline_only": True, "error_type": type(exc).__name__, "output_dir": str(root)}
            print(f"离线验收失败：{type(exc).__name__}；保留独立目录供排查。", file=sys.stderr)
        (root / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"验收结果：{root / 'acceptance.json'}；{'通过' if result['success'] else '失败'}。不是实际 Gemini/宿主验收。")
        return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
