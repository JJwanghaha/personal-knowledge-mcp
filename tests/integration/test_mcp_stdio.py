"""E1-E2 通过官方 MCP SDK 与独立 stdio 子进程的协议验收。"""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch

import anyio
import pytest
import yaml
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ENTRY = REPO_ROOT / "main.py"
MODEL_KEY_NAMES = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
}
SAFE_INHERITED_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")


def _isolated_child_environment(tmp_path: Path) -> dict[str, str]:
    environment = {key: os.environ[key] for key in SAFE_INHERITED_ENV if key in os.environ}
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    assert MODEL_KEY_NAMES.isdisjoint(environment)
    return environment


def _write_network_guard(tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import socket

            def _deny_network(*args, **kwargs):
                raise AssertionError("network access is disabled in this test subprocess")

            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            socket.create_connection = _deny_network
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _write_settings(tmp_path: Path) -> Path:
    settings = {
        "llm": {"provider": "openai", "model": "test-only", "temperature": 0.0, "max_tokens": 32},
        "embedding": {"provider": "openai", "model": "test-only", "dimensions": 8},
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "vector-store-must-not-be-created"),
            "collection_name": "mcp_protocol_probe",
        },
        "retrieval": {"dense_top_k": 2, "sparse_top_k": 2, "fusion_top_k": 2, "rrf_k": 10},
        "rerank": {"enabled": False, "provider": "none", "model": "test-only", "top_k": 1},
        "evaluation": {"enabled": False, "provider": "custom", "metrics": []},
        "observability": {
            "log_level": "INFO",
            "trace_enabled": False,
            "trace_file": str(tmp_path / "trace-must-not-be-created.jsonl"),
            "structured_logging": False,
        },
    }
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return config_path


def _text_content(result: types.CallToolResult) -> str:
    return "\n".join(item.text for item in result.content if isinstance(item, types.TextContent))


def test_registered_tools_work_through_official_stdio_client(tmp_path: Path) -> None:
    _write_network_guard(tmp_path)
    wrapper = tmp_path / "mcp_probe_server.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            from src.mcp_server.protocol_handler import ProtocolHandler
            from src.mcp_server.server import run_stdio_server

            handler = ProtocolHandler("stdio-probe", "9.8.7")
            schema = {
                "type": "object",
                "properties": {"value": {"type": "string", "maxLength": 32}},
                "required": ["value"],
                "additionalProperties": False,
            }

            def echo_mapping(value):
                return {"echo": value}

            async def echo_async(value):
                return "async:" + value

            def fail_with_private_detail(value):
                raise RuntimeError("PRIVATE_SENTINEL:" + value)

            handler.register_tool("echo_mapping", "Return a mapping", schema, echo_mapping)
            handler.register_tool("echo_async", "Return asynchronous text", schema, echo_async)
            handler.register_tool("fail", "Raise a sanitized probe error", schema, fail_with_private_detail)
            raise SystemExit(run_stdio_server(handler, log_level="INFO"))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    environment = _isolated_child_environment(tmp_path)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(wrapper)],
        env=environment,
        cwd=tmp_path,
    )
    captured_stderr = tmp_path / "probe-server.stderr"

    async def exercise_protocol() -> None:
        with anyio.fail_after(10):
            with captured_stderr.open("w", encoding="utf-8") as stderr_log:
                async with stdio_client(parameters, errlog=stderr_log) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        initialized = await session.initialize()
                        assert initialized.protocolVersion == types.LATEST_PROTOCOL_VERSION
                        assert initialized.serverInfo.name == "personal-knowledge-mcp"
                        assert initialized.serverInfo.version == "0.1.0"
                        assert initialized.capabilities.tools is not None
                        assert initialized.capabilities.prompts is None
                        assert initialized.capabilities.resources is None

                        listing = await session.list_tools()
                        assert {tool.name for tool in listing.tools} == {"echo_mapping", "echo_async", "fail"}
                        echo_schema = next(tool.inputSchema for tool in listing.tools if tool.name == "echo_mapping")
                        assert echo_schema["required"] == ["value"]
                        assert echo_schema["additionalProperties"] is False

                        mapping_result = await session.call_tool("echo_mapping", {"value": "hello"})
                        assert mapping_result.isError is False
                        assert mapping_result.structuredContent == {"echo": "hello"}
                        assert _text_content(mapping_result) == '{"echo": "hello"}'

                        async_result = await session.call_tool("echo_async", {"value": "world"})
                        assert async_result.isError is False
                        assert _text_content(async_result) == "async:world"

                        invalid_result = await session.call_tool("echo_mapping", {"value": "TOO_LONG_" * 8})
                        assert invalid_result.isError is True
                        assert "TOO_LONG_" not in _text_content(invalid_result)

                        unknown_result = await session.call_tool("not_registered", {"value": "PRIVATE_SENTINEL"})
                        assert unknown_result.isError is True
                        assert "PRIVATE_SENTINEL" not in _text_content(unknown_result)

                        failure_result = await session.call_tool("fail", {"value": "PRIVATE_SENTINEL"})
                        assert failure_result.isError is True
                        assert "PRIVATE_SENTINEL" not in _text_content(failure_result)

                        recovery_result = await session.call_tool("echo_mapping", {"value": "still-usable"})
                        assert recovery_result.isError is False
                        assert recovery_result.structuredContent == {"echo": "still-usable"}

    with patch.dict(os.environ, environment, clear=True):
        asyncio.run(exercise_protocol())

    stderr_text = captured_stderr.read_text(encoding="utf-8")
    assert "Starting MCP stdio server" in stderr_text
    assert "Tool execution failed: RuntimeError" in stderr_text
    assert "MCP stdio server stopped" in stderr_text
    assert "PRIVATE_SENTINEL" not in stderr_text


def test_import_help_and_console_check_do_not_initialize_rag_resources(tmp_path: Path) -> None:
    _write_network_guard(tmp_path)
    config_path = _write_settings(tmp_path)
    environment = _isolated_child_environment(tmp_path)
    import_probe = tmp_path / "import_probe.py"
    import_probe.write_text(
        textwrap.dedent(
            """
            import sys
            import main
            import src.mcp_server.protocol_handler
            import src.mcp_server.server

            assert "mcp" not in sys.modules
            assert "openai" not in sys.modules
            assert "google.genai" not in sys.modules
            assert "chromadb" not in sys.modules
            """
        ).lstrip(),
        encoding="utf-8",
    )

    imported = subprocess.run(
        [sys.executable, str(import_probe)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout == ""
    assert not (tmp_path / "vector-store-must-not-be-created").exists()

    help_result = subprocess.run(
        [sys.executable, str(MAIN_ENTRY), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--check" in help_result.stdout
    assert help_result.stderr == ""

    console_entrypoint = Path(sys.executable).with_name("mcp-server")
    assert console_entrypoint.is_file(), "the installed mcp-server console entrypoint is required for this check"
    check_result = subprocess.run(
        [str(console_entrypoint), "--config", str(config_path), "--check"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert check_result.returncode == 0, check_result.stderr
    assert check_result.stdout == ""
    assert "检查通过" in check_result.stderr
    assert not (tmp_path / "vector-store-must-not-be-created").exists()
    assert not (tmp_path / "trace-must-not-be-created.jsonl").exists()


def test_main_stdio_starts_with_three_default_tools_and_exits_on_disconnect(tmp_path: Path) -> None:
    _write_network_guard(tmp_path)
    config_path = _write_settings(tmp_path)
    environment = _isolated_child_environment(tmp_path)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(MAIN_ENTRY), "--config", str(config_path)],
        env=environment,
        cwd=tmp_path,
    )
    captured_stderr = tmp_path / "main-server.stderr"

    async def initialize_and_disconnect() -> None:
        with anyio.fail_after(10):
            with captured_stderr.open("w", encoding="utf-8") as stderr_log:
                async with stdio_client(parameters, errlog=stderr_log) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        initialized = await session.initialize()
                        assert initialized.serverInfo.name == "personal-knowledge-mcp"
                        assert initialized.serverInfo.version == "0.1.0"
                        assert initialized.capabilities.tools is not None
                        listing = await session.list_tools()
                        assert {tool.name for tool in listing.tools} == {
                            "query_knowledge_hub", "list_collections", "get_document_summary",
                        }

    with patch.dict(os.environ, environment, clear=True):
        asyncio.run(initialize_and_disconnect())

    stderr_text = captured_stderr.read_text(encoding="utf-8")
    assert "MCP stdio server stopped" in stderr_text
    assert "SENTINEL" not in stderr_text
    assert not (tmp_path / "vector-store-must-not-be-created").exists()
    assert not (tmp_path / "trace-must-not-be-created.jsonl").exists()
