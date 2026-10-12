"""Offline demo CLI safeguards and the saved public acceptance summary."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_DEMO = REPO_ROOT / "scripts" / "offline_demo.py"
MODEL_KEY_NAMES = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_BASE_URL",
}
SAFE_INHERITED_ENV = (
    "PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL",
)
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_documents" / "evaluation"


def _offline_child_env(tmp_path: Path) -> dict[str, str]:
    """Pass only startup variables and guard Python socket APIs in the child."""
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import socket

            def _deny_network(*args, **kwargs):
                raise RuntimeError("network access is disabled in this test subprocess")

            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            socket.create_connection = _deny_network
            """
        ).lstrip(),
        encoding="utf-8",
    )
    environment = {name: os.environ[name] for name in SAFE_INHERITED_ENV if name in os.environ}
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    assert MODEL_KEY_NAMES.isdisjoint(environment)
    return environment


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(OFFLINE_DEMO), *args],
        cwd=tmp_path,
        env=_offline_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _saved_acceptance() -> tuple[Path, dict]:
    if "RAG_OFFLINE_ACCEPTANCE_ROOT" not in os.environ:
        pytest.skip("set RAG_OFFLINE_ACCEPTANCE_ROOT to audit a completed offline demo")
    selected = os.environ["RAG_OFFLINE_ACCEPTANCE_ROOT"]
    assert selected.strip(), "RAG_OFFLINE_ACCEPTANCE_ROOT must name an existing artifact directory"
    root = Path(selected).resolve()
    manifest = root / "acceptance.json"
    assert manifest.is_file(), "selected offline acceptance artifact must exist"
    return root, json.loads(manifest.read_text(encoding="utf-8"))


def test_offline_demo_help_is_available_without_starting_the_demo(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--help")

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "尚不存在" in result.stdout
    assert result.stderr == ""
    assert {path.name for path in tmp_path.iterdir()} == {"sitecustomize.py"}


def test_offline_demo_refuses_an_existing_output_directory_without_changing_it(tmp_path: Path) -> None:
    existing = tmp_path / "existing-output"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("user data stays intact\n", encoding="utf-8")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    result = _run_cli(tmp_path, "--output-dir", str(existing))

    assert result.returncode == 2
    assert "拒绝覆盖" in result.stderr
    assert result.stdout == ""
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    assert not (existing / "acceptance.json").exists()


def test_src_entry_disables_ort_telemetry_before_chromadb_import(tmp_path: Path) -> None:
    environment = _offline_child_env(tmp_path)
    environment["ORT_DISABLE_TELEMETRY"] = "0"
    probe = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        class CheckTelemetryBeforeChroma(importlib.abc.MetaPathFinder):
            chromadb_seen = False

            def find_spec(self, fullname, path=None, target=None):
                if fullname == "chromadb" or fullname.startswith("chromadb."):
                    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1", (
                        "src must disable ONNX Runtime telemetry before Chroma import"
                    )
                    self.chromadb_seen = True
                return None

        assert os.environ.get("ORT_DISABLE_TELEMETRY") == "0"
        guard = CheckTelemetryBeforeChroma()
        sys.meta_path.insert(0, guard)
        import src
        assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
        assert "chromadb" not in sys.modules

        import chromadb
        assert guard.chromadb_seen
        assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
        print("ORT_DISABLE_TELEMETRY=1 before chromadb import")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ORT_DISABLE_TELEMETRY=1 before chromadb import" in result.stdout


def test_saved_acceptance_exposes_the_offline_public_summary_and_completed_evidence() -> None:
    acceptance_root, acceptance = _saved_acceptance()
    assert set(acceptance) == {
        "success", "offline_only", "model", "documents_checked", "mcp", "dashboard", "evaluation", "artifacts",
    }
    assert acceptance["success"] is True
    assert acceptance["offline_only"] is True
    assert acceptance["model"] == "hash embedding; not Gemini semantic acceptance"
    assert acceptance["documents_checked"] == 5

    mcp = acceptance["mcp"]
    assert mcp["tool_count"] == 3
    assert mcp["queries_checked"] == 3
    assert {Path(citation["source"]).name for citation in mcp["citations"]} == {
        "01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md",
    }
    for citation in mcp["citations"]:
        source = Path(citation["source"])
        assert source.resolve().is_relative_to((acceptance_root / "documents").resolve())
        assert citation["metadata"]["source_ref"].startswith("doc_")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == citation["metadata"]["doc_hash"]
    assert mcp["image_checked"] is True
    assert mcp["summary_checked"] is True
    assert mcp["error_recovery_checked"] is True

    assert acceptance["dashboard"]["pages_checked"] == [
        "overview", "data_browser", "ingestion_manager", "ingestion_traces", "query_traces", "evaluation_panel",
    ]
    assert "not an actual Agent host" in acceptance["dashboard"]["mode"]
    assert acceptance["evaluation"] == {
        "metrics": {"hit_rate": 1.0, "mrr": 1.0},
        "metric_counts": {"hit_rate": 3, "mrr": 3},
        "gate_passed": True,
    }

    for key in ("config", "documents", "data", "trace", "report"):
        path = Path(acceptance["artifacts"][key])
        assert path.exists()
        assert path.resolve().is_relative_to(acceptance_root)

    service_report = json.loads(Path(acceptance["artifacts"]["report"]).read_text(encoding="utf-8"))
    cli_report = json.loads((acceptance_root / "cli-report.json").read_text(encoding="utf-8"))
    for report in (service_report, cli_report):
        assert report["collection"] == "offline-acceptance"
        assert report["top_k"] == 1
        assert report["query_count"] == 3
        assert report["aggregate_metrics"] == {"hit_rate": 1.0, "mrr": 1.0}
        assert report["metric_counts"] == {"hit_rate": 3, "mrr": 3}
        assert report["status_counts"] == {"success": 3, "partial": 0, "skipped": 0, "error": 0}

    trace_path = Path(acceptance["artifacts"]["trace"])
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert Counter(trace["trace_type"] for trace in traces) == {"ingestion": 10, "query": 6}
    assert all(trace["finished_at"] for trace in traces)
    assert Counter(trace["metadata"]["status"] for trace in traces if trace["trace_type"] == "ingestion") == {
        "success": 5, "skipped": 5,
    }
    assert Counter(trace["metadata"]["status"] for trace in traces if trace["trace_type"] == "query") == {
        "success": 5, "empty": 1,
    }


def test_saved_demo_inputs_are_public_or_synthetic_and_confined_to_the_new_output() -> None:
    root, acceptance = _saved_acceptance()
    documents = Path(acceptance["artifacts"]["documents"]).resolve()
    assert documents.is_relative_to(root)

    expected_files = {
        "01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md", "04-图片验收.md", "05-PDF验收.pdf",
        "assets/acceptance.png",
    }
    paths = list(documents.rglob("*"))
    assert {path.relative_to(documents).as_posix() for path in paths if path.is_file()} == expected_files
    assert all(not path.is_symlink() and path.resolve().is_relative_to(root) for path in paths)

    for name in ("01-摄取流程.md", "02-混合检索.md", "03-MCP通信.md"):
        assert (documents / name).read_bytes() == (FIXTURE_ROOT / name).read_bytes()
    assert "assets/acceptance.png" in (documents / "04-图片验收.md").read_text(encoding="utf-8")
    assert (documents / "05-PDF验收.pdf").stat().st_size > 0

    ingestion_sources = {
        Path(trace["metadata"]["source_path"]).resolve()
        for line in Path(acceptance["artifacts"]["trace"]).read_text(encoding="utf-8").splitlines()
        for trace in (json.loads(line),)
        if trace["trace_type"] == "ingestion"
    }
    assert ingestion_sources == {documents / name for name in expected_files if "/" not in name}
