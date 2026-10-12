"""G1：只绑定本机的 Streamlit 后台启动入口。"""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="启动本地知识库管理后台，不启动远程服务。")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "settings.yaml"))
    parser.add_argument("--data-dir", help="与摄取/查询 CLI 相同的隔离数据根目录")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1–65535")
    if importlib.util.find_spec("streamlit") is None:
        print("请先安装 .[dashboard] 依赖。", file=sys.stderr)
        return 2
    environment = os.environ.copy()
    environment["RAG_DASHBOARD_CONFIG"] = str(Path(args.config).resolve())
    environment.pop("RAG_DASHBOARD_DATA_DIR", None)
    if args.data_dir is not None:
        environment["RAG_DASHBOARD_DATA_DIR"] = str(Path(args.data_dir).resolve())
    command = [sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / "src/observability/dashboard/app.py"),
               "--server.address=127.0.0.1", f"--server.port={args.port}",
               "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxUploadSize=50"]
    print(f"本地后台：http://127.0.0.1:{args.port}；仅点击摄取时调用配置的模型。")
    return subprocess.run(command, cwd=REPO_ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
