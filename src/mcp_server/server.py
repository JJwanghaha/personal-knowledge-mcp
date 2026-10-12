"""E1：官方 MCP SDK 的本地 stdio 服务入口，stdout 专用于协议。"""

import argparse
import asyncio
from contextlib import redirect_stdout
import logging
import sys
from pathlib import Path

from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.mcp_server.protocol_handler import ProtocolHandler, create_mcp_server

SERVER_NAME = "personal-knowledge-mcp"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr(log_level: str = "INFO") -> None:
    """服务运行时把控制台日志统一到 stderr，保留 FileHandler，不关闭标准流。"""
    root = logging.getLogger()
    loggers = [root] + [item for item in logging.Logger.manager.loggerDict.values() if isinstance(item, logging.Logger)]
    for logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        if logger is not root:
            logger.propagate = True
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(log_level.upper())


def create_default_protocol_handler(settings=None, *, data_dir=None, embedding_client=None) -> ProtocolHandler:
    """仅登记三个业务工具；外部模型 seam 可注入供离线验收，不触发实际调用。"""
    from src.mcp_server.tools import register_default_tools

    handler = ProtocolHandler(SERVER_NAME, SERVER_VERSION)
    register_default_tools(handler, settings, data_dir=data_dir, embedding_client=embedding_client)
    return handler


def _preload_heavy_imports() -> None:
    """按上游在 stdio IO 线程启动前导入 Chroma，降低 worker 首次导入锁风险。"""
    with redirect_stdout(sys.stderr):
        try:
            import chromadb
        except ImportError:
            logging.getLogger(__name__).warning("Chroma SDK 未安装；工具列表可用，实际存储操作需要 vector-stores 依赖。")


async def run_stdio_server_async(protocol_handler: ProtocolHandler | None = None,
                                 log_level: str = "INFO", *, settings=None, data_dir=None) -> int:
    """默认登记 E3–E5；注入自定义注册表时不混入默认工具，客户端断开后退出。"""
    _redirect_all_loggers_to_stderr(log_level)
    import mcp.server.stdio

    handler = protocol_handler if protocol_handler is not None else create_default_protocol_handler(settings, data_dir=data_dir)
    _preload_heavy_imports()
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION, handler)
    logging.getLogger(__name__).info("Starting MCP stdio server; registered tools=%d.", len(handler.tools))
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    logging.getLogger(__name__).info("MCP stdio server stopped.")
    return 0


def run_stdio_server(protocol_handler: ProtocolHandler | None = None, log_level: str = "INFO", *, settings=None, data_dir=None) -> int:
    """同步进程入口；已有事件循环中的调用者使用 async 版本。"""
    return asyncio.run(run_stdio_server_async(protocol_handler, log_level, settings=settings, data_dir=data_dir))


def main(argv: list[str] | None = None) -> int:
    """加载配置后开始 stdio；help/check 不启动协议服务，check 信息只写 stderr。"""
    parser = argparse.ArgumentParser(description="启动本地知识库 MCP stdio 服务，提供检索、集合列表和文档摘要工具。")
    parser.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH), help="YAML 配置文件")
    parser.add_argument("--check", action="store_true", help="仅检查配置和 MCP 注册表，不等待客户端输入")
    parser.add_argument("--data-dir", help="与 ingest.py/query.py 相同的隔离数据根目录")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(Path(args.config).resolve())
        if args.check:
            _redirect_all_loggers_to_stderr(settings.observability.log_level)
            handler = create_default_protocol_handler(settings, data_dir=args.data_dir)
            server = create_mcp_server(protocol_handler=handler)
            server.create_initialization_options()
            logging.getLogger(__name__).info("MCP 配置与 SDK 检查通过；已登记 %d 个工具，未执行模型或存储操作。", len(handler.tools))
            return 0
        return run_stdio_server(log_level=settings.observability.log_level, settings=settings, data_dir=args.data_dir)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"MCP 启动失败：{type(exc).__name__}。请检查配置及 .[mcp] 依赖。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
