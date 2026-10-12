"""E2：登记工具、列出 schema、把参数分发给已注册 Python 函数。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """客户端只收到名称/说明/schema，handler 引用留在本进程内。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ProtocolHandler:
    """沿用上游注册表 interface，不执行客户端提供的 Python 代码。"""

    server_name: str = "personal-knowledge-mcp"
    server_version: str = "0.1.0"
    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register_tool(self, name: str, description: str, input_schema: dict[str, Any],
                      handler: Callable[..., Any]) -> None:
        """注册函数引用而非调用结果；支持同步函数和异步函数。"""
        from jsonschema import Draft202012Validator

        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
            raise ValueError("Tool name must be a valid nonempty identifier")
        if name in self.tools:
            raise ValueError("Tool name is already registered")
        if not callable(handler):
            raise ValueError("Tool handler must be callable")
        if not isinstance(description, str) or not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise ValueError("Tool needs a description and an object input schema")
        Draft202012Validator.check_schema(input_schema)
        self.tools[name] = ToolDefinition(name, description, deepcopy(input_schema), handler)
        logger.info("Registered tool: %s", name)

    def get_tool_schemas(self) -> list:
        """列工具不执行 handler；导入 SDK 本身不创建模型或数据库。"""
        from mcp import types

        return [types.Tool(name=tool.name, description=tool.description, inputSchema=deepcopy(tool.input_schema))
                for tool in self.tools.values()]

    def get_capabilities(self) -> dict[str, Any]:
        """支持 tools/list 和 tools/call；工具可以尚未注册，不宣称支持其他能力。"""
        return {"tools": {}}

    async def execute_tool(self, name: str, arguments: dict[str, Any] | None = None):
        """校验参数后执行注册函数；业务错误用 isError 返回，保持连接可继续使用。"""
        from jsonschema import Draft202012Validator, ValidationError
        from mcp import types

        def error(message: str):
            return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)

        if not isinstance(name, str) or name not in self.tools:
            return error("工具不存在。")
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            return error("工具参数必须是对象。")
        tool = self.tools[name]
        try:
            # 非标准 NaN/Infinity 不应流入 JSON-RPC 或模型请求。
            json.dumps(arguments, allow_nan=False)
            Draft202012Validator(tool.input_schema).validate(arguments)
        except (ValidationError, TypeError, ValueError):
            return error("参数不符合工具 schema。")
        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = await asyncio.to_thread(tool.handler, **arguments)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, types.CallToolResult):
                return result
            if isinstance(result, str):
                return types.CallToolResult(content=[types.TextContent(type="text", text=result)], isError=False)
            if isinstance(result, dict):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, allow_nan=False))],
                    structuredContent=deepcopy(result), isError=False,
                )
            if isinstance(result, list):
                return types.CallToolResult(content=result, isError=False)
            if result is None:
                return types.CallToolResult(content=[], isError=False)
            raise TypeError("Handler must return supported MCP content")
        except Exception as exc:
            # 不回显参数、异常正文或 traceback，防止密钥/私有资料进入日志和响应。
            logger.error("Tool execution failed: %s", type(exc).__name__)
            return error(f"工具执行失败：{type(exc).__name__}。")


def create_mcp_server(server_name: str = "personal-knowledge-mcp", server_version: str = "0.1.0",
                      protocol_handler: ProtocolHandler | None = None):
    """官方 v1 SDK 负责协议与初始化，本模块接入工具发现和分发。"""
    from mcp.server.lowlevel import Server

    handler = protocol_handler if protocol_handler is not None else ProtocolHandler(server_name, server_version)
    server = Server(server_name, version=server_version)

    @server.list_tools()
    async def handle_list_tools():
        return handler.get_tool_schemas()

    # 在注册表处校验并返回脱敏错误，避免 SDK 把原始错误值直接写回客户端。
    @server.call_tool(validate_input=False)
    async def handle_call_tool(name: str, arguments: dict[str, Any]):
        return await handler.execute_tool(name, arguments)

    server._protocol_handler = handler
    return server


def get_protocol_handler(server) -> ProtocolHandler:
    """按上游提供注册表的访问入口，便于下一批接入真实业务工具。"""
    return server._protocol_handler
