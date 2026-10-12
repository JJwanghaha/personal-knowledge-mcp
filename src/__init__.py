"""Modular RAG MCP Server 源码包。"""

import os

# 项目入口先关闭 ONNX Runtime 的原生遥测，再由 Chroma 等可选依赖导入它。
# 仅调用 disable_telemetry_events() 太晚：初始化时可能已启动上传线程。
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
