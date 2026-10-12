"""页面统一读取启动参数；session state 不存长期数据库连接。"""

import os

from src.observability.dashboard.services.config_service import ConfigService
from src.observability.dashboard.services.data_service import DataService
from src.observability.dashboard.services.trace_service import TraceService


def get_services():
    config = ConfigService(os.environ.get("RAG_DASHBOARD_CONFIG"))
    data = DataService(config.settings, data_dir=os.environ.get("RAG_DASHBOARD_DATA_DIR"))
    traces = TraceService(config.settings.observability.trace_file)
    return config, data, traces
