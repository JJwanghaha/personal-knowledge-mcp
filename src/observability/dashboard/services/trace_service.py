"""G5/G6：只读 Trace JSONL，区分完整执行记录与普通结构化日志。"""

import json
from math import isfinite

from src.core.settings import resolve_path


class TraceService:
    def __init__(self, traces_path="logs/traces.jsonl"):
        self.traces_path = resolve_path(traces_path)
        self.skipped_lines = 0

    def _load_all(self):
        self.skipped_lines = 0
        if not self.traces_path.is_file():
            return []
        result = []
        with self.traces_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    self.skipped_lines += 1
                    continue
                if (not isinstance(value, dict) or not isinstance(value.get("trace_type"), str)
                        or value.get("trace_type") not in {"query", "ingestion"}
                        or not isinstance(value.get("trace_id"), str) or not isinstance(value.get("stages"), list)
                        or not isinstance(value.get("metadata", {}), dict)):
                    self.skipped_lines += 1
                    continue
                result.append(value)
        return result

    def list_traces(self, trace_type=None, limit=100, *, keyword="", collection=None):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Trace limit must be positive")
        traces = self._load_all()
        traces = [t for t in traces if (trace_type is None or t["trace_type"] == trace_type)
                  and (collection is None or t.get("metadata", {}).get("collection") == collection)
                  and (not keyword or keyword.casefold() in json.dumps(t, ensure_ascii=False).casefold())]
        return sorted(traces, key=lambda t: str(t.get("started_at", "")), reverse=True)[:limit]

    def get_trace(self, trace_id):
        return next((t for t in reversed(self._load_all()) if t["trace_id"] == trace_id), None)

    def get_stage_timings(self, trace):
        result = []
        for stage in trace.get("stages", []):
            if not isinstance(stage, dict):
                continue
            duration = stage.get("elapsed_ms", 0)
            duration = duration if isinstance(duration, (int, float)) and not isinstance(duration, bool) and isfinite(duration) and duration >= 0 else 0
            result.append({"stage_name": str(stage.get("stage", "unknown")), "elapsed_ms": duration,
                           "data": stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}})
        return result
