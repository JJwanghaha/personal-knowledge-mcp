"""G5：从完整 Trace 查看解析、切分、清理前后、编码和存储映射。"""

import streamlit as st

from src.observability.dashboard.pages.trace_views import select_trace, show_timings
from src.observability.dashboard.services.runtime import get_services


def render():
    st.title("摄取追踪")
    try:
        _, _, service = get_services()
        trace = select_trace(service, "ingestion")
        if trace is None:
            return
        timings = show_timings(service, trace)
        st.json(trace.get("metadata", {}), expanded=False)
        for index, timing in enumerate(timings):
            name, data = timing["stage_name"], timing["data"]
            with st.expander(f"{index + 1}. {name} · {data.get('status', '')}"):
                if name == "load":
                    st.code(data.get("text_preview", ""), language=None)
                if name == "split":
                    for chunk in data.get("chunks", []):
                        st.text(chunk.get("chunk_id", ""))
                        st.code(chunk.get("text", ""), language=None)
                if name == "transform":
                    for chunk in data.get("chunks", []):
                        st.text(chunk.get("chunk_id", ""))
                        before, after = st.columns(2)
                        before.caption("处理前")
                        before.code(chunk.get("text_before", ""), language=None)
                        after.caption("处理后")
                        after.code(chunk.get("text_after", ""), language=None)
                st.json(data, expanded=False)
        with st.expander("完整 Trace（包含私人正文，不要公开分享）"):
            st.json(trace)
    except Exception as exc:
        st.error(f"读取摄取记录失败：{type(exc).__name__}。")


if __name__ == "__main__":
    render()
