"""G6：双路候选对照、融合与重排名次变化；不混用分数尺度。"""

import streamlit as st

from src.observability.dashboard.pages.trace_views import select_trace, show_candidates, show_timings
from src.observability.dashboard.services.runtime import get_services


def render():
    st.title("查询追踪")
    try:
        _, _, service = get_services()
        trace = select_trace(service, "query")
        if trace is None:
            return
        timings = show_timings(service, trace)
        by_name = {t["stage_name"]: t["data"] for t in timings}
        st.subheader("原始问题")
        st.code(trace.get("metadata", {}).get("query", by_name.get("query_processing", {}).get("original_query", "")), language=None)
        st.json(by_name.get("query_processing", {}), expanded=False)
        st.caption("Dense、BM25、RRF 和重排的分数尺度不同，分数不是正确率，不按统一阈值标红/标绿。")
        dense, sparse = st.columns(2)
        with dense:
            st.subheader("Dense 召回")
            st.json({k: v for k, v in by_name.get("dense_retrieval", {}).items() if k != "chunks"}, expanded=False)
            show_candidates(by_name.get("dense_retrieval", {}).get("chunks", []), "Dense")
        with sparse:
            st.subheader("BM25 召回")
            st.json({k: v for k, v in by_name.get("sparse_retrieval", {}).items() if k != "chunks"}, expanded=False)
            show_candidates(by_name.get("sparse_retrieval", {}).get("chunks", []), "BM25")
        st.subheader("融合 / 过滤结果")
        show_candidates(by_name.get("fusion", {}).get("chunks", []), "融合")
        st.subheader("重排前后")
        rerank = by_name.get("rerank", {})
        before = rerank.get("input_chunks", []) or by_name.get("fusion", {}).get("chunks", [])
        after = rerank.get("chunks", []) or trace.get("metadata", {}).get("final_results", [])
        ranks = {c["chunk_id"]: i for i, c in enumerate(before, 1)}
        if after:
            st.dataframe([{"chunk_id": c["chunk_id"], "重排前名次": ranks.get(c["chunk_id"]),
                           "重排后名次": i, "当前分数": c.get("score")} for i, c in enumerate(after, 1)],
                         hide_index=True, width="stretch")
        st.json({k: v for k, v in rerank.items() if k not in {"chunks", "input_chunks"}}, expanded=False)
        show_candidates(after, "最终")
        with st.expander("完整 Trace（含全文）"):
            st.json(trace)
    except Exception as exc:
        st.error(f"读取查询记录失败：{type(exc).__name__}。")


if __name__ == "__main__":
    render()
