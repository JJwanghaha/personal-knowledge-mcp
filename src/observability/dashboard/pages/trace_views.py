"""G5/G6 共用 Trace 选择、阶段图和候选全文显示。"""

import streamlit as st


def select_trace(service, kind):
    keyword = st.text_input("搜索问题 / 来源 / Trace ID", key=f"{kind}_keyword")
    traces = service.list_traces(kind, limit=100, keyword=keyword)
    if service.skipped_lines:
        st.warning(f"已跳过 {service.skipped_lines} 行损坏数据或普通日志；未回显原始行。")
    if not traces:
        st.info("暂无匹配的记录。先运行摄取或查询，并开启 trace_enabled。")
        return None
    collections = sorted({str(t.get("metadata", {}).get("collection", "")) for t in traces})
    selected = st.selectbox("集合筛选", ["全部", *collections], key=f"{kind}_collection")
    if selected != "全部":
        traces = [t for t in traces if str(t.get("metadata", {}).get("collection", "")) == selected]
    st.dataframe([{"trace_id": t["trace_id"], "开始时间": t.get("started_at"),
                   "状态": t.get("metadata", {}).get("status", "unknown"),
                   "总耗时毫秒": t.get("total_elapsed_ms"),
                   "问题或来源": t.get("metadata", {}).get("query", t.get("metadata", {}).get("source_path", ""))}
                  for t in traces], hide_index=True, width="stretch")
    choice = st.selectbox("查看 Trace", range(len(traces)), key=f"{kind}_trace",
                          format_func=lambda i: f"{traces[i].get('started_at', '')} · {traces[i]['trace_id']}")
    return traces[choice]


def show_timings(service, trace):
    st.text(f"Trace ID：{trace['trace_id']}")
    st.metric("独立测量的总耗时（毫秒）", trace.get("total_elapsed_ms", "—"))
    timings = service.get_stage_timings(trace)
    st.caption("以下是阶段耗时分布，不是精确执行瀑布图；并行检索耗时可重叠，不能简单相加。")
    if timings:
        st.bar_chart([{"阶段": f"{i + 1}. {t['stage_name']}", "毫秒": t["elapsed_ms"]} for i, t in enumerate(timings)],
                     x="阶段", y="毫秒")
        st.dataframe([{"阶段": t["stage_name"], "毫秒": t["elapsed_ms"],
                       "方法": t["data"].get("method"), "状态": t["data"].get("status", "unknown")}
                      for t in timings], hide_index=True, width="stretch")
    return timings


def show_candidates(chunks, prefix):
    if not chunks:
        st.info("没有候选。可能为空结果、跳过或该阶段失败。")
        return
    st.dataframe([{"排名": i + 1, "chunk_id": c.get("chunk_id"), "分数": c.get("score"),
                   "来源": c.get("source")} for i, c in enumerate(chunks)], hide_index=True, width="stretch")
    for i, chunk in enumerate(chunks):
        with st.expander(f"{prefix} 候选 {i + 1} · {chunk.get('chunk_id', '')}"):
            st.code(chunk.get("text", ""), language=None)
            st.json({key: value for key, value in chunk.items() if key != "text"}, expanded=False)
