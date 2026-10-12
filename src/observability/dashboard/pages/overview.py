"""G1：系统配置、集合统计与 Trace 数量。"""

import streamlit as st
from src.observability.dashboard.services.runtime import get_services


def render():
    st.title("系统总览")
    st.caption("配置说明能力选择，统计说明实际数据；二者不等于真实模型效果已验收。")
    try:
        config, data, traces = get_services()
        collections = data.list_collections()
        rows = [data.get_collection_stats(name) for name in collections]
        metrics = st.columns(4)
        for column, label, count in zip(metrics, ["集合", "文档来源", "Chunk", "图片登记"],
                                       [len(rows), sum(r["document_count"] for r in rows),
                                        sum(r["chunk_count"] for r in rows), sum(r["image_count"] for r in rows)]):
            column.metric(label, count)
        st.subheader("当前配置")
        columns = st.columns(3)
        for index, card in enumerate(config.get_component_cards()):
            with columns[index % 3].container(border=True):
                st.subheader(card.name)
                st.text(f"{card.provider} / {card.model}")
                st.json(card.extra, expanded=False)
        st.subheader("集合统计")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("尚无知识库。可以先浏览后台，再到摄取管理上传文档。")
        st.subheader("近期运行记录")
        recent = traces.list_traces(limit=100)
        st.caption(f"最近最多 100 条：查询 {sum(t['trace_type'] == 'query' for t in recent)}，摄取 {sum(t['trace_type'] == 'ingestion' for t in recent)}。")
        if traces.skipped_lines:
            st.warning(f"跳过 {traces.skipped_lines} 行非 Trace 或损坏记录。")
        st.text(f"Trace 文件：{traces.traces_path}")
    except Exception as exc:
        st.error(f"读取总览失败：{type(exc).__name__}。请检查配置和存储，未回显原始错误。")


if __name__ == "__main__":
    render()
