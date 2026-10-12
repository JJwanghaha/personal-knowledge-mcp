"""G/H4：六页面本地管理后台。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st


def main():
    st.set_page_config(page_title="个人知识库 · RAG 后台", page_icon="📚", layout="wide")
    st.sidebar.title("📚 个人知识库")
    st.sidebar.caption("管理资料 · 查看检索 · 理解每一步")
    st.sidebar.info("本地管理工具，非个人主页。浏览不调用模型；摄取、评估和出题需明确确认模型调用。")
    if st.sidebar.button("刷新页面数据", key="refresh_dashboard"):
        st.rerun()
    pages = [st.Page("pages/overview.py", title="系统总览", icon="📊", default=True),
             st.Page("pages/data_browser.py", title="数据浏览", icon="📚"),
             st.Page("pages/ingestion_manager.py", title="摄取管理", icon="📥"),
             st.Page("pages/ingestion_traces.py", title="摄取追踪", icon="🔬"),
             st.Page("pages/query_traces.py", title="查询追踪", icon="🔎"),
             st.Page("pages/evaluation_panel.py", title="评估面板", icon="📏")]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
