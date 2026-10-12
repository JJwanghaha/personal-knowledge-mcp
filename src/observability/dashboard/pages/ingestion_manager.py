"""G4：明确确认后摄取或删除；页面刷新不执行写操作。"""

import streamlit as st
from src.observability.dashboard.services.runtime import get_services

STAGE_LABELS = {"integrity": "文件检查", "load": "文档解析", "split": "切分", "transform": "清理与增强", "embed": "编码", "upsert": "入库与成功登记"}


def render():
    st.title("摄取管理")
    notice = st.session_state.pop("document_delete_notice", None)
    if notice:
        st.success(notice)
    try:
        _, data, _ = get_services()
        st.info("支持 PDF、MD、Markdown；可同时上传同级 PNG/JPEG/WebP 图片附件。嵌套目录图片、DOCX、OCR 尚未支持。上传源保留在 data/uploads；删除索引不删除源文件。")
        with st.form("ingestion_form"):
            collection = st.text_input("目标集合", value="default", key="ingest_collection")
            files = st.file_uploader("选择文档与图片附件", type=["pdf", "md", "markdown", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
            force = st.checkbox("强制重新摄取（不自动清理旧向量版本）")
            overwrite = st.checkbox("允许覆盖同名上传源（旧内容保留备份）")
            confirmed = st.checkbox("确认读取所选资料并使用配置的模型 API；可能产生费用", key="confirm_ingestion")
            submit = st.form_submit_button("开始摄取")
        if submit:
            if not confirmed or not files:
                st.warning("请先选择资料并确认模型调用。")
            else:
                bar = st.progress(0.0)
                status = st.empty()

                def on_progress(stage, current, total):
                    bar.progress(current / total)
                    status.text(f"{current}/{total}：{STAGE_LABELS.get(stage, stage)}（按阶段计数，不是耗时百分比）")

                with st.spinner("正在摄取……"):
                    outcomes = data.ingest_uploads([(file.name, file.getvalue()) for file in files], collection.strip(),
                                                  force=force, overwrite=overwrite, on_progress=on_progress)
                for outcome in outcomes:
                    result = outcome["result"]
                    label = "已跳过重复文件" if result["stages"].get("integrity", {}).get("skipped") else "摄取成功"
                    if result["success"]:
                        st.success(f"{label}：{result['chunk_count']} 个块，{result['image_count']} 张图片。")
                    else:
                        st.error(f"摄取失败：{result['error']}；进度不代表最终成功。")
                    st.json(outcome, expanded=False)
        st.divider()
        st.subheader("删除文档索引")
        collections = data.list_collections()
        if not collections:
            st.info("暂无可删除的文档。")
            return
        selected = st.selectbox("删除范围：集合", collections, key="delete_collection")
        documents = data.list_documents(selected)
        if not documents:
            st.info("该集合暂无文档。")
            return
        source = st.selectbox("删除范围：文档来源", [d["source_path"] for d in documents], key="delete_source")
        doc = next(d for d in documents if d["source_path"] == source)
        st.warning(f"只删除此来源在 {selected} 集合的全部索引版本：{doc['chunk_count']} 个块；不会删除源文件。受管理图片移到 .trash，多存储删除可能部分失败。")
        token = f"{selected}:{source}"
        with st.form("delete_form"):
            confirmation = st.text_input("输入所选源文件路径以确认", key="delete_confirmation_" + token)
            delete = st.form_submit_button("确认删除该文档")
        if delete:
            if confirmation != source:
                st.warning("确认路径不一致，未执行删除。")
            else:
                result = data.delete_document(source, selected, doc["source_hash"] or None)
                if result.success:
                    st.session_state["document_delete_notice"] = f"已删除 {result.chunks_deleted} 个块和 {result.images_deleted} 条图片登记；源文件保留，移出的图片可从 .trash 恢复。"
                    st.rerun()
                else:
                    st.error(f"部分删除失败：{result.errors}。历史保留，可检查后重试。")
    except Exception as exc:
        st.error(f"管理操作失败：{type(exc).__name__}。没有回显原始异常。")


if __name__ == "__main__":
    render()
