"""G3：查看实际文档、Chunk 正文、metadata 与受管理图片。"""

import base64
import streamlit as st

from src.core.types import RetrievalResult
from src.core.response.multimodal_assembler import MultimodalAssembler
from src.observability.dashboard.services.runtime import get_services


def render():
    st.title("数据浏览")
    try:
        _, data, _ = get_services()
        collections = data.list_collections()
        if not collections:
            st.info("暂无集合；浏览不会创建空知识库。")
            return
        collection = st.selectbox("知识库集合", collections, key="browse_collection")
        documents = data.list_documents(collection)
        if not documents:
            st.info("该集合没有可浏览的文档。")
            return
        st.dataframe(documents, hide_index=True, width="stretch")
        source = st.selectbox("查看文档", [doc["source_path"] for doc in documents], key="browse_document")
        doc = next(d for d in documents if d["source_path"] == source)
        st.text(source)
        st.caption(f"实际块数 {doc['chunk_count']} · 图片登记 {doc['image_count']} · 历史状态 {doc['status']}")
        st.json(data.get_document_detail(doc["source_hash"] or source, collection, source_path=source), expanded=False)
        chunks = data.get_chunks(doc["source_hash"], collection, source_path=source)
        for index, chunk in enumerate(chunks):
            with st.expander(f"Chunk {index + 1} · {chunk['id']}"):
                st.code(chunk["text"], language=None)
                st.json(chunk["metadata"], expanded=False)
        assembler = MultimodalAssembler(images_root=data.data_root / "images",
                                       image_index_path=data.data_root / "db" / "image_index.db")
        blocks = assembler.assemble([RetrievalResult(c["id"], 0, c["text"], c["metadata"]) for c in chunks], collection)
        if blocks:
            st.subheader("关联图片（最多五张，验证实际格式与路径）")
            for index, block in enumerate(blocks, 1):
                st.image(base64.b64decode(block.data), caption=f"图片 {index}", width=350)
    except Exception as exc:
        st.error(f"读取文档失败：{type(exc).__name__}。")


if __name__ == "__main__":
    render()
