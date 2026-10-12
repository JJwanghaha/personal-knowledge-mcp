# MCP 本地通信

MCP 服务使用 stdio 与宿主交换协议消息。stdout 专用于协议，运行日志写到 stderr，避免日志污染通信。

query_knowledge_hub 返回检索证据与结构化来源引用，不生成最终回答。list_collections 列出已有集合，get_document_summary 回查文档摘要。

启动服务或列出工具不调用模型，也不创建知识库。实际向量查询才使用配置的 Embedding 服务。
