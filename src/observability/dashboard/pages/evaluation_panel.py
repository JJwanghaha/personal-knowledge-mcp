"""H4：固定试卷、批量评估、质量门槛、报告历史与显式出题。"""

from dataclasses import asdict
import json
import os
import streamlit as st

from src.observability.dashboard.services.evaluation_service import DEFAULT_TEST_SET, EvaluationService
from src.observability.dashboard.services.runtime import get_services
from src.observability.evaluation.eval_runner import load_test_set
from src.observability.evaluation.quality_gate import check_quality_gate


def _show_report(report):
    st.subheader("评估报告")
    count = report.get("query_count", 0)
    st.caption(f"固定试卷 SHA256：{report['test_set_sha256']} · 集合 {report.get('collection')} · TopK {report.get('top_k')}")
    metrics = report.get("aggregate_metrics", {})
    if metrics:
        st.dataframe([{"指标": name, "均值": value, "评分题数": report.get("metric_counts", {}).get(name, 0), "总题数": count}
                      for name, value in metrics.items()], hide_index=True, width="stretch")
    else:
        st.warning("没有可评指标。缺失标签/真实答案或失败不会显示为零分。")
    st.json(report.get("status_counts", {}), expanded=False)
    for row in report.get("query_results", []):
        with st.expander(f"{row['case_id']} · {row['status']} · {row['query']}"):
            st.text(f"证据口径：{row['label_scope']}；待评答案来源：{row['answer_source']}")
            st.json({"metrics": row["metrics"], "errors": row["errors"], "skipped": row["skipped"]}, expanded=False)
            if row.get("generated_answer"):
                st.caption("实际待评回答")
                st.code(row["generated_answer"], language=None)
            if row.get("reference_answer"):
                st.caption("参考答案（不是待评回答）")
                st.code(row["reference_answer"], language=None)
            for chunk in row.get("retrieved_chunks", []):
                st.text(f"{chunk['chunk_id']} · {chunk.get('source', '')} · {chunk['score']}")
                st.code(chunk["text"], language=None)
    st.download_button("下载私人评估报告 JSON", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="evaluation-report.json", mime="application/json")


def render():
    st.title("评估面板")
    st.info("先审核固定测试集，再运行评估。规则打分本身不调用模型，但 Dense 检索仍调用 Embedding；Ragas/生成答案/出题还会调用 LLM。")
    try:
        config, data, _ = get_services()
        service = EvaluationService(config.settings, data_dir=os.environ.get("RAG_DASHBOARD_DATA_DIR"), embedding_client=data.embedding_client)
        collections = data.list_collections()
        collection = st.selectbox("评估集合", collections or ["default"], key="eval_collection")
        test_path = st.text_input("审核后的固定测试集路径", value=str(DEFAULT_TEST_SET), key="eval_test_path")
        try:
            cases = load_test_set(test_path)
            st.caption(f"已加载 {len(cases)} 道题。expected_sources 只评文档来源，精确片段请标 expected_chunk_ids。")
            with st.expander("预览固定试卷"):
                st.dataframe([asdict(case) for case in cases], hide_index=True, width="stretch")
        except Exception as exc:
            st.warning(f"试卷尚不可评：{type(exc).__name__}。请检查格式和 reviewed 审核状态。")
        with st.form("evaluation_form"):
            backend = st.selectbox("评估后端", ["custom", "ragas", "composite"], key="eval_backend")
            top_k = st.number_input("TopK", min_value=1, max_value=50, value=5, step=1)
            generate_answers = st.checkbox("使用生成模型产生待评答案（不会使用参考答案作提示词）")
            no_rerank = st.checkbox("本次关闭重排", value=True)
            gate_enabled = st.checkbox("启用检索质量门槛")
            minimum_hit = st.number_input("最低命中率", min_value=0.01, max_value=1.0, value=0.8)
            minimum_mrr = st.number_input("最低 MRR", min_value=0.01, max_value=1.0, value=0.5)
            confirmed = st.checkbox("确认执行检索/评分并使用所选模型 API；报告包含私人正文", key="confirm_evaluation")
            run = st.form_submit_button("运行固定测试集")
        if run:
            if not confirmed:
                st.warning("未确认模型调用，未运行评估。")
            elif collection not in collections:
                st.warning("请先摄取所选试卷的资料，当前没有可用集合。")
            else:
                with st.spinner("逐题检索和评分中……"):
                    report = service.run(test_path, collection, backend=backend, top_k=int(top_k),
                                         generate_answers=generate_answers, no_rerank=no_rerank)
                path = service.save_report(report)
                if gate_enabled:
                    gate = check_quality_gate(report, {"hit_rate": minimum_hit, "mrr": minimum_mrr})
                    (st.success if gate.passed else st.error)(f"质量门槛：{'通过' if gate.passed else '未通过'}；{gate.failures}")
                st.session_state["evaluation_last_report"] = report.to_dict()
                st.caption(f"报告保存到：{path}")
        if "evaluation_last_report" in st.session_state:
            _show_report(st.session_state["evaluation_last_report"])
        st.divider()
        st.subheader("历史报告")
        history = service.list_reports()
        if history:
            rows = [{"时间": item["report"].get("created_at"), "集合": item["report"].get("collection"),
                     "TopK": item["report"].get("top_k"), "试卷SHA": item["report"]["test_set_sha256"],
                     "语料指纹": item["report"].get("configuration", {}).get("corpus_fingerprint"),
                     **item["report"].get("aggregate_metrics", {})} for item in history]
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("不同试卷或语料指纹不能直接比较分数；参数改变应连同覆盖率和失败样例一起看。")
            choice = st.selectbox("查看历史报告", range(len(history)), key="eval_history",
                                  format_func=lambda i: history[i]["report"].get("created_at", history[i]["path"]))
            with st.expander("历史报告详情"):
                st.json(history[choice]["report"], expanded=False)
        else:
            st.info("暂无历史报告；浏览本页不会创建报告目录或调用模型。")
        st.divider()
        st.subheader("LLM 生成候选测试题")
        st.caption("先选择已有文档，再显式调用生成模型。返回 reviewed=false 草稿，审核后另存为固定试卷；不会生成后立刻自评。")
        documents = data.list_documents(collection) if collection in collections else []
        with st.form("draft_generation_form"):
            sources = st.multiselect("出题资料（仅发送选中来源的 Chunk 正文）", [doc["source_path"] for doc in documents])
            count = st.number_input("候选题数", min_value=1, max_value=50, value=10, step=1)
            generate_confirmed = st.checkbox("允许把所选资料发送给配置的 LLM 并消耗额度", key="confirm_draft_generation")
            generate = st.form_submit_button("生成未审核草稿")
        if generate:
            if not generate_confirmed or not sources:
                st.warning("请先选择资料并确认发送给模型。")
            else:
                with st.spinner("生成候选题中……"):
                    draft = service.generate_draft(collection, sources, count=int(count))
                path = service.save_draft(draft)
                st.session_state["evaluation_draft"] = draft
                st.caption(f"草稿保存到：{path}；人工审核后才能运行。")
        if "evaluation_draft" in st.session_state:
            draft = st.session_state["evaluation_draft"]
            st.json(draft, expanded=False)
            st.download_button("下载未审核草稿", json.dumps(draft, ensure_ascii=False, indent=2),
                               file_name="testset-draft.json", mime="application/json")
    except Exception as exc:
        st.error(f"评估操作失败：{type(exc).__name__}。请检查配置与模型环境，未回显原始异常。")


if __name__ == "__main__":
    render()
