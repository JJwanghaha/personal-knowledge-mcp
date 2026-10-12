"""H3/H5：批量评估和可选质量门槛；check/help 不调用模型或写入数据。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.observability.dashboard.services.evaluation_service import DEFAULT_TEST_SET, EvaluationService
from src.observability.evaluation.eval_runner import load_test_set, write_report
from src.observability.evaluation.quality_gate import check_quality_gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对审核后的固定测试集运行检索与评分。实际运行会调用配置的模型。")
    parser.add_argument("--test-set", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--collection", default="default")
    parser.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--data-dir")
    parser.add_argument("--backend", choices=["custom", "ragas", "composite"], default="custom")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate-answers", action="store_true", help="显式生成待评回答；否则使用已提供回答或跳过")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--check", action="store_true", help="只检查配置和试卷格式，不打开存储或调用模型")
    parser.add_argument("--output", help="保存报告到新文件，拒绝覆盖")
    parser.add_argument("--json", action="store_true", help="stdout 输出完整私人报告，请勿公开分享")
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-mrr", type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        settings = load_settings(Path(args.config).resolve())
        cases = load_test_set(args.test_set)
        if args.top_k < 1:
            raise ValueError("top_k must be positive")
        thresholds = {name: value for name, value in (("hit_rate", args.min_hit_rate), ("mrr", args.min_mrr)) if value is not None}
        if any(not 0 < value <= 1 for value in thresholds.values()):
            raise ValueError("Retrieval thresholds must be greater than zero and at most one")
        if args.check:
            print(f"配置和 {len(cases)} 道评估题检查通过；未调用模型或创建数据。")
            return 0
        if args.output and Path(args.output).exists():
            raise FileExistsError("Report output already exists")
        report = EvaluationService(settings, data_dir=args.data_dir).run(args.test_set, args.collection,
            backend=args.backend, top_k=args.top_k, generate_answers=args.generate_answers, no_rerank=args.no_rerank)
        gate = check_quality_gate(report, thresholds) if thresholds else None
        if args.output:
            write_report(report, args.output)
        if args.json:
            payload = report.to_dict()
            payload["quality_gate"] = asdict(gate) if gate is not None else None
            print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(f"评估题数：{len(report.query_results)}；TopK：{report.top_k}；后端：{report.evaluator_name}")
            for metric, value in report.aggregate_metrics.items():
                print(f"{metric}={value:.4f}，评分覆盖 {report.metric_counts[metric]}/{len(cases)}")
            for row in report.query_results:
                print(f"{row.case_id}：{row.status}；证据口径={row.label_scope}；错误={row.errors}；跳过={row.skipped}")
            if not report.aggregate_metrics:
                print("没有可评指标；缺失标签/答案或执行失败不是有效零分。")
            if gate is not None:
                print(f"质量门槛：{'通过' if gate.passed else '未通过'}；{gate.failures}")
        failed = any(row.errors for row in report.query_results) or not report.aggregate_metrics or gate is not None and not gate.passed
        return 1 if failed else 0
    except Exception as exc:
        print(f"评估入口失败：{type(exc).__name__}。请检查配置、试卷审核状态、集合及模型环境。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
