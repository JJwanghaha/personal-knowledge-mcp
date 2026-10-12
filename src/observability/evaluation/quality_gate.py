"""H5：检查分数下限、评分覆盖与执行错误，不靠降低门槛自动通过。"""

from dataclasses import dataclass
from math import isfinite

from src.observability.evaluation.eval_runner import EvalReport


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def check_quality_gate(report: EvalReport, thresholds: dict[str, float]) -> GateResult:
    if not thresholds:
        raise ValueError("Quality gate requires explicit thresholds")
    failures = []
    count = len(report.query_results)
    if not count:
        failures.append("empty_test_set")
    if any(row.errors for row in report.query_results):
        failures.append("execution_errors")
    for metric, minimum in thresholds.items():
        if not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not isfinite(minimum):
            raise ValueError("Thresholds must be finite numeric values")
        if metric in {"hit_rate", "mrr"} and not 0 < minimum <= 1:
            raise ValueError("Retrieval quality thresholds must be greater than zero and at most one")
        if metric not in report.aggregate_metrics:
            failures.append(f"missing_metric:{metric}")
        elif (not isinstance(report.aggregate_metrics[metric], (int, float)) or isinstance(report.aggregate_metrics[metric], bool)
              or not isfinite(report.aggregate_metrics[metric])):
            failures.append(f"invalid_metric:{metric}")
        elif report.metric_counts.get(metric) != count:
            failures.append(f"incomplete_coverage:{metric}")
        elif report.aggregate_metrics[metric] < minimum:
            failures.append(f"below_threshold:{metric}")
    return GateResult(not failures, failures)
