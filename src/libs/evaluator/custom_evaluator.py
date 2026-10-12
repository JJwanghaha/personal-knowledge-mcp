"""无需模型的确定性检索评估，沿用上游 hit_rate/mrr 命名。"""

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """单次查询计算命中和倒数排名；跨问题平均由后续 EvalRunner 负责。"""

    SUPPORTED_METRICS = {"hit_rate", "mrr"}

    def __init__(self, settings=None, metrics=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs
        if metrics is None and settings is not None:
            metrics = settings.evaluation.metrics
        self.metrics = [str(name).strip().lower() for name in (metrics or ["hit_rate", "mrr"])]
        unsupported = [name for name in self.metrics if name not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(f"Unsupported custom metrics: {', '.join(unsupported)}")

    def evaluate(
        self, query, retrieved_chunks, generated_answer=None, ground_truth=None,
        trace=None, **kwargs,
    ) -> dict[str, float]:
        """找出第一个相关候选的排名。"""
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        retrieved_ids = self._extract_ids(retrieved_chunks)
        expected_ids = self._ground_truth_ids(ground_truth)
        first_rank = next(
            (rank for rank, item in enumerate(retrieved_ids, start=1) if item in expected_ids),
            None,
        )
        values = {
            "hit_rate": 1.0 if first_rank is not None else 0.0,
            "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        }
        return {name: values[name] for name in self.metrics}

    def _extract_ids(self, items) -> list[str]:
        """接受上游的字符串 ID、记录字典和带 id 的对象。"""
        identifiers = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                identifiers.append(item)
            elif isinstance(item, dict):
                for field in ("id", "chunk_id", "document_id", "doc_id"):
                    if field in item:
                        identifiers.append(str(item[field]))
                        break
                else:
                    raise ValueError(f"Missing id field at index {index}")
            elif hasattr(item, "chunk_id"):
                identifiers.append(str(item.chunk_id))
            elif hasattr(item, "id"):
                identifiers.append(str(item.id))
            else:
                raise ValueError(f"Unable to extract id at index {index}")
        return identifiers

    def _ground_truth_ids(self, ground_truth) -> list[str]:
        """把标准答案的不同形状转换为 ID 列表。"""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth)
        if isinstance(ground_truth, dict):
            if isinstance(ground_truth.get("ids"), list):
                return self._extract_ids(ground_truth["ids"])
            return self._extract_ids([ground_truth])
        raise ValueError(f"Unsupported ground_truth type: {type(ground_truth).__name__}")
