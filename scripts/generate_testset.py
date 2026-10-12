"""H3：从显式选择的已索引资料生成候选试卷，需要人工审核再评分。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.observability.dashboard.services.evaluation_service import EvaluationService
from src.observability.evaluation.testset_generator import save_draft


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成未审核测试集草稿，会把选择的资料发送给配置的 LLM。")
    parser.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--data-dir")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--source", action="append", required=True, help="已索引的准确来源路径，可重复选择")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-model-call", action="store_true", help="确认发送所选资料给 LLM 并使用 API 额度")
    args = parser.parse_args(argv)
    if not args.confirm_model_call:
        print("未确认模型调用，未读取文档或执行出题。", file=sys.stderr)
        return 2
    try:
        if Path(args.output).exists():
            raise FileExistsError("Draft output already exists")
        service = EvaluationService(load_settings(Path(args.config).resolve()), data_dir=args.data_dir)
        draft = service.generate_draft(args.collection, args.source, count=args.count)
        save_draft(draft, args.output)
        print(f"已生成 {len(draft['test_cases'])} 道候选题；请审核后设置 reviewed=true。草稿不能直接评分。")
        return 0
    except Exception as exc:
        print(f"出题失败：{type(exc).__name__}。未回显原始模型错误。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
