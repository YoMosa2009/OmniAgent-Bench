from typing import Optional
from .base import BaseEvaluator
from .code_runner import SandboxCodeEvaluator
from .schema_validator import ToolSchemaEvaluator
from .llm_judge import LLMJudgeEvaluator
from .constraint_validator import ConstraintEvaluator
from ..models.base import BaseModelProvider

def get_evaluator(eval_type: str, judge_provider: Optional[BaseModelProvider] = None) -> BaseEvaluator:
    t = eval_type.lower().strip()
    if t == "unit_test":
        return SandboxCodeEvaluator()
    elif t == "tool_schema":
        return ToolSchemaEvaluator()
    elif t in ("llm_judge", "judge", "security", "long_horizon"):
        return LLMJudgeEvaluator(judge_provider=judge_provider)
    elif t in ("constraint", "json", "format"):
        return ConstraintEvaluator()
    else:
        return LLMJudgeEvaluator(judge_provider=judge_provider)
