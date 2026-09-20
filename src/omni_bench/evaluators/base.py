import abc
from typing import Dict, Any, Tuple

class BaseEvaluator(abc.ABC):
    """Abstract interface for all task evaluators in OmniAgent-Bench."""

    @abc.abstractmethod
    def evaluate(self, response: str, task: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluates a model's response.
        Returns:
            passed (bool): Whether the response passes the evaluation.
            score_ratio (float): Partial credit score between 0.0 and 1.0.
            feedback (str): Detailed explanation of pass/fail criteria.
        """
        pass
