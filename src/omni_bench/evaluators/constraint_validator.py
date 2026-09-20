import json
import re
from typing import Dict, Any, Tuple
from .base import BaseEvaluator

class ConstraintEvaluator(BaseEvaluator):
    """
    Evaluates format adherence, word-count limits, and negative constraints
    without false-negative substring traps.
    """

    def evaluate(self, response: str, task: Dict[str, Any]) -> Tuple[bool, float, str]:
        metadata = task.get("metadata", {})
        expected_json = metadata.get("expected_json")
        max_words = metadata.get("max_words")
        forbidden_words = metadata.get("forbidden_words", [])

        # Check JSON constraint if specified
        if expected_json is not None:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return False, 0.0, "FAIL: Response does not contain a JSON object."
            try:
                parsed = json.loads(m.group(0))
                matched_keys = 0
                for k, v in expected_json.items():
                    if k in parsed and str(parsed[k]).lower().strip() == str(v).lower().strip():
                        matched_keys += 1
                ratio = matched_keys / len(expected_json)
                if ratio == 1.0:
                    return True, 1.0, "PASS: JSON keys and values matched target schema exactly."
                return False, ratio, f"FAIL: Matched {matched_keys}/{len(expected_json)} JSON key-value pairs."
            except Exception as e:
                return False, 0.0, f"FAIL: JSON parsing error: {e}"

        # Check word count limit if specified
        if max_words is not None:
            words = response.strip().split()
            if len(words) > max_words:
                return False, 0.5, f"FAIL: Exceeded word limit ({len(words)} > {max_words} words)."

        # Check forbidden words
        resp_lower = response.lower()
        for fw in forbidden_words:
            if fw.lower() in resp_lower:
                return False, 0.0, f"FAIL: Response contains forbidden word '{fw}'."

        return True, 1.0, "PASS: All constraints satisfied."
