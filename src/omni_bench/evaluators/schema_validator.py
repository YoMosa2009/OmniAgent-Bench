import json
import re
from typing import Dict, Any, Tuple
from .base import BaseEvaluator

class ToolSchemaEvaluator(BaseEvaluator):
    """
    Evaluates agentic tool calling and schema adherence.
    Validates function name and arguments flexibly without brittle syntax traps.
    """

    def evaluate(self, response: str, task: Dict[str, Any]) -> Tuple[bool, float, str]:
        metadata = task.get("metadata", {})
        expected_tool = metadata.get("expected_tool")
        expected_args = metadata.get("expected_args", {})
        arg_validators = metadata.get("arg_validators", {})

        # Extract JSON object (handles <tool_call> tags, ```json blocks, or bare JSON)
        clean_text = re.sub(r"</?tool_call>", "", response)
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            return False, 0.0, "FAIL: Could not extract valid JSON tool call object."

        try:
            parsed = json.loads(json_match.group(0))
        except Exception as e:
            return False, 0.0, f"FAIL: Malformed JSON syntax: {e}"

        # Normalize schema: either {"name": "...", "arguments": {...}} or direct tool invocation
        tool_name = parsed.get("name") or parsed.get("function")
        args = parsed.get("arguments") or parsed.get("parameters") or parsed

        if not tool_name and expected_tool in parsed:
            tool_name = expected_tool
            args = parsed[expected_tool]

        if tool_name != expected_tool:
            return False, 0.0, f"FAIL: Expected tool '{expected_tool}', but model called '{tool_name}'"

        # Check required arguments
        if not isinstance(args, dict):
            return False, 0.0, f"FAIL: Arguments must be a JSON dictionary, got {type(args).__name__}"

        # Validate exact arguments if specified
        for k, v in expected_args.items():
            if k not in args:
                return False, 0.5, f"FAIL: Missing required argument '{k}'"
            if str(args[k]).lower().strip() != str(v).lower().strip():
                return False, 0.5, f"FAIL: Argument '{k}' value '{args[k]}' does not match expected '{v}'"

        # Validate fuzzy substring constraints on arguments if specified
        for arg_key, keywords in arg_validators.items():
            val = str(args.get(arg_key, "")).lower()
            for kw in keywords:
                if kw.lower() not in val:
                    return False, 0.5, f"FAIL: Argument '{arg_key}' does not include expected term '{kw}'"

        return True, 1.0, f"PASS: Correctly invoked '{expected_tool}' with valid schema arguments."
