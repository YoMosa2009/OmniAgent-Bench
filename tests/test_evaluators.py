import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omni_bench.evaluators.code_runner import SandboxCodeEvaluator
from omni_bench.evaluators.schema_validator import ToolSchemaEvaluator
from omni_bench.evaluators.constraint_validator import ConstraintEvaluator

def test_code_evaluator():
    evaluator = SandboxCodeEvaluator(timeout_seconds=5)
    task = {
        "metadata": {
            "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        }
    }
    
    # Valid code
    good_resp = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    passed, ratio, msg = evaluator.evaluate(good_resp, task)
    assert passed is True, f"Failed on good code: {msg}"
    print("[TEST] SandboxCodeEvaluator PASS (valid code)")

    # Broken code
    bad_resp = "```python\ndef add(a, b):\n    return a - b\n```"
    passed, ratio, msg = evaluator.evaluate(bad_resp, task)
    assert passed is False, "Allowed broken code!"
    print("[TEST] SandboxCodeEvaluator PASS (broken code rejected)")

def test_schema_evaluator():
    evaluator = ToolSchemaEvaluator()
    task = {
        "metadata": {
            "expected_tool": "fetch_weather",
            "expected_args": {"location": "Seattle"}
        }
    }
    good_resp = '<tool_call>\n{"name": "fetch_weather", "arguments": {"location": "Seattle"}}\n</tool_call>'
    passed, ratio, msg = evaluator.evaluate(good_resp, task)
    assert passed is True, f"Failed on tool call: {msg}"
    print("[TEST] ToolSchemaEvaluator PASS")

def test_constraint_evaluator():
    evaluator = ConstraintEvaluator()
    task = {
        "metadata": {
            "expected_json": {"tier": "pro"},
            "max_words": 10
        }
    }
    good_resp = '{"tier": "pro"}'
    passed, ratio, msg = evaluator.evaluate(good_resp, task)
    assert passed is True, f"Failed constraint: {msg}"
    print("[TEST] ConstraintEvaluator PASS")

if __name__ == "__main__":
    test_code_evaluator()
    test_schema_evaluator()
    test_constraint_evaluator()
    print("\nALL EVALUATOR SUITE TESTS PASSED SUCCESSFULLY!")
