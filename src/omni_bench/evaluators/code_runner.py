import os
import re
import sys
import tempfile
import subprocess
from typing import Dict, Any, Tuple
from .base import BaseEvaluator

class SandboxCodeEvaluator(BaseEvaluator):
    """
    Executes generated Python code against real assertion unit tests in an isolated subprocess.
    Eliminates brittle substring checks.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def _extract_code(self, response: str) -> str:
        # Match ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the largest code block or combine
            return matches[-1].strip()
        # If no code fence found, return response as-is
        return response.strip()

    def evaluate(self, response: str, task: Dict[str, Any]) -> Tuple[bool, float, str]:
        metadata = task.get("metadata", {})
        test_code = metadata.get("test_code", "")
        code = self._extract_code(response)

        if not code:
            return False, 0.0, "FAIL: No code block extracted from model response."

        full_script = f"""
import sys, os

# Model Code Under Test:
{code}

# Unit Test Assertions:
if __name__ == '__main__':
{self._indent(test_code, 4)}
    print('ALL_TESTS_PASSED')
"""

        sandbox_dir = r"D:\fable_benchmark_sandbox\workspace"
        os.makedirs(sandbox_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8", dir=sandbox_dir) as tf:
            tf.write(full_script)
            tf_path = tf.name

        try:
            res = subprocess.run(
                [sys.executable, tf_path],
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            if "ALL_TESTS_PASSED" in res.stdout:
                return True, 1.0, "PASS: All functional unit test assertions succeeded."
            else:
                err_msg = res.stderr.strip().splitlines()[-1] if res.stderr else "Assertion failed without stderr"
                return False, 0.0, f"FAIL: Test assertions failed ({err_msg})"
        except subprocess.TimeoutExpired:
            return False, 0.0, f"FAIL: Execution timed out after {self.timeout_seconds}s (infinite loop or hang)."
        except Exception as e:
            return False, 0.0, f"FAIL: Sandbox runner exception: {str(e)}"
        finally:
            try:
                os.remove(tf_path)
            except Exception:
                pass

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(pad + line if line.strip() else "" for line in text.splitlines())
