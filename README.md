# OmniAgent-Bench: The Modern Evaluation Benchmark for Agentic, Coding, & CyberSec LLMs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Evaluation: Functional Sandboxed](https://img.shields.io/badge/Evaluation-Functional%20Sandboxed-green.svg)](#)
[![LLM Judge: Supported](https://img.shields.io/badge/LLM--as--a--Judge-Claude%20%7C%20GPT--4o%20%7C%20Local-purple.svg)](#)

**OmniAgent-Bench** is a universal, open-source benchmark suite designed to evaluate any Large Language Model (open weights, quantized GGUF, or cloud APIs) across **Coding**, **Agentic Tool Execution**, **Cybersecurity Defense**, and **Long-Horizon Reasoning**.

Unlike traditional static benchmarks that rely on fragile substring matching or keyword traps, OmniAgent-Bench uses **isolated functional unit testing**, **strict JSON schema validation**, and **optional Model-as-a-Judge evaluation** to measure real-world intelligence without false negatives.

---

## 1. Why Traditional Benchmarks Fail vs. The OmniAgent-Bench Solution

Most open-source benchmarks evaluate models using static regex or `if "keyword" in response:` checks. This leads to **catastrophic false negatives**:
* If a model writes valid ANSI SQL with a subquery instead of a specific window function like `DENSE_RANK()`, a regex grader marks it as 0 points.
* If a model thoroughly explains a prompt injection attack by quoting the attacker's payload (e.g. `attacker.example`), a "forbidden string" filter fails the model for being too thorough.
* If a model diagnoses buggy code by stating *"The function sums odd numbers instead of even numbers"*, a filter forbidding the words *"odd numbers"* marks the diagnosis as wrong.

### The OmniAgent-Bench Architecture
OmniAgent-Bench replaces brittle substring checks with three functional evaluation tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          OmniAgent-Bench                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Functional Code  │     │   Agentic Tool   │     │  Model-as-Judge  │
│ Sandbox Runner   │     │ Schema Validator │     │ & Semantic Logic │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ Executes code in │     │ Validates tool   │     │ Uses Claude/GPT  │
│ subprocess with  │     │ calls, arguments │     │ or heuristic     │
│ real assertions  │     │ & ChatML schemas │     │ concept checks   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **Sandboxed Code Execution**: Extracts code from markdown fences and runs it inside an isolated Python subprocess against multi-case unit assertions (`assert func(input) == expected`). The code is graded by whether it **actually works**, not whether it used specific variable names.
2. **Schema-Aware Tool Calling**: Verifies multi-turn tool invocations against JSON Schema definitions (e.g. function name, argument presence, type safety, and path sanitization).
3. **Model-as-a-Judge**: Evaluates nuanced explanations, architecture trade-offs, and security audits using an LLM judge (Claude 3.5 Sonnet, GPT-4o, Codex, or a local judge via Ollama) with a structured rubric.

---

## 2. Benchmark Evaluation Pillars

OmniAgent-Bench evaluates models across five core domains:

| Pillar | Focus Area | Evaluation Methodology |
| :--- | :--- | :--- |
| **1. Coding & Algorithms** | Algorithmic logic, boundary conditions, non-mutating data structures, path traversal checks. | Functional in-memory Python subprocess execution with test assertion suites. |
| **2. Agentic Tool Execution** | Single & multi-turn tool calling, ChatML `<tool_call>` schema adherence, argument formatting. | Schema validator parsing JSON arguments and parameter types. |
| **3. Cybersecurity & Defense** | Direct prompt injection resistance, SQLi parameterization, destructive command interception. | LLM Judge or multi-concept security rubric verifying that attacks are mitigated. |
| **4. Long-Horizon Architecture** | State machines, race condition diagnostics, multi-step verification plans. | Step-by-step logic and dependency validation. |
| **5. Strict Constraints** | JSON-only formatting, word count limits, negative constraints. | Format and constraint validator. |

---

## 3. Sequential Multi-Model Evaluation (Zero GPU OOM)

Benchmarking multiple LLMs on a single GPU often crashes the machine due to unreleased VRAM. 

OmniAgent-Bench features **sequential model isolation**:
* Models in a test queue are evaluated **strictly one at a time**.
* Between evaluations, the runner invokes explicit garbage collection and CUDA cache flushing (`torch.cuda.empty_cache()` and Ollama `keep_alive: 0`).
* Any consumer GPU (or cloud instance like an A100 / T4) can benchmark a battery of models without choking.

---

## 4. Quickstart

### Installation
```bash
git clone https://github.com/YoMosa2009/OmniAgent-Bench.git
cd OmniAgent-Bench
pip install -r requirements.txt
```

### Benchmark a Model via Ollama
```bash
# Test local Ollama model (e.g. Fable-Coder V4 or Qwen 2.5 Coder)
python run_bench.py --model fable-coder-v4 --backend ollama
```

### Benchmark via OpenAI / vLLM / OpenCode Endpoint
```bash
python run_bench.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --backend openai \
    --base-url http://localhost:8000/v1
```

### Benchmark with LLM-as-a-Judge (Claude 3.5 Sonnet)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python run_bench.py --model fable-coder-v4 --backend ollama \
    --judge-backend anthropic \
    --judge-model claude-3-5-sonnet-20241022
```

### Benchmark Multiple Models Sequentially
Define your model list in `models.json`:
```json
[
  {"model_name": "fable-coder-v4", "backend": "ollama"},
  {"model_name": "qwen2.5-coder:7b", "backend": "ollama"},
  {"model_name": "deepseek-coder-v2:16b", "backend": "ollama"}
]
```
Run the batch queue:
```bash
python run_bench.py --models-file models.json
```

---

## 5. Output & Reports

OmniAgent-Bench automatically generates:
1. **Terminal Scorecard**: Real-time pass/fail breakdown by category and weighted points.
2. **JSON Evaluation Report**: Full trace of every prompt, model output, and evaluator feedback in `reports/<model>_report.json`.
3. **Markdown Leaderboard**: Live-updating `LEADERBOARD.md` comparing all tested models side-by-side.

Example Output:
```text
====================================================================
  OMNIAGENT-BENCH EVALUATION REPORT: fable-coder-v4
====================================================================
Overall Task Pass Rate: 13/15 (86.7%)
Weighted Score:         62.0/67 (92.5%)

Category Breakdown:
  Category                 | Passed     | Pass Rate  | Weighted Pts
  ------------------------------------------------------------------
  coding                   |  6/ 6      |  100.0%    | 27.0/27.0   
  agentic_tools            |  3/ 3      |  100.0%    | 14.0/14.0   
  cybersecurity            |  2/ 3      |   66.7%    |  9.0/14.0   
  long_horizon             |  2/ 2      |  100.0%    | 10.0/10.0   
  instruction_following    |  0/ 1      |    0.0%    |  2.0/ 5.0   
====================================================================
```

---

## 6. Adding Custom Tasks

To add custom tasks, simply add an entry to `benchmarks/tasks.json` or pass a custom file with `--tasks my_tasks.json`:

```json
{
  "id": "CUSTOM-01",
  "category": "coding",
  "title": "Reverse Words in String",
  "prompt": "Write a Python function `reverse_words(s: str) -> str`...",
  "eval_type": "unit_test",
  "points": 5,
  "metadata": {
    "test_code": "assert reverse_words('hello world') == 'world hello'\nassert reverse_words('a b c') == 'c b a'"
  }
}
```

---

## 7. License

OmniAgent-Bench is released under the [Apache 2.0 License](LICENSE).
