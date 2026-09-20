import argparse
import sys
import os
import json

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omni_bench.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(
        description="OmniAgent-Bench: The Modern Evaluation Benchmark for Agentic, Coding, & CyberSec LLMs"
    )
    parser.add_argument("--model", type=str, help="Model name or path to evaluate (e.g. 'fable-coder-v4', 'qwen2.5-coder:7b', 'gpt-4o')")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openai", "vllm", "anthropic", "huggingface"], help="Model execution backend")
    parser.add_argument("--base-url", type=str, help="Custom API Base URL (for vLLM, LocalAI, OpenCode)")
    parser.add_argument("--api-key", type=str, help="API key for cloud or protected backends")
    parser.add_argument("--tasks", type=str, default="benchmarks/tasks.json", help="Path to benchmark tasks JSON file")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate per task")
    
    # LLM Judge Configuration
    parser.add_argument("--judge-backend", type=str, help="Backend for LLM-as-a-Judge (e.g. 'anthropic', 'openai')")
    parser.add_argument("--judge-model", type=str, help="Model name for LLM-as-a-Judge (e.g. 'claude-3-5-sonnet-20241022', 'gpt-4o')")
    parser.add_argument("--judge-api-key", type=str, help="API key for judge model")

    # Batch Sequential Execution
    parser.add_argument("--models-file", type=str, help="JSON file containing a list of model specs to evaluate sequentially")

    args = parser.parse_args()

    judge_kwargs = {}
    if args.judge_api_key:
        judge_kwargs["api_key"] = args.judge_api_key

    runner = BenchmarkRunner(
        tasks_file=args.tasks,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        **judge_kwargs
    )

    if args.models_file:
        if not os.path.exists(args.models_file):
            print(f"Error: models file '{args.models_file}' not found.")
            sys.exit(1)
        with open(args.models_file, "r", encoding="utf-8") as f:
            specs = json.load(f)
        runner.run_multiple_models(specs)
    elif args.model:
        kwargs = {}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        if args.api_key:
            kwargs["api_key"] = args.api_key
        runner.run_model(
            model_name=args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **kwargs
        )
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
