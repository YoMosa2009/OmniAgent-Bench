import os
import json
import time
from typing import Dict, Any, List, Optional
from .models import get_model_provider
from .evaluators import get_evaluator
from .reporter import BenchmarkReporter

class BenchmarkRunner:
    """Orchestrates benchmark task execution and sequential evaluation."""

    def __init__(self, tasks_file: str = "benchmarks/tasks.json", judge_backend: Optional[str] = None, judge_model: Optional[str] = None, **judge_kwargs):
        self.tasks_file = tasks_file
        self.tasks_data = self._load_tasks()
        self.judge_provider = None
        if judge_backend and judge_model:
            print(f"[Runner] Initializing LLM Judge: {judge_model} ({judge_backend})...", flush=True)
            self.judge_provider = get_model_provider(judge_backend, judge_model, **judge_kwargs)
        self.reporter = BenchmarkReporter()

    def _load_tasks(self) -> Dict[str, Any]:
        if not os.path.exists(self.tasks_file):
            raise FileNotFoundError(f"Benchmark tasks file not found: {self.tasks_file}")
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_model(self, model_name: str, backend: str, temperature: float = 0.0, max_tokens: int = 1024, **kwargs) -> Dict[str, Any]:
        """Runs the entire benchmark on a single model with guaranteed memory cleanup."""
        print(f"\n=======================================================", flush=True)
        print(f"  STARTING BENCHMARK: {model_name} [Backend: {backend}]", flush=True)
        print(f"=======================================================\n", flush=True)

        provider = get_model_provider(backend, model_name, **kwargs)
        tasks = self.tasks_data.get("tasks", [])
        
        task_results = []
        passed_count = 0
        total_points = 0
        earned_points = 0
        cat_stats = {}

        start_time = time.time()

        for idx, task in enumerate(tasks, 1):
            t_id = task["id"]
            cat = task.get("category", "general")
            title = task.get("title", "")
            prompt = task.get("prompt", "")
            eval_type = task.get("eval_type", "unit_test")
            pts = task.get("points", 1)
            total_points += pts

            if cat not in cat_stats:
                cat_stats[cat] = {"passed": 0, "total": 0, "pts_earned": 0, "pts_total": 0}
            cat_stats[cat]["total"] += 1
            cat_stats[cat]["pts_total"] += pts

            print(f"[{idx:2d}/{len(tasks):2d}] Testing {t_id:8s} ({cat:16s}) - {title[:32]}...", end="", flush=True)

            # Generate response
            resp = provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)

            # Evaluate response
            evaluator = get_evaluator(eval_type, judge_provider=self.judge_provider)
            passed, score_ratio, feedback = evaluator.evaluate(resp, task)

            task_pts = pts * score_ratio
            earned_points += task_pts

            if passed:
                passed_count += 1
                cat_stats[cat]["passed"] += 1
            cat_stats[cat]["pts_earned"] += task_pts

            status_str = "PASS" if passed else "FAIL"
            print(f" [{status_str}] ({task_pts:.1f}/{pts} pts)", flush=True)

            task_results.append({
                "id": t_id,
                "category": cat,
                "title": title,
                "eval_type": eval_type,
                "passed": passed,
                "score_ratio": score_ratio,
                "points_earned": task_pts,
                "points_possible": pts,
                "feedback": feedback,
                "response_preview": resp[:300]
            })

        duration = time.time() - start_time
        accuracy = (passed_count / len(tasks)) * 100 if tasks else 0
        weighted = (earned_points / total_points) * 100 if total_points > 0 else 0

        # Essential for multi-model sequential execution: flush memory
        provider.cleanup()

        result = {
            "model_name": model_name,
            "backend": backend,
            "evaluation_time_seconds": round(duration, 2),
            "tasks_passed": passed_count,
            "total_tasks": len(tasks),
            "accuracy_pct": round(accuracy, 2),
            "earned_points": round(earned_points, 2),
            "total_points": total_points,
            "weighted_score_pct": round(weighted, 2),
            "category_breakdown": cat_stats,
            "task_results": task_results
        }

        self.reporter.print_terminal_summary(result)
        self.reporter.save_json_report(result)
        self.reporter.update_leaderboard(result)

        return result

    def run_multiple_models(self, model_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluates multiple models sequentially one at a time to prevent GPU VRAM overload."""
        print(f"\n[OmniAgent-Bench] Queueing {len(model_specs)} models for sequential evaluation...")
        all_results = []
        for idx, spec in enumerate(model_specs, 1):
            print(f"\n>>> Running Model {idx}/{len(model_specs)}: {spec['model_name']} <<<")
            res = self.run_model(
                model_name=spec["model_name"],
                backend=spec.get("backend", "ollama"),
                temperature=spec.get("temperature", 0.0),
                max_tokens=spec.get("max_tokens", 1024),
                **spec.get("extra_kwargs", {})
            )
            all_results.append(res)
            # Brief pause between models
            time.sleep(2)
        return all_results
