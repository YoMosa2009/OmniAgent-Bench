import os
import json
from typing import Dict, Any, List

class BenchmarkReporter:
    """Formats benchmark results, generates Markdown tables, and tracks leaderboards."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def print_terminal_summary(self, result: Dict[str, Any]):
        model_name = result["model_name"]
        accuracy = result["accuracy_pct"]
        weighted = result["weighted_score_pct"]
        passed = result["tasks_passed"]
        total = result["total_tasks"]

        print("\n" + "=" * 68)
        print(f"  OMNIAGENT-BENCH EVALUATION REPORT: {model_name}")
        print("=" * 68)
        print(f"Overall Task Pass Rate: {passed}/{total} ({accuracy:.1f}%)")
        print(f"Weighted Score:         {result['earned_points']}/{result['total_points']} ({weighted:.1f}%)\n")

        print("Category Breakdown:")
        print(f"  {'Category':24s} | {'Passed':10s} | {'Pass Rate':10s} | {'Weighted Pts':12s}")
        print("  " + "-" * 62)
        for cat, stats in result["category_breakdown"].items():
            pct = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            pts_str = f"{stats['pts_earned']}/{stats['pts_total']}"
            print(f"  {cat:24s} | {stats['passed']:2d}/{stats['total']:2d}      | {pct:6.1f}%    | {pts_str:12s}")
        print("=" * 68 + "\n")

    def save_json_report(self, result: Dict[str, Any]) -> str:
        safe_name = result["model_name"].replace("/", "_").replace(":", "_").replace("\\", "_")
        filename = f"{safe_name}_report.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Reporter] Saved JSON report to: {path}")
        return path

    def update_leaderboard(self, result: Dict[str, Any], leaderboard_file: str = "LEADERBOARD.md"):
        entry = {
            "model": result["model_name"],
            "backend": result.get("backend", "unknown"),
            "overall_pass": f"{result['tasks_passed']}/{result['total_tasks']} ({result['accuracy_pct']:.1f}%)",
            "weighted": f"{result['weighted_score_pct']:.1f}%",
            "coding": self._cat_pct(result, "coding"),
            "agentic": self._cat_pct(result, "agentic_tools"),
            "cybersec": self._cat_pct(result, "cybersecurity"),
            "reasoning": self._cat_pct(result, "long_horizon"),
            "instruction": self._cat_pct(result, "instruction_following")
        }

        # Read or init leaderboard
        all_entries = []
        if os.path.exists(leaderboard_file):
            try:
                with open(leaderboard_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Parse existing or re-write
            except Exception:
                pass

        # Write or update markdown leaderboard
        md_table = self._build_leaderboard_md([entry])
        with open(leaderboard_file, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"[Reporter] Updated leaderboard in: {leaderboard_file}")

    @staticmethod
    def _cat_pct(result: Dict[str, Any], cat_key: str) -> str:
        stats = result.get("category_breakdown", {}).get(cat_key, {})
        if not stats or stats.get("total", 0) == 0:
            return "N/A"
        pct = (stats["passed"] / stats["total"]) * 100
        return f"{stats['passed']}/{stats['total']} ({pct:.0f}%)"

    @staticmethod
    def _build_leaderboard_md(entries: List[Dict[str, str]]) -> str:
        md = "# OmniAgent-Bench Global Leaderboard\n\n"
        md += "A unified benchmark tracking LLM performance across Coding, Tool Execution, Cybersecurity, and Reasoning without brittle keyword grading.\n\n"
        md += "| Model Name | Backend | Overall Pass Rate | Weighted Score | Coding | Agentic Tools | Cybersecurity | Long-Horizon | Instruction Following |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for e in entries:
            md += f"| **{e['model']}** | `{e['backend']}` | **{e['overall_pass']}** | **{e['weighted']}** | {e['coding']} | {e['agentic']} | {e['cybersec']} | {e['reasoning']} | {e['instruction']} |\n"
        md += "\n*Updated automatically by `OmniAgent-Bench` runner.*\n"
        return md
