"""
run_gpu_suite.py - Automated Sequential Head-to-Head GPU Benchmark Harness for OmniAgent-Bench.
Executes Fable-Coder V4, Qwen2.5 Base, and Fable-Coder V1 sequentially on RTX 3060 with zero VRAM leaks.
Generates publication-ready SVG bar charts and updates Fable-Coder-V4 deliverables.
"""

import os
import sys
import time
import json
import socket
import subprocess
import urllib.request
import urllib.error
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omni_bench.runner import BenchmarkRunner

SERVER_BIN = r"D:\llama_cuda_bin\llama-server.exe"
HOST = "127.0.0.1"

MODELS = [
    {
        "id": "fable-v4",
        "name": "Fable-Coder V4 (7.6B Replay-LoRA)",
        "path": r"D:\fable-coder-v4-7b.Q4_K_M.gguf",
        "port": 8084
    },
    {
        "id": "qwen2.5-base",
        "name": "Qwen2.5-Coder-7B-Instruct (Base)",
        "path": r"D:\qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "port": 8085
    },
    {
        "id": "fable-v1",
        "name": "Fable-Coder V1 (7B DPO)",
        "path": r"D:\fable-coder-7b-dpo.Q4_K_M.gguf",
        "port": 8086
    }
]

def wait_for_port_free(port: int, max_wait: int = 20) -> bool:
    for _ in range(max_wait):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((HOST, port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(1)
    return False

def wait_for_server_ready(server_proc: subprocess.Popen, port: int, max_retries: int = 150) -> bool:
    health_url = f"http://{HOST}:{port}/health"
    for i in range(max_retries):
        if server_proc.poll() is not None:
            print(f"    ERROR: llama-server exited prematurely with code {server_proc.returncode}!", flush=True)
            return False
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    print(f"    Server is ready on GPU! (took {i+1}s)", flush=True)
                    return True
        except Exception:
            if (i + 1) % 5 == 0:
                print(f"    Loading weights & context to RTX 3060... ({i+1}s)", flush=True)
            time.sleep(1)
    return False

def generate_svg_chart(summary_results: List[Dict[str, Any]], out_path: str):
    """Generates an SVG bar chart for OmniAgent-Bench results."""
    models = [m["model_name"] for m in summary_results]
    colors = ["#a855f7", "#38bdf8", "#fb7185"] # V4: Royal Purple, Qwen: Sky Blue, V1: Coral Rose

    categories = [
        ("Overall Pass Rate", "accuracy_pct"),
        ("Weighted Score", "weighted_score_pct"),
        ("Coding & Algorithms", "coding"),
        ("Agentic Tools", "agentic_tools"),
        ("Cybersecurity & Defense", "cybersecurity"),
        ("Long-Horizon Architecture", "long_horizon"),
        ("Instruction & Constraints", "instruction_following")
    ]

    data: Dict[str, List[float]] = {display_name: [] for display_name, _ in categories}

    for m in summary_results:
        cb = m.get("category_breakdown", {})
        for display_name, key in categories:
            if key == "accuracy_pct":
                data[display_name].append(round(m["accuracy_pct"], 1))
            elif key == "weighted_score_pct":
                data[display_name].append(round(m["weighted_score_pct"], 1))
            else:
                cat_info = cb.get(key, {})
                tot = cat_info.get("total", 0)
                pas = cat_info.get("passed", 0)
                pct = round((pas / tot) * 100, 1) if tot > 0 else 0.0
                data[display_name].append(pct)

    svg_w = 980
    svg_h = 580
    margin_l = 240
    margin_r = 60
    margin_t = 85
    margin_b = 65
    plot_w = svg_w - margin_l - margin_r
    plot_h = svg_h - margin_t - margin_b

    num_cats = len(categories)
    group_h = plot_h / num_cats
    bar_h = 16
    bar_gap = 4

    import html

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">',
        '<!-- Title & Subtitle -->',
        f'<text x="{svg_w // 2}" y="35" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="700">OmniAgent-Bench: Head-to-Head Performance Evaluation</text>',
        f'<text x="{svg_w // 2}" y="58" text-anchor="middle" fill="#94a3b8" font-size="13">NVIDIA RTX 3060 (12GB) • Functional Sandboxed Unit Tests • Schema-Aware Tool Execution</text>',
        '<!-- Grid lines -->',
    ]

    for pct in range(0, 101, 20):
        x = margin_l + (pct / 100) * plot_w
        svg_lines.append(f'<line x1="{x}" y1="{margin_t}" x2="{x}" y2="{margin_t + plot_h}" stroke="#334155" stroke-dasharray="3,3" stroke-width="1" />')
        svg_lines.append(f'<text x="{x}" y="{margin_t + plot_h + 20}" text-anchor="middle" fill="#64748b" font-size="11">{pct}%</text>')

    for i, (cat_display, _) in enumerate(categories):
        y_group = margin_t + i * group_h
        is_highlight = i < 2
        font_weight = "700" if is_highlight else "500"
        label_color = "#f8fafc" if is_highlight else "#cbd5e1"
        escaped_label = html.escape(cat_display)

        svg_lines.append(f'<text x="{margin_l - 15}" y="{y_group + group_h/2 + 3}" text-anchor="end" fill="{label_color}" font-size="12" font-weight="{font_weight}">{escaped_label}</text>')

        for m_idx, m_val in enumerate(data[cat_display]):
            bw = (m_val / 100) * plot_w
            by = y_group + (group_h - (3 * bar_h + 2 * bar_gap)) / 2 + m_idx * (bar_h + bar_gap)
            color = colors[m_idx]

            svg_lines.append(f'<rect x="{margin_l}" y="{by}" width="{max(bw, 3)}" height="{bar_h}" rx="3" fill="{color}" />')
            svg_lines.append(f'<text x="{margin_l + bw + 8}" y="{by + bar_h - 3}" fill="#f8fafc" font-size="10" font-weight="600">{m_val}%</text>')

    legend_y = svg_h - 22
    for idx, (m_name, col) in enumerate(zip(models, colors)):
        lx = margin_l + idx * 240
        escaped_model_name = html.escape(m_name)
        svg_lines.append(f'<rect x="{lx}" y="{legend_y - 10}" width="14" height="14" rx="2" fill="{col}" />')
        svg_lines.append(f'<text x="{lx + 20}" y="{legend_y + 1}" fill="#e2e8f0" font-size="11" font-weight="500">{escaped_model_name}</text>')

    svg_lines.append('</svg>')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"[SVG] Generated chart: {out_path}")

def run_suite(single_model: str = None):
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_file = os.path.join(bench_dir, "benchmarks", "tasks.json")
    reports_dir = os.path.join(bench_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    summary_file = os.path.join(reports_dir, "omni_benchmark_summary.json")

    target_models = MODELS
    if single_model:
        target_models = [m for m in MODELS if m["id"] == single_model]
        if not target_models:
            print(f"Unknown model id: {single_model}. Choose from {[m['id'] for m in MODELS]}")
            return

    runner = BenchmarkRunner(tasks_file=tasks_file)

    for model_cfg in target_models:
        port = model_cfg["port"]
        print(f"\n=======================================================", flush=True)
        print(f"  LAUNCHING: {model_cfg['name']} (Port: {port})", flush=True)
        print(f"  Model File: {model_cfg['path']}", flush=True)
        print(f"=======================================================\n", flush=True)

        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        wait_for_port_free(port)

        cmd = [
            SERVER_BIN,
            "-m", model_cfg["path"],
            "-ngl", "99",
            "-c", "4096",
            "-np", "1",
            "--port", str(port),
            "--host", HOST
        ]

        server_log_path = os.path.join(reports_dir, f"{model_cfg['id']}_server.log")
        server_log = open(server_log_path, "w", encoding="utf-8")
        server_proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)

        try:
            if not wait_for_server_ready(server_proc, port=port):
                raise RuntimeError(f"Server failed to start for {model_cfg['name']}.")

            # Run OmniAgent-Bench on the model
            runner.run_model(
                model_name=model_cfg["name"],
                backend="openai",
                base_url=f"http://{HOST}:{port}/v1",
                temperature=0.0,
                max_tokens=1024
            )

        finally:
            print(f"Terminating server for {model_cfg['name']} and freeing VRAM...", flush=True)
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(4)

    # Re-aggregate all reports
    all_summaries = []
    for m in MODELS:
        safe_name = m["name"].replace("/", "_").replace(":", "_").replace("\\", "_")
        rep_file = os.path.join(reports_dir, f"{safe_name}_report.json")
        if os.path.exists(rep_file):
            with open(rep_file, "r", encoding="utf-8") as rf:
                all_summaries.append(json.load(rf))

    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(all_summaries, sf, indent=2)

    # Generate charts and sync
    svg_path = os.path.join(reports_dir, "omniagent_benchmark_comparison.svg")
    generate_svg_chart(all_summaries, svg_path)

    # Sync to Fable-Coder-V4 repo
    v4_eval_dir = r"D:\Fable-Coder-V4\eval"
    os.makedirs(v4_eval_dir, exist_ok=True)
    with open(svg_path, "rb") as rf, open(os.path.join(v4_eval_dir, "omniagent_benchmark_comparison.svg"), "wb") as wf:
        wf.write(rf.read())
    with open(summary_file, "rb") as rf, open(os.path.join(v4_eval_dir, "omni_benchmark_summary.json"), "wb") as wf:
        wf.write(rf.read())

    # Sync to brain artifacts
    artifact_dir = r"C:\Users\mosaa\.gemini\antigravity\brain\be70db23-8782-46b9-8ba5-c44b60e86b10"
    if os.path.exists(artifact_dir):
        with open(svg_path, "rb") as rf, open(os.path.join(artifact_dir, "omniagent_benchmark_comparison.svg"), "wb") as wf:
            wf.write(rf.read())

    print(f"\n[OmniAgent-Bench] Multi-model suite complete! Results saved to {reports_dir} and synced to Fable-Coder-V4.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Run single model (e.g. fable-v4, qwen2.5-base, fable-v1)")
    args = parser.parse_args()
    run_suite(single_model=args.model)
