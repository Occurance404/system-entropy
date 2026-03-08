from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _iso_to_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_manifest(session_dir: str, run_id: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, "run_artifacts", run_id, "manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_run_rows(session_dir: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(session_dir, "logs", "rescue", "sim_baseline_*.jsonl")
    files = sorted(glob.glob(pattern))
    run_rows: List[Dict[str, Any]] = []

    for log_path in files:
        rows = _read_jsonl(log_path)
        if not rows:
            continue
        first = rows[0]
        last = rows[-1]
        first_ts = _iso_to_dt(str(first.get("timestamp", "")))
        last_ts = _iso_to_dt(str(last.get("timestamp", "")))
        duration_s = None
        if first_ts and last_ts:
            duration_s = (last_ts - first_ts).total_seconds()

        event_counts = Counter(str(r.get("event_type", "unknown")) for r in rows)
        tool_counts = Counter()
        for r in rows:
            m = r.get("metrics", {})
            tool = m.get("tool")
            if tool:
                tool_counts[str(tool)] += 1

        m_last = last.get("metrics", {})
        run_id = str(m_last.get("run_id") or "")
        manifest = _load_manifest(session_dir, run_id) if run_id else {}
        env = manifest.get("env", {}) if isinstance(manifest, dict) else {}

        scenario_id = (
            manifest.get("scenario_id")
            or m_last.get("scenario_id")
            or str(os.path.basename(log_path))
        )
        seed = env.get("SCENARIO_SEED")
        validation_passed = bool(m_last.get("validation_passed"))
        task_complete = bool(m_last.get("task_complete"))
        steps = len(rows)
        prompt_tokens = sum((r.get("metrics", {}).get("prompt_tokens") or 0) for r in rows)
        completion_tokens = sum((r.get("metrics", {}).get("completion_tokens") or 0) for r in rows)
        total_tokens = sum((r.get("metrics", {}).get("total_tokens") or 0) for r in rows)

        run_rows.append(
            {
                "log_file": os.path.relpath(log_path, session_dir),
                "run_id": run_id,
                "scenario_id": str(scenario_id),
                "seed": str(seed) if seed is not None else "",
                "steps": steps,
                "duration_s": f"{duration_s:.3f}" if duration_s is not None else "",
                "task_complete": task_complete,
                "validation_passed": validation_passed,
                "validation_details": str(m_last.get("validation_details") or ""),
                "last_event_type": str(last.get("event_type") or ""),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "event_tool_execution": event_counts.get("tool_execution", 0),
                "event_llm_reply": event_counts.get("llm_reply", 0),
                "event_perturbation_triggered": event_counts.get("perturbation_triggered", 0),
                "tool_read_file": tool_counts.get("read_file", 0),
                "tool_write_file": tool_counts.get("write_file", 0),
                "tool_execute_python": tool_counts.get("execute_python", 0),
                "tool_run_shell": tool_counts.get("run_shell", 0),
            }
        )

    return run_rows


def write_run_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize_by_scenario(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_s: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_s[r["scenario_id"]].append(r)

    out: List[Dict[str, Any]] = []
    for scenario_id in sorted(by_s):
        vals = by_s[scenario_id]
        n = len(vals)
        passes = sum(1 for r in vals if r["validation_passed"])
        steps = [int(r["steps"]) for r in vals]
        toks = [int(r["total_tokens"]) for r in vals]
        dur = [float(r["duration_s"]) for r in vals if str(r["duration_s"]).strip()]
        out.append(
            {
                "scenario_id": scenario_id,
                "runs": n,
                "passes": passes,
                "pass_rate": round(passes / n, 4) if n else 0.0,
                "avg_steps": round(sum(steps) / n, 2) if n else 0.0,
                "median_steps": median(steps) if steps else 0.0,
                "avg_total_tokens": round(sum(toks) / n, 1) if n else 0.0,
                "median_total_tokens": median(toks) if toks else 0.0,
                "avg_duration_s": round(sum(dur) / len(dur), 3) if dur else 0.0,
            }
        )
    return out


def write_scenario_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _scenario_order(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted({r["scenario_id"] for r in rows})


def plot_pass_rate(rows: List[Dict[str, Any]], out_path: str) -> None:
    scenarios = [r["scenario_id"] for r in rows]
    rates = [100.0 * float(r["pass_rate"]) for r in rows]
    plt.figure(figsize=(12, 5))
    colors = ["#2E8B57" if rate >= 80 else "#B22222" for rate in rates]
    plt.bar(scenarios, rates, color=colors)
    plt.ylim(0, 105)
    plt.ylabel("Pass Rate (%)")
    plt.title("Validation Pass Rate by Scenario")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(rates):
        plt.text(i, min(103, v + 2), f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_steps_box(run_rows: List[Dict[str, Any]], out_path: str) -> None:
    by_s: Dict[str, List[int]] = defaultdict(list)
    for r in run_rows:
        by_s[r["scenario_id"]].append(int(r["steps"]))
    scenarios = sorted(by_s)
    data = [by_s[s] for s in scenarios]
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=scenarios, showmeans=True)
    plt.ylabel("Steps to Stop")
    plt.title("Step Count Distribution by Scenario")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tokens_box(run_rows: List[Dict[str, Any]], out_path: str) -> None:
    by_s: Dict[str, List[float]] = defaultdict(list)
    for r in run_rows:
        by_s[r["scenario_id"]].append(int(r["total_tokens"]) / 1000.0)
    scenarios = sorted(by_s)
    data = [by_s[s] for s in scenarios]
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=scenarios, showmeans=True)
    plt.ylabel("Total Tokens (k)")
    plt.title("Token Cost Distribution by Scenario")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_seed_heatmap(run_rows: List[Dict[str, Any]], out_path: str) -> None:
    scenarios = _scenario_order(run_rows)
    seeds = sorted({r["seed"] for r in run_rows if r["seed"] != ""}, key=lambda x: int(x))
    if not seeds:
        return
    matrix = np.full((len(scenarios), len(seeds)), np.nan)
    s_idx = {s: i for i, s in enumerate(scenarios)}
    z_idx = {z: i for i, z in enumerate(seeds)}
    for r in run_rows:
        seed = r["seed"]
        if seed == "":
            continue
        i = s_idx[r["scenario_id"]]
        j = z_idx[seed]
        matrix[i, j] = 1.0 if r["validation_passed"] else 0.0

    plt.figure(figsize=(8, 6))
    cmap = plt.matplotlib.colors.ListedColormap(["#b22222", "#2e8b57"])
    plt.imshow(np.nan_to_num(matrix, nan=0.0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="Validation Pass (0=fail, 1=pass)")
    plt.yticks(range(len(scenarios)), scenarios)
    plt.xticks(range(len(seeds)), seeds)
    plt.xlabel("Seed")
    plt.ylabel("Scenario")
    plt.title("Pass/Fail by Scenario and Seed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_markdown_summary(
    session_dir: str,
    run_rows: List[Dict[str, Any]],
    scenario_rows: List[Dict[str, Any]],
    out_path: str,
) -> None:
    total = len(run_rows)
    passed = sum(1 for r in run_rows if r["validation_passed"])
    failed = total - passed
    total_prompt = sum(int(r["prompt_tokens"]) for r in run_rows)
    total_completion = sum(int(r["completion_tokens"]) for r in run_rows)
    total_tokens = sum(int(r["total_tokens"]) for r in run_rows)
    avg_steps = sum(int(r["steps"]) for r in run_rows) / total if total else 0.0

    failed_rows = [r for r in run_rows if not r["validation_passed"]]
    failed_rows.sort(key=lambda x: (x["scenario_id"], x["seed"]))

    lines: List[str] = []
    lines.append("# Session Summary")
    lines.append("")
    lines.append(f"- Session: `{os.path.basename(session_dir)}`")
    lines.append(f"- Runs: `{total}`")
    lines.append(f"- Pass: `{passed}`")
    lines.append(f"- Fail: `{failed}`")
    lines.append(f"- Pass Rate: `{(100.0 * passed / total):.1f}%`" if total else "- Pass Rate: `0.0%`")
    lines.append(f"- Avg Steps: `{avg_steps:.2f}`")
    lines.append(f"- Total Prompt Tokens: `{total_prompt}`")
    lines.append(f"- Total Completion Tokens: `{total_completion}`")
    lines.append(f"- Total Tokens: `{total_tokens}`")
    lines.append("")
    lines.append("## Per Scenario")
    lines.append("")
    lines.append("| scenario_id | pass/runs | pass_rate | avg_steps | avg_total_tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in scenario_rows:
        lines.append(
            f"| {r['scenario_id']} | {r['passes']}/{r['runs']} | {100*float(r['pass_rate']):.1f}% | {r['avg_steps']} | {r['avg_total_tokens']} |"
        )
    lines.append("")
    lines.append("## Failed Runs")
    lines.append("")
    if not failed_rows:
        lines.append("- None")
    else:
        for r in failed_rows:
            detail = r["validation_details"].replace("\n", " ").strip()
            if len(detail) > 220:
                detail = detail[:220] + "..."
            lines.append(
                f"- `{r['scenario_id']}` seed `{r['seed']}` steps `{r['steps']}`: {detail}"
            )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- `plots/pass_rate_by_scenario.png`")
    lines.append("- `plots/steps_boxplot_by_scenario.png`")
    lines.append("- `plots/tokens_boxplot_by_scenario.png`")
    lines.append("- `plots/pass_fail_heatmap.png`")
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def resolve_session_dir(arg_session_dir: str) -> str:
    p = os.path.abspath(arg_session_dir)
    if os.path.isdir(p):
        return p
    raise FileNotFoundError(f"Session directory not found: {arg_session_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a managed experiment session and generate paper-friendly plots.")
    parser.add_argument(
        "--session-dir",
        default="data/experiments/latest",
        help="Path to session directory (default: data/experiments/latest).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <session-dir>/results/session_summary).",
    )
    args = parser.parse_args()

    session_dir = resolve_session_dir(args.session_dir)
    out_dir = args.out_dir or os.path.join(session_dir, "results", "session_summary")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    run_rows = collect_run_rows(session_dir)
    if not run_rows:
        print(f"No rescue JSONL logs found under: {session_dir}")
        return 1

    run_csv = os.path.join(out_dir, "run_summary.csv")
    write_run_csv(run_csv, run_rows)

    scenario_rows = summarize_by_scenario(run_rows)
    scenario_csv = os.path.join(out_dir, "scenario_summary.csv")
    write_scenario_csv(scenario_csv, scenario_rows)

    plot_pass_rate(scenario_rows, os.path.join(plots_dir, "pass_rate_by_scenario.png"))
    plot_steps_box(run_rows, os.path.join(plots_dir, "steps_boxplot_by_scenario.png"))
    plot_tokens_box(run_rows, os.path.join(plots_dir, "tokens_boxplot_by_scenario.png"))
    plot_seed_heatmap(run_rows, os.path.join(plots_dir, "pass_fail_heatmap.png"))

    report_md = os.path.join(out_dir, "session_summary.md")
    write_markdown_summary(session_dir, run_rows, scenario_rows, report_md)

    print(f"Session: {session_dir}")
    print(f"Runs: {len(run_rows)}")
    print(f"Run CSV: {run_csv}")
    print(f"Scenario CSV: {scenario_csv}")
    print(f"Report: {report_md}")
    print(f"Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
