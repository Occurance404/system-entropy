from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Any

import pandas as pd


def _safe_copy(src: str, dst: str) -> bool:
    if not src or not isinstance(src, str) or not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a benchmark CSV + artifacts into a shareable folder.")
    parser.add_argument("--results", required=True, help="CSV produced by experiments/run_benchmark.py")
    parser.add_argument("--out-dir", default=None, help="Output folder (default: data/datasets/<name>_<ts>/)")
    parser.add_argument("--copy-logs", action="store_true", help="Copy JSONL log files referenced by the CSV.")
    parser.add_argument("--copy-manifests", action="store_true", help="Copy data/run_artifacts/<run_id>/manifest.json files.")
    args = parser.parse_args()

    results_path = args.results
    df = pd.read_csv(results_path)
    if df.empty:
        raise SystemExit("No rows found in results CSV.")

    base_name = os.path.splitext(os.path.basename(results_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("data", "datasets", f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    packaged_results = os.path.join(out_dir, "benchmark.csv")
    _safe_copy(results_path, packaged_results)

    summary_path = os.path.splitext(results_path)[0] + "_summary.csv"
    if os.path.exists(summary_path):
        _safe_copy(summary_path, os.path.join(out_dir, "benchmark_summary.csv"))

    deltas_path = os.path.splitext(results_path)[0] + "_shock_deltas.csv"
    if os.path.exists(deltas_path):
        _safe_copy(deltas_path, os.path.join(out_dir, "benchmark_shock_deltas.csv"))

    index: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "source_results": os.path.abspath(results_path),
        "rows": int(len(df)),
        "copied_logs": 0,
        "copied_manifests": 0,
        "runs": [],
    }

    if args.copy_logs:
        logs_dir = os.path.join(out_dir, "logs")
        for _, row in df.iterrows():
            run_id = row.get("run_id")
            log_file = row.get("log_file")
            if not isinstance(run_id, str) or not run_id:
                continue
            if not isinstance(log_file, str) or not log_file:
                continue
            dst = os.path.join(logs_dir, f"{run_id}.jsonl")
            if _safe_copy(log_file, dst):
                index["copied_logs"] += 1

    if args.copy_manifests:
        manifests_dir = os.path.join(out_dir, "manifests")
        for _, row in df.iterrows():
            run_id = row.get("run_id")
            run_dir = row.get("run_dir")
            if not isinstance(run_id, str) or not run_id:
                continue
            manifest_src = None
            if isinstance(run_dir, str) and run_dir:
                candidate = os.path.join(run_dir, "manifest.json")
                if os.path.exists(candidate):
                    manifest_src = candidate
            if manifest_src is None:
                continue
            dst = os.path.join(manifests_dir, f"{run_id}.manifest.json")
            if _safe_copy(manifest_src, dst):
                index["copied_manifests"] += 1

    # Minimal run index for quick browsing.
    for _, row in df.iterrows():
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        manifest_path = None
        run_dir = row.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            manifest_path = os.path.join(run_dir, "manifest.json")
        index["runs"].append(
            {
                "run_id": run_id,
                "model_name": row.get("model_name"),
                "model": row.get("model"),
                "scenario_id": row.get("scenario_id"),
                "rep_index": row.get("rep_index"),
                "validation_passed": row.get("validation_passed"),
                "log_file": row.get("log_file") if not args.copy_logs else f"logs/{run_id}.jsonl",
                "manifest": (manifest_path if manifest_path else None)
                if not args.copy_manifests
                else f"manifests/{run_id}.manifest.json",
            }
        )

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    readme_path = os.path.join(out_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "# Benchmark Dataset\n\n"
                f"- Created at: `{index['created_at']}`\n"
                f"- Source CSV: `{index['source_results']}`\n"
                f"- Rows: `{index['rows']}`\n"
                f"- Logs copied: `{index['copied_logs']}`\n"
                f"- Manifests copied: `{index['copied_manifests']}`\n\n"
                "## Files\n"
                "- `benchmark.csv`: raw per-run results\n"
                "- `benchmark_summary.csv`: aggregated summary (if present)\n"
                "- `benchmark_shock_deltas.csv`: baseline→shock deltas (if present)\n"
                "- `index.json`: quick index of runs\n"
                "- `logs/`: per-run JSONL logs (if copied)\n"
                "- `manifests/`: per-run manifests (if copied)\n"
            )

    print(f"Packaged dataset written to: {out_dir}")


if __name__ == "__main__":
    main()
