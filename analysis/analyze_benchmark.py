import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _extract_log_features(log_file: str) -> Dict[str, Any]:
    if not log_file or not isinstance(log_file, str) or not os.path.exists(log_file):
        return {
            "entropy_coverage": np.nan,
            "scr_shock": np.nan,
        }

    rows = _read_jsonl(log_file)
    if not rows:
        return {
            "entropy_coverage": np.nan,
            "scr_shock": np.nan,
        }

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "entropy_coverage": np.nan,
            "scr_shock": np.nan,
        }

    if "event_type" not in df.columns:
        df["event_type"] = "unknown"

    # Entropy coverage: fraction of non-probe events with entropy present.
    non_probe = ~df["event_type"].isin(["periodic_probe", "perturbation_triggered"])
    entropy_present = df.get("current_entropy").notna() if "current_entropy" in df.columns else pd.Series(False, index=df.index)
    denom = int(non_probe.sum())
    entropy_cov = float((entropy_present & non_probe).sum() / denom) if denom > 0 else np.nan

    # Shock SCR: peak SCR at perturbation steps (if any).
    scr_shock = np.nan
    if "scr" in df.columns:
        shock_rows = df[df["event_type"] == "perturbation_triggered"]
        if not shock_rows.empty:
            scr_shock = float(pd.to_numeric(shock_rows["scr"], errors="coerce").max())

    return {
        "entropy_coverage": entropy_cov,
        "scr_shock": scr_shock,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark CSV into paper-ready tables.")
    parser.add_argument("--results", required=True, help="CSV produced by experiments/run_benchmark.py")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <results>_summary.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if df.empty:
        raise SystemExit("No rows to analyze.")

    df = df.copy()
    df["validation_passed"] = df.get("validation_passed")

    # Attach log-derived features.
    features = df["log_file"].apply(lambda p: pd.Series(_extract_log_features(str(p))))
    df = pd.concat([df, features], axis=1)

    def _success_rate(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty:
            return np.nan
        return float((s == True).mean())  # noqa: E712

    group_keys = ["model_name", "scenario_id"]
    for optional in ["metric_embedding_backend", "metric_embedding_model"]:
        if optional in df.columns:
            group_keys.append(optional)

    grouped = df.groupby(group_keys, dropna=False)
    summary = grouped.agg(
        # Count rows, not non-null run_id values: failed runs may have missing run_id.
        runs=("scenario_id", "size"),
        success_rate=("validation_passed", _success_rate),
        median_steps=("steps_executed", "median"),
        median_total_tokens=("total_tokens", "median"),
        median_probe_tokens=("probe_total_tokens", "median"),
        median_total_tokens_incl_probes=("total_tokens_including_probes", "median"),
        mean_entropy_coverage=("entropy_coverage", "mean"),
        median_scr_shock=("scr_shock", "median"),
    ).reset_index()

    out_path = args.out or os.path.splitext(args.results)[0] + "_summary.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
