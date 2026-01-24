from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd


def _scenario_family(scenario_id: str) -> tuple[str, str] | None:
    if not isinstance(scenario_id, str):
        return None
    scenario_id = scenario_id.strip()
    if scenario_id.endswith("_baseline"):
        return scenario_id[: -len("_baseline")], "baseline"
    if scenario_id.endswith("_shock"):
        return scenario_id[: -len("_shock")], "shock"
    return None


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    lo = float(max(0.0, center - half))
    hi = float(min(1.0, center + half))
    return lo, hi


def _load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("No rows found in results CSV.")
    if "scenario_id" not in df.columns or "model_name" not in df.columns:
        raise SystemExit("Missing required columns: scenario_id, model_name")
    if "validation_passed" not in df.columns:
        raise SystemExit("Missing required column: validation_passed")
    df = df.copy()
    df["pair"] = df["scenario_id"].apply(_scenario_family)
    df = df[df["pair"].notna()]
    df["family"] = df["pair"].apply(lambda x: x[0])
    df["condition"] = df["pair"].apply(lambda x: x[1])
    df["validation_passed"] = pd.to_numeric(df["validation_passed"], errors="coerce")
    df = df[df["validation_passed"].notna()]
    df["validation_passed"] = df["validation_passed"].astype(int)
    return df


def _summarize_success(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "family", "condition"]
    if "metric_embedding_backend" in df.columns:
        group_cols.append("metric_embedding_backend")
    grouped = df.groupby(group_cols, dropna=False)["validation_passed"]
    rows: list[dict[str, Any]] = []
    for keys, s in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {k: v for k, v in zip(group_cols, keys)}
        n = int(s.shape[0])
        succ = int(s.sum())
        rate = float(succ / n) if n else float("nan")
        lo, hi = _wilson_ci(succ, n)
        rows.append({**key_map, "n": n, "successes": succ, "success_rate": rate, "ci95_lo": lo, "ci95_hi": hi})
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No baseline/shock rows to summarize.")
    return out


def _plot_success_rates(summary: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    summary = summary.copy()
    summary["label"] = summary["family"].astype(str)

    models = list(summary["model_name"].dropna().unique())
    families = list(summary["family"].dropna().unique())
    models.sort()
    families.sort()

    nrows = max(1, len(models))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(max(10, 1.4 * len(families) + 4), 3.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = summary[summary["model_name"] == model]
        x = np.arange(len(families))
        width = 0.35

        def _vals(cond: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            s = sub[sub["condition"] == cond].set_index("family")
            y = np.array([float(s.loc[f, "success_rate"]) if f in s.index else np.nan for f in families], dtype=float)
            lo = np.array([float(s.loc[f, "ci95_lo"]) if f in s.index else np.nan for f in families], dtype=float)
            hi = np.array([float(s.loc[f, "ci95_hi"]) if f in s.index else np.nan for f in families], dtype=float)
            return y, lo, hi

        yb, lob, hib = _vals("baseline")
        ys, los, his = _vals("shock")

        ax.bar(x - width / 2, yb, width, label="baseline", color="#4C78A8")
        ax.bar(x + width / 2, ys, width, label="shock", color="#F58518")
        ax.errorbar(x - width / 2, yb, yerr=[yb - lob, hib - yb], fmt="none", ecolor="black", capsize=3, lw=1)
        ax.errorbar(x + width / 2, ys, yerr=[ys - los, his - ys], fmt="none", ecolor="black", capsize=3, lw=1)

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Success rate")
        ax.set_title(f"Success rate by family (model={model})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=0)
        ax.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_deltas(deltas: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    required = {"model_name", "family", "delta_success_rate_shock_minus_baseline", "delta_ci95_lo", "delta_ci95_hi"}
    if not required.issubset(set(deltas.columns)):
        raise SystemExit(f"Delta file missing columns: {sorted(required - set(deltas.columns))}")

    deltas = deltas.copy()
    models = list(deltas["model_name"].dropna().unique())
    families = list(deltas["family"].dropna().unique())
    models.sort()
    families.sort()

    nrows = max(1, len(models))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(max(10, 1.4 * len(families) + 4), 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = deltas[deltas["model_name"] == model].set_index("family")
        x = np.arange(len(families))
        y = np.array([float(sub.loc[f, "delta_success_rate_shock_minus_baseline"]) if f in sub.index else np.nan for f in families], dtype=float)
        lo = np.array([float(sub.loc[f, "delta_ci95_lo"]) if f in sub.index else np.nan for f in families], dtype=float)
        hi = np.array([float(sub.loc[f, "delta_ci95_hi"]) if f in sub.index else np.nan for f in families], dtype=float)
        ax.axhline(0, color="black", lw=1)
        ax.bar(x, y, color="#E45756")
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_ylabel("Δ success (shock-baseline)")
        ax.set_title(f"Shock effect by family (model={model})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from benchmark CSVs.")
    parser.add_argument("--results", required=True, help="CSV produced by experiments/run_benchmark.py")
    parser.add_argument("--deltas", default=None, help="Shock deltas CSV (default: <results>_shock_deltas.csv)")
    parser.add_argument("--out-dir", default=None, help="Output directory for figures (default: <results_dir>/figures/)")
    args = parser.parse_args()

    results_path = args.results
    deltas_path = args.deltas or os.path.splitext(results_path)[0] + "_shock_deltas.csv"
    out_dir = args.out_dir or os.path.join(os.path.dirname(results_path) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = _load_results(results_path)
    summary = _summarize_success(df)
    _plot_success_rates(summary, os.path.join(out_dir, "fig_success_rates.png"))

    if os.path.exists(deltas_path):
        deltas = pd.read_csv(deltas_path)
        if not deltas.empty:
            _plot_deltas(deltas, os.path.join(out_dir, "fig_shock_deltas.png"))

    summary_out = os.path.join(out_dir, "success_rates_table.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Figures written to: {out_dir}")


if __name__ == "__main__":
    main()

