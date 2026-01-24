from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _scenario_family(scenario_id: str) -> tuple[str, str] | None:
    """
    Returns (family, condition) where condition is 'baseline' or 'shock'.
    Example: drug_filter_baseline -> ('drug_filter', 'baseline')
    """
    if not isinstance(scenario_id, str):
        return None
    scenario_id = scenario_id.strip()
    if scenario_id.endswith("_baseline"):
        return scenario_id[: -len("_baseline")], "baseline"
    if scenario_id.endswith("_shock"):
        return scenario_id[: -len("_shock")], "shock"
    return None


def _success_rate(x: pd.Series) -> float:
    s = x.dropna()
    if s.empty:
        return float("nan")
    return float((s == True).mean())  # noqa: E712


@dataclass(frozen=True)
class DeltaResult:
    delta: float
    lo: float
    hi: float


def _bootstrap_delta(
    baseline: np.ndarray,
    shock: np.ndarray,
    *,
    iters: int,
    seed: int,
) -> DeltaResult:
    """
    Bootstrap CI for difference in means (shock - baseline) on boolean arrays.
    """
    rng = np.random.default_rng(seed)

    baseline = baseline.astype(np.float32, copy=False)
    shock = shock.astype(np.float32, copy=False)

    if baseline.size == 0 or shock.size == 0:
        return DeltaResult(delta=float("nan"), lo=float("nan"), hi=float("nan"))

    delta_hat = float(shock.mean() - baseline.mean())

    deltas = np.empty(iters, dtype=np.float32)
    for i in range(iters):
        b = baseline[rng.integers(0, baseline.size, size=baseline.size)]
        s = shock[rng.integers(0, shock.size, size=shock.size)]
        deltas[i] = float(s.mean() - b.mean())

    lo, hi = np.percentile(deltas, [2.5, 97.5]).astype(float)
    return DeltaResult(delta=delta_hat, lo=float(lo), hi=float(hi))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline→shock deltas (with bootstrap CIs) from benchmark CSV.")
    parser.add_argument("--results", required=True, help="CSV produced by experiments/run_benchmark.py")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <results>_shock_deltas.csv)")
    parser.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap iterations for CI.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap.")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if df.empty:
        raise SystemExit("No rows found in results CSV.")

    if "scenario_id" not in df.columns or "model_name" not in df.columns:
        raise SystemExit("Missing required columns: scenario_id, model_name")

    df = df.copy()
    if "validation_passed" not in df.columns:
        raise SystemExit("Missing required column: validation_passed")

    df["pair"] = df["scenario_id"].apply(_scenario_family)
    df = df[df["pair"].notna()]
    if df.empty:
        raise SystemExit("No baseline/shock scenarios found (expected *_baseline and *_shock).")

    df["family"] = df["pair"].apply(lambda x: x[0])
    df["condition"] = df["pair"].apply(lambda x: x[1])

    group_cols = ["model_name", "family"]
    if "metric_embedding_backend" in df.columns:
        group_cols.append("metric_embedding_backend")

    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        g_base = g[g["condition"] == "baseline"]
        g_shock = g[g["condition"] == "shock"]
        if g_base.empty or g_shock.empty:
            continue

        b = pd.to_numeric(g_base["validation_passed"], errors="coerce").dropna().to_numpy(dtype=np.float32)
        s = pd.to_numeric(g_shock["validation_passed"], errors="coerce").dropna().to_numpy(dtype=np.float32)

        delta = _bootstrap_delta(b, s, iters=max(100, int(args.bootstrap_iters)), seed=int(args.seed))
        row: dict[str, Any] = {k: v for k, v in zip(group_cols, key if isinstance(key, tuple) else (key,))}
        row.update(
            {
                "n_baseline": int(len(b)),
                "n_shock": int(len(s)),
                "success_rate_baseline": float(b.mean()) if len(b) else float("nan"),
                "success_rate_shock": float(s.mean()) if len(s) else float("nan"),
                "delta_success_rate_shock_minus_baseline": float(delta.delta),
                "delta_ci95_lo": float(delta.lo),
                "delta_ci95_hi": float(delta.hi),
            }
        )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No complete (baseline, shock) pairs found after grouping.")

    out_df = out_df.sort_values(group_cols).reset_index(drop=True)
    out_path = args.out or os.path.splitext(args.results)[0] + "_shock_deltas.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Shock delta table saved to: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

