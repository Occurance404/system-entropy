from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_bool(value: object) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("No rows found in results CSV.")
    required = {"sample_index", "instance_id", "language", "status", "steps_executed", "validation_passed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    _to_numeric(
        df,
        [
            "sample_index",
            "steps_executed",
            "total_tokens",
            "probe_total_tokens",
            "total_tokens_including_probes",
            "max_scr",
            "probe_events",
        ],
    )
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["validation_passed_bool"] = df["validation_passed"].apply(_safe_bool)
    return df


def _plot_status_by_language(df: pd.DataFrame, out_path: str) -> None:
    agg = df.groupby(["language", "status"]).size().unstack(fill_value=0)
    for col in ("completed", "missing"):
        if col not in agg.columns:
            agg[col] = 0
    agg = agg.sort_index()

    x = np.arange(len(agg.index))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, agg["completed"].values, width=w, color="#2A9D8F", label="completed")
    ax.bar(x + w / 2, agg["missing"].values, width=w, color="#E76F51", label="missing")
    ax.set_title("Task Status by Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_validation_by_language(df: pd.DataFrame, out_path: str) -> None:
    done = df[df["status"] == "completed"].copy()
    done = done[done["validation_passed_bool"].notna()]
    if done.empty:
        return

    agg = done.groupby(["language", "validation_passed_bool"]).size().unstack(fill_value=0)
    for col in (True, False):
        if col not in agg.columns:
            agg[col] = 0
    agg = agg.sort_index()

    x = np.arange(len(agg.index))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, agg[True].values, width=w, color="#3A86FF", label="passed")
    ax.bar(x + w / 2, agg[False].values, width=w, color="#FF006E", label="failed")
    ax.set_title("Validation Outcome on Completed Tasks")
    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_steps_box(df: pd.DataFrame, out_path: str) -> None:
    done = df[df["status"] == "completed"].copy()
    if done.empty:
        return
    langs = sorted(done["language"].dropna().unique())
    data = [done.loc[done["language"] == lang, "steps_executed"].dropna().values for lang in langs]
    if not any(len(x) for x in data):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=langs, showmeans=True)
    ax.set_title("Steps Executed Distribution (Completed Tasks)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_tokens_box(df: pd.DataFrame, out_path: str) -> None:
    done = df[df["status"] == "completed"].copy()
    token_col = "total_tokens_including_probes"
    if done.empty or token_col not in done.columns:
        return

    langs = sorted(done["language"].dropna().unique())
    data = [done.loc[done["language"] == lang, token_col].dropna().values for lang in langs]
    if not any(len(x) for x in data):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=langs, showmeans=True)
    ax.set_yscale("log")
    ax.set_title("Total Tokens (Including Probes) by Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Tokens (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_scr_hist(df: pd.DataFrame, out_path: str) -> None:
    done = df[df["status"] == "completed"].copy()
    if done.empty or "max_scr" not in done.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for lang, sub in done.groupby("language"):
        vals = sub["max_scr"].dropna().values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=12, alpha=0.45, label=str(lang))
    ax.set_title("Max SCR Distribution (Completed Tasks)")
    ax.set_xlabel("max_scr")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_steps_tokens_scatter(df: pd.DataFrame, out_path: str) -> None:
    done = df[df["status"] == "completed"].copy()
    token_col = "total_tokens_including_probes"
    if done.empty or token_col not in done.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "Java": "#1D3557",
        "JavaScript": "#457B9D",
        "Python": "#2A9D8F",
        "TypeScript": "#E76F51",
    }
    for lang, sub in done.groupby("language"):
        ax.scatter(
            sub["steps_executed"],
            sub[token_col],
            alpha=0.7,
            s=36,
            label=str(lang),
            color=colors.get(str(lang), None),
        )
    ax.set_yscale("log")
    ax.set_title("Steps vs Total Tokens (Completed Tasks)")
    ax.set_xlabel("steps_executed")
    ax.set_ylabel("total_tokens_including_probes (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_dir: str) -> None:
    done = df[df["status"] == "completed"].copy()
    miss = df[df["status"] == "missing"].copy()

    payload: dict[str, object] = {}
    payload["total_tasks"] = int(len(df))
    payload["completed_tasks"] = int(len(done))
    payload["missing_tasks"] = int(len(miss))
    payload["completion_rate"] = float(len(done) / len(df)) if len(df) else 0.0

    if not done.empty:
        payload["avg_steps_completed"] = float(done["steps_executed"].dropna().mean())
        payload["median_steps_completed"] = float(done["steps_executed"].dropna().median())
        if "total_tokens_including_probes" in done.columns:
            payload["avg_tokens_completed"] = float(done["total_tokens_including_probes"].dropna().mean())
            payload["median_tokens_completed"] = float(done["total_tokens_including_probes"].dropna().median())
        if "max_scr" in done.columns:
            payload["avg_max_scr_completed"] = float(done["max_scr"].dropna().mean())
            payload["median_max_scr_completed"] = float(done["max_scr"].dropna().median())

        valid = done[done["validation_passed_bool"].notna()]
        if not valid.empty:
            passed = int((valid["validation_passed_bool"] == True).sum())  # noqa: E712
            failed = int((valid["validation_passed_bool"] == False).sum())  # noqa: E712
            payload["validation_passed_completed"] = passed
            payload["validation_failed_completed"] = failed
            payload["validation_pass_rate_completed"] = float(passed / len(valid))

    by_lang: dict[str, object] = {}
    for lang, sub in df.groupby("language"):
        item: dict[str, object] = {
            "total": int(len(sub)),
            "completed": int((sub["status"] == "completed").sum()),
            "missing": int((sub["status"] == "missing").sum()),
        }
        done_sub = sub[sub["status"] == "completed"]
        if not done_sub.empty:
            item["avg_steps"] = float(done_sub["steps_executed"].dropna().mean())
            if "max_scr" in done_sub.columns:
                item["avg_max_scr"] = float(done_sub["max_scr"].dropna().mean())
            valid_sub = done_sub[done_sub["validation_passed_bool"].notna()]
            if not valid_sub.empty:
                passed_sub = int((valid_sub["validation_passed_bool"] == True).sum())  # noqa: E712
                item["passed"] = passed_sub
                item["failed"] = int((valid_sub["validation_passed_bool"] == False).sum())  # noqa: E712
                item["pass_rate_on_completed"] = float(passed_sub / len(valid_sub))
        by_lang[str(lang)] = item
    payload["by_language"] = by_lang

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines: list[str] = []
    lines.append("# SWE-PolyBench Sample Summary")
    lines.append("")
    lines.append(f"- Total tasks: {payload['total_tasks']}")
    lines.append(f"- Completed: {payload['completed_tasks']}")
    lines.append(f"- Missing: {payload['missing_tasks']}")
    lines.append(f"- Completion rate: {payload['completion_rate']:.1%}")
    if "validation_passed_completed" in payload:
        lines.append(
            f"- Validation (completed): {payload['validation_passed_completed']} passed / "
            f"{payload['validation_failed_completed']} failed "
            f"({payload['validation_pass_rate_completed']:.1%} pass)"
        )
    if "avg_steps_completed" in payload:
        lines.append(f"- Avg steps (completed): {payload['avg_steps_completed']:.2f}")
    if "avg_tokens_completed" in payload:
        lines.append(f"- Avg tokens incl probes (completed): {payload['avg_tokens_completed']:.1f}")
    if "avg_max_scr_completed" in payload:
        lines.append(f"- Avg max SCR (completed): {payload['avg_max_scr_completed']:.3f}")
    lines.append("")
    lines.append("## Generated Plots")
    lines.append("- status_by_language.png")
    lines.append("- validation_by_language.png")
    lines.append("- steps_boxplot_by_language.png")
    lines.append("- tokens_boxplot_by_language.png")
    lines.append("- scr_histogram_by_language.png")
    lines.append("- steps_vs_tokens_scatter.png")

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SWE-PolyBench sample benchmark outcomes.")
    parser.add_argument(
        "--results",
        default="data/results/swepolybench_sample100_recovered_partial_20260218_231830.csv",
        help="CSV file for sample benchmark results.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/results/figures_<results_basename>).",
    )
    args = parser.parse_args()

    _set_plot_style()
    df = _load_results(args.results)
    out_dir = args.out_dir or os.path.join("data", "results", f"figures_{_basename_no_ext(args.results)}")
    _ensure_dir(out_dir)

    _plot_status_by_language(df, os.path.join(out_dir, "status_by_language.png"))
    _plot_validation_by_language(df, os.path.join(out_dir, "validation_by_language.png"))
    _plot_steps_box(df, os.path.join(out_dir, "steps_boxplot_by_language.png"))
    _plot_tokens_box(df, os.path.join(out_dir, "tokens_boxplot_by_language.png"))
    _plot_scr_hist(df, os.path.join(out_dir, "scr_histogram_by_language.png"))
    _plot_steps_tokens_scatter(df, os.path.join(out_dir, "steps_vs_tokens_scatter.png"))
    _write_summary(df, out_dir)

    print(f"Visualization bundle written to: {out_dir}")


if __name__ == "__main__":
    main()
