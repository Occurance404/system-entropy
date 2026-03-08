from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_bool(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _cat(error: str, details: str, steps: float, success: bool) -> str:
    e = (error or "").strip()
    d = (details or "").strip()
    if "Could not resolve host: github.com" in e or "Failed to update mirror for" in e:
        return "infra_dns_mirror_fetch"
    if "NoneType' object has no attribute 'splitlines'" in e:
        return "validator_none_splitlines"
    if "Request timed out" in e:
        return "api_timeout"
    if "No space left on device" in e:
        return "disk_full"
    if not success and steps == 0:
        return "zero_step_other"
    if "COMPILATION ERROR" in d or "cannot find symbol" in d or "Compilation failure" in d:
        return "compile_error"
    if "There are test failures" in d or "FAILURES!!!" in d or "AssertionError" in d:
        return "test_failure"
    if "npm ERR!" in d or ("yarn" in d and "error" in d.lower()):
        return "js_build_or_test_failure"
    if "validator_exit=124" in d or "timed out" in d.lower():
        return "validator_timeout"
    if not success:
        return "agent_executed_other_failure"
    return "success"


def _short(m: str) -> str:
    mapping = {
        "moonshotai/kimi-k2.5": "kimi-k2.5",
        "deepseek/deepseek-v3.2": "deepseek-v3.2",
        "z-ai/glm-4.7": "glm-4.7",
        "openai/gpt-5-mini": "gpt-5-mini",
        "google/gemini-3-flash-preview": "gemini-3-flash",
        "minimax/minimax-m2.5": "minimax-m2.5",
        "qwen/qwen3-coder-next": "qwen3-coder-next",
    }
    return mapping.get(m, m.split("/")[-1])


def _set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_overall_categories(cat_counts: pd.DataFrame, out: Path) -> None:
    df = cat_counts.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["failure_category"], df["count"], color="#457b9d")
    ax.set_title("Failure Categories (all failed runs)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(df["count"]):
        ax.text(i, v + 1, f"{int(v)}", ha="center", fontsize=9)
    _save(fig, out)


def _plot_categories_by_model(by_model_cat: pd.DataFrame, out: Path) -> None:
    pivot = by_model_cat.pivot(index="model_name", columns="failure_category", values="count").fillna(0)
    cols = [c for c in [
        "infra_dns_mirror_fetch",
        "zero_step_other",
        "validator_none_splitlines",
        "compile_error",
        "test_failure",
        "js_build_or_test_failure",
        "api_timeout",
        "disk_full",
        "agent_executed_other_failure",
    ] if c in pivot.columns]
    pivot = pivot[cols]

    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [
        "#d62828",
        "#f77f00",
        "#8338ec",
        "#3a86ff",
        "#2a9d8f",
        "#ff006e",
        "#6c757d",
        "#b56576",
        "#ffb703",
    ]
    for i, c in enumerate(cols):
        vals = pivot[c].values
        ax.bar(x, vals, bottom=bottom, label=c, color=colors[i % len(colors)])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in pivot.index], rotation=15, ha="right")
    ax.set_title("Failure Categories by Model (count)")
    ax.set_ylabel("Failed runs")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out)


def _plot_repo_failures(repo_counts: pd.DataFrame, out: Path) -> None:
    top = repo_counts.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["repo"], top["count"], color="#264653")
    ax.set_title("Top Repositories by Failure Count (non-infra)")
    ax.set_xlabel("Failure runs")
    for i, v in enumerate(top["count"]):
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=9)
    _save(fig, out)


def _plot_hard_instance_heatmap(hard_matrix: pd.DataFrame, out: Path) -> None:
    if hard_matrix.empty:
        return
    mat = hard_matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(hard_matrix.index)))
    ax.set_yticklabels(hard_matrix.index)
    ax.set_xticks(np.arange(len(hard_matrix.columns)))
    ax.set_xticklabels([_short(c) for c in hard_matrix.columns], rotation=20, ha="right")
    ax.set_title("Hard Instances Heatmap (non-infra failure rate by model)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Failure rate")
    _save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    in_csv = Path(f"data/results/benchmark_swepolybench_models_{args.run_tag}.csv")
    if not in_csv.exists():
        raise SystemExit(f"Missing merged CSV: {in_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/results/figures_models_{args.run_tag}_forensics")
    _ensure_dir(out_dir)

    df = pd.read_csv(in_csv)
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"].astype(str)
    df["success"] = df["validation_passed"].map(_safe_bool)
    df["steps_executed"] = pd.to_numeric(df.get("steps_executed", 0), errors="coerce").fillna(0)
    df["error"] = df.get("error", "").fillna("").astype(str)
    df["validation_details"] = df.get("validation_details", "").fillna("").astype(str)
    df["failure_category"] = df.apply(
        lambda r: _cat(
            error=r["error"],
            details=r["validation_details"],
            steps=float(r["steps_executed"]),
            success=bool(r["success"]),
        ),
        axis=1,
    )

    failed = df[~df["success"]].copy()
    noninfra = failed[failed["failure_category"] != "infra_dns_mirror_fetch"].copy()

    cat_counts = failed.groupby("failure_category").size().reset_index(name="count").sort_values("count", ascending=False)
    by_model_cat = (
        failed.groupby(["model_name", "failure_category"]).size().reset_index(name="count").sort_values(["model_name", "count"], ascending=[True, False])
    )
    repo_counts = (
        noninfra.groupby("repo").size().reset_index(name="count").sort_values("count", ascending=False)
    )

    # Hard instances: seen by >=3 models, failure rate per model
    fail_matrix = (
        noninfra.assign(failed=1)
        .pivot_table(index="instance_id", columns="model_name", values="failed", aggfunc="max", fill_value=0.0)
    )
    seen = (
        df.assign(seen=1)
        .pivot_table(index="instance_id", columns="model_name", values="seen", aggfunc="max", fill_value=0.0)
    )
    common_instances = seen[seen.sum(axis=1) >= 3].index
    fail_matrix = fail_matrix.loc[fail_matrix.index.intersection(common_instances)]
    hard_order = fail_matrix.sum(axis=1).sort_values(ascending=False).head(25).index
    hard_matrix = fail_matrix.loc[hard_order].sort_values(by=list(fail_matrix.columns), ascending=False)

    # Save tables
    cat_counts.to_csv(out_dir / "tbl_failure_category_counts.csv", index=False)
    by_model_cat.to_csv(out_dir / "tbl_failure_category_by_model.csv", index=False)
    repo_counts.to_csv(out_dir / "tbl_repo_failure_counts_noninfra.csv", index=False)
    hard_matrix.to_csv(out_dir / "tbl_hard_instances_heatmap_matrix.csv")

    # Plots
    _set_style()
    _plot_overall_categories(cat_counts, out_dir / "fig_forensics_failure_categories_overall.png")
    _plot_categories_by_model(by_model_cat, out_dir / "fig_forensics_failure_categories_by_model.png")
    _plot_repo_failures(repo_counts, out_dir / "fig_forensics_repo_failures_noninfra_top15.png")
    _plot_hard_instance_heatmap(hard_matrix, out_dir / "fig_forensics_hard_instances_heatmap_noninfra.png")

    # Markdown summary
    md = []
    md.append(f"# Failure Forensics ({args.run_tag})")
    md.append("")
    md.append(f"- total runs: `{len(df)}`")
    md.append(f"- failed runs: `{len(failed)}`")
    md.append(f"- non-infra failed runs: `{len(noninfra)}`")
    md.append("")
    md.append("## Top Failure Categories")
    md.append(cat_counts.head(10).to_markdown(index=False))
    md.append("")
    md.append("## Top Repositories by Non-Infra Failures")
    md.append(repo_counts.head(15).to_markdown(index=False))
    (out_dir / "FAILURE_FORENSICS_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[ok] forensics dir: {out_dir}")
    print(f"[ok] report: {out_dir / 'FAILURE_FORENSICS_REPORT.md'}")


if __name__ == "__main__":
    main()
