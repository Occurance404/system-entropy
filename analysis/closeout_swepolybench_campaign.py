from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_bool(v: object) -> bool:
    s = str(v).strip().lower()
    return s in {"1", "true", "yes"}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _short_model(name: str) -> str:
    mapping = {
        "moonshotai/kimi-k2.5": "kimi-k2.5",
        "deepseek/deepseek-v3.2": "deepseek-v3.2",
        "z-ai/glm-4.7": "glm-4.7",
        "openai/gpt-5-mini": "gpt-5-mini",
        "google/gemini-3-flash-preview": "gemini-3-flash",
        "minimax/minimax-m2.5": "minimax-m2.5",
        "qwen/qwen3-coder-next": "qwen3-coder-next",
    }
    return mapping.get(name, name.split("/")[-1])


def _error_category(err: str) -> str:
    e = (err or "").strip()
    if not e:
        return "none"
    if "Could not resolve host: github.com" in e or "Failed to update mirror for" in e:
        return "dns_mirror_fetch"
    if "NoneType' object has no attribute 'splitlines'" in e:
        return "validator_none_splitlines"
    if "Request timed out" in e:
        return "api_timeout"
    if "Event loop is closed" in e:
        return "event_loop_closed"
    if "No space left on device" in e:
        return "disk_full"
    return "other"


def _get_model_order(models_json: Path) -> list[str]:
    if not models_json.exists():
        return []
    try:
        arr = json.loads(models_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for m in arr:
        model = str(m.get("model", "")).strip()
        if model:
            out.append(model)
    return out


def _sort_by_model_order(df: pd.DataFrame, model_col: str, order: list[str]) -> pd.DataFrame:
    if not order:
        return df
    rank = {m: i for i, m in enumerate(order)}
    return df.assign(_ord=df[model_col].map(lambda x: rank.get(x, 10_000))).sort_values("_ord").drop(columns="_ord")


@dataclass
class Paths:
    run_tag: str
    run_dir: Path
    merged_csv: Path
    model_summary_all: Path
    model_summary_clean: Path
    failure_comp_all: Path
    failure_comp_clean: Path
    error_counts: Path
    probe_per_run: Path
    probe_summary_model: Path
    probe_summary_success: Path
    chunk_timing: Path
    report_md: Path
    fig_dir: Path


def _paths(run_tag: str) -> Paths:
    return Paths(
        run_tag=run_tag,
        run_dir=Path(f"data/results/rerun_{run_tag}"),
        merged_csv=Path(f"data/results/benchmark_swepolybench_models_{run_tag}.csv"),
        model_summary_all=Path(f"data/results/swepolybench_models_{run_tag}_model_summary_all.csv"),
        model_summary_clean=Path(f"data/results/swepolybench_models_{run_tag}_model_summary_clean_noninfra.csv"),
        failure_comp_all=Path(f"data/results/swepolybench_models_{run_tag}_failure_composition_all.csv"),
        failure_comp_clean=Path(f"data/results/swepolybench_models_{run_tag}_failure_composition_clean_noninfra.csv"),
        error_counts=Path(f"data/results/swepolybench_models_{run_tag}_error_category_counts.csv"),
        probe_per_run=Path(f"data/results/swepolybench_models_{run_tag}_probe_scr_per_run.csv"),
        probe_summary_model=Path(f"data/results/swepolybench_models_{run_tag}_probe_scr_summary_by_model.csv"),
        probe_summary_success=Path(f"data/results/swepolybench_models_{run_tag}_probe_scr_summary_by_success.csv"),
        chunk_timing=Path(f"data/results/swepolybench_models_{run_tag}_chunk_timing.csv"),
        report_md=Path(f"data/results/swepolybench_models_{run_tag}_CLOSEOUT_REPORT.md"),
        fig_dir=Path(f"data/results/figures_models_{run_tag}_final"),
    )


def _load_chunks(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("*/*_chunk_*.csv"))
    if not files:
        raise SystemExit(f"No chunk CSVs found under {run_dir}")
    frames: list[pd.DataFrame] = []
    for p in files:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        stem = p.stem
        chunk = stem.split("_chunk_")[-1] if "_chunk_" in stem else ""
        df["chunk_id"] = f"chunk_{chunk}" if chunk else ""
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "model_name" not in out.columns and "model" in out.columns:
        out["model_name"] = out["model"].astype(str)
    return out


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["success"] = out.get("validation_passed", False).map(_safe_bool)
    out["steps_executed"] = _to_num(out.get("steps_executed", pd.Series([0] * len(out))))
    out["total_tokens"] = _to_num(out.get("total_tokens", pd.Series([0] * len(out))))
    out["probe_total_tokens"] = _to_num(out.get("probe_total_tokens", pd.Series([0] * len(out))))
    if "total_tokens_including_probes" in out.columns:
        out["total_tokens_including_probes"] = _to_num(out["total_tokens_including_probes"])
    else:
        out["total_tokens_including_probes"] = out["total_tokens"] + out["probe_total_tokens"]
    out["error"] = out.get("error", pd.Series([""] * len(out))).fillna("").astype(str)
    out["error_category"] = out["error"].map(_error_category)
    out["infra_prefetch_failure"] = out["error_category"].eq("dns_mirror_fetch")
    out["zero_step"] = out["steps_executed"].eq(0)
    out["zero_step_noninfra_failure"] = (~out["success"]) & (~out["infra_prefetch_failure"]) & out["zero_step"]
    out["agent_executed_failure"] = (~out["success"]) & (~out["infra_prefetch_failure"]) & (~out["zero_step"])
    out["other_failure"] = (~out["success"]) & (~out["infra_prefetch_failure"]) & (~out["zero_step_noninfra_failure"]) & (~out["agent_executed_failure"])
    out["model_short"] = out["model_name"].astype(str).map(_short_model)
    return out


def _model_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model_name", dropna=False)
    out = (
        pd.DataFrame(
            {
                "runs": g.size(),
                "successes": g["success"].sum(),
                "success_rate": g["success"].mean(),
                "infra_prefetch_failures": g["infra_prefetch_failure"].sum(),
                "infra_prefetch_failure_rate": g["infra_prefetch_failure"].mean(),
                "zero_step_runs": g["zero_step"].sum(),
                "zero_step_rate": g["zero_step"].mean(),
                "median_steps": g["steps_executed"].median(),
                "median_total_tokens": g["total_tokens"].median(),
                "median_probe_tokens": g["probe_total_tokens"].median(),
                "median_total_tokens_incl_probes": g["total_tokens_including_probes"].median(),
                "total_tokens_incl_probes_sum": g["total_tokens_including_probes"].sum(),
            }
        )
        .reset_index()
        .assign(
            probe_share_of_total=lambda x: np.where(
                x["median_total_tokens_incl_probes"] > 0,
                x["median_probe_tokens"] / x["median_total_tokens_incl_probes"],
                np.nan,
            )
        )
    )
    out["successes_per_million_tokens"] = np.where(
        out["total_tokens_incl_probes_sum"] > 0,
        out["successes"] / (out["total_tokens_incl_probes_sum"] / 1_000_000.0),
        np.nan,
    )
    return out


def _failure_composition(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for m, sub in df.groupby("model_name", dropna=False):
        runs = len(sub)
        components = {
            "success": int(sub["success"].sum()),
            "infra_prefetch_failure": int(sub["infra_prefetch_failure"].sum()),
            "zero_step_noninfra_failure": int(sub["zero_step_noninfra_failure"].sum()),
            "agent_executed_failure": int(sub["agent_executed_failure"].sum()),
            "other": max(
                0,
                runs
                - int(sub["success"].sum())
                - int(sub["infra_prefetch_failure"].sum())
                - int(sub["zero_step_noninfra_failure"].sum())
                - int(sub["agent_executed_failure"].sum()),
            ),
        }
        for cls, cnt in components.items():
            rows.append(
                {
                    "model_name": m,
                    "failure_class": cls,
                    "count": cnt,
                    "runs": runs,
                    "rate": (cnt / runs) if runs else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _extract_probe_scr(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        log_file = str(r.get("log_file", "") or "")
        if not log_file or not os.path.exists(log_file):
            rows.append(
                {
                    "run_id": r.get("run_id"),
                    "scenario_id": r.get("scenario_id"),
                    "model_name": r.get("model_name"),
                    "success": bool(r.get("success")),
                    "probe_events": 0,
                    "probe_scr_count": 0,
                    "probe_scr_median": np.nan,
                    "probe_scr_max": np.nan,
                    "probe_scr_min": np.nan,
                    "probe_scr_range": np.nan,
                    "probe_scr_last": np.nan,
                }
            )
            continue

        probe_events = 0
        scr_vals: list[float] = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("event_type") != "periodic_probe":
                        continue
                    probe_events += 1
                    scr = ev.get("scr")
                    if scr is None:
                        scr = (ev.get("metrics") or {}).get("scr")
                    if scr is None:
                        continue
                    try:
                        scr_vals.append(float(scr))
                    except Exception:
                        continue
        except Exception:
            pass

        rows.append(
            {
                "run_id": r.get("run_id"),
                "scenario_id": r.get("scenario_id"),
                "model_name": r.get("model_name"),
                "success": bool(r.get("success")),
                "probe_events": int(probe_events),
                "probe_scr_count": int(len(scr_vals)),
                "probe_scr_median": float(np.median(scr_vals)) if scr_vals else np.nan,
                "probe_scr_max": float(np.max(scr_vals)) if scr_vals else np.nan,
                "probe_scr_min": float(np.min(scr_vals)) if scr_vals else np.nan,
                "probe_scr_range": float(np.max(scr_vals) - np.min(scr_vals)) if scr_vals else np.nan,
                "probe_scr_last": float(scr_vals[-1]) if scr_vals else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _probe_summary_by_model(probe: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    total_runs = df.groupby("model_name").size().rename("total_runs").reset_index()
    g = probe.groupby("model_name", dropna=False)
    summ = (
        pd.DataFrame(
            {
                "runs_with_probe_logs": g["probe_events"].apply(lambda s: int((s > 0).sum())),
                "runs_with_probe_scr": g["probe_scr_count"].apply(lambda s: int((s > 0).sum())),
                "median_probe_events": g["probe_events"].median(),
                "mean_probe_events": g["probe_events"].mean(),
                "median_probe_scr_median": g["probe_scr_median"].median(),
                "median_probe_scr_max": g["probe_scr_max"].median(),
                "median_probe_scr_range": g["probe_scr_range"].median(),
                "median_probe_scr_last": g["probe_scr_last"].median(),
            }
        )
        .reset_index()
        .merge(total_runs, on="model_name", how="left")
    )
    summ["probe_log_coverage"] = np.where(
        summ["total_runs"] > 0, summ["runs_with_probe_logs"] / summ["total_runs"], np.nan
    )
    return summ


def _chunk_timing(progress_tsv: Path) -> pd.DataFrame:
    if not progress_tsv.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    with open(progress_tsv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            chunk = row.get("chunk", "")
            n = np.nan
            if chunk.startswith("chunk_"):
                try:
                    n = int(chunk.split("_")[1])
                except Exception:
                    n = np.nan
            rows.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "model_name": row.get("model_name", ""),
                    "chunk": chunk,
                    "chunk_num": n,
                    "status": row.get("status", ""),
                    "out_csv": row.get("out_csv", ""),
                    "note": row.get("note", ""),
                }
            )
    return pd.DataFrame(rows)


def _set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _model_labels(summary: pd.DataFrame) -> list[str]:
    return [f"{_short_model(m)} (n={int(n)})" for m, n in zip(summary["model_name"], summary["runs"])]


def _plot_workload_panel(summary: pd.DataFrame, out: Path, title: str) -> None:
    s = summary.copy()
    labels = _model_labels(s)
    x = np.arange(len(s))
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()

    axs[0].bar(x, s["success_rate"], color="#2a9d8f")
    axs[0].set_title("Success rate")
    axs[0].set_ylim(0, 1)
    for i, v in enumerate(s["success_rate"]):
        axs[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    axs[1].bar(x, s["zero_step_rate"], color="#e76f51")
    axs[1].set_title("Zero-step failure rate")
    axs[1].set_ylim(0, 1)
    for i, v in enumerate(s["zero_step_rate"]):
        axs[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    axs[2].bar(x, s["median_steps"], color="#457b9d")
    axs[2].set_title("Median steps")
    for i, v in enumerate(s["median_steps"]):
        axs[2].text(i, v + 0.4, f"{v:.0f}", ha="center", fontsize=9)

    axs[3].bar(x, s["median_total_tokens_incl_probes"] / 1000.0, color="#6a4c93")
    axs[3].set_title("Median total tokens incl. probes")
    axs[3].set_ylabel("Tokens (k)")
    for i, v in enumerate(s["median_total_tokens_incl_probes"]):
        axs[3].text(i, (v / 1000.0) + 2.0, f"{v/1000:.0f}k", ha="center", fontsize=9)

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle(title, fontsize=15)
    _save(fig, out)


def _plot_failure_comp(comp: pd.DataFrame, out: Path, title: str) -> None:
    classes = ["success", "infra_prefetch_failure", "zero_step_noninfra_failure", "agent_executed_failure", "other"]
    colors = {
        "success": "#2ca02c",
        "infra_prefetch_failure": "#d62728",
        "zero_step_noninfra_failure": "#9467bd",
        "agent_executed_failure": "#ff7f0e",
        "other": "#7f7f7f",
    }
    pivot = comp.pivot(index="model_name", columns="failure_class", values="rate").fillna(0.0)
    for c in classes:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot[classes]
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    fig, ax = plt.subplots(figsize=(12, 7))
    for c in classes:
        vals = pivot[c].values
        ax.bar(x, vals, bottom=bottom, label=c, color=colors[c])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in pivot.index], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of runs")
    ax.set_title(title)
    ax.legend(ncol=2)
    _save(fig, out)


def _plot_probe_dist(probe: pd.DataFrame, probe_summary: pd.DataFrame, out: Path, title: str) -> None:
    use = probe[probe["probe_scr_count"] > 0].copy()
    if use.empty:
        return
    models = sorted(use["model_name"].dropna().unique().tolist())
    data = [use.loc[use["model_name"] == m, "probe_scr_median"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.boxplot(data, tick_labels=[_short_model(m) for m in models], showmeans=True, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("#8fb3d0")
        b.set_alpha(0.6)
    rng = np.random.default_rng(42)
    for i, m in enumerate(models, start=1):
        vals = use.loc[use["model_name"] == m, "probe_scr_median"].dropna().values
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(x, vals, s=14, alpha=0.45, color="#1f77b4")
        row = probe_summary.loc[probe_summary["model_name"] == m]
        if not row.empty:
            cov = float(row["runs_with_probe_logs"].iloc[0])
            total = float(row["total_runs"].iloc[0])
            ax.text(i, 1.02, f"cov={int(cov)}/{int(total)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Per-run SCR_probe (median over periodic probes)")
    _save(fig, out)


def _plot_probe_coverage(probe_summary: pd.DataFrame, out: Path, title: str) -> None:
    s = probe_summary.copy()
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, s["probe_log_coverage"], color="#3a86ff")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in s["model_name"]], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Coverage")
    for i, (cov, n, t) in enumerate(zip(s["probe_log_coverage"], s["runs_with_probe_logs"], s["total_runs"])):
        ax.text(i, cov + 0.02, f"{cov:.2f}\n({int(n)}/{int(t)})", ha="center", fontsize=8)
    _save(fig, out)


def _plot_infra_rate(summary_all: pd.DataFrame, out: Path, title: str) -> None:
    s = summary_all.copy()
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, s["infra_prefetch_failure_rate"], color="#d62728")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in s["model_name"]], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Rate")
    for i, v in enumerate(s["infra_prefetch_failure_rate"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    _save(fig, out)


def _plot_token_breakdown(summary_all: pd.DataFrame, out: Path, title: str) -> None:
    s = summary_all.copy()
    x = np.arange(len(s))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, s["median_total_tokens"] / 1000.0, width=w, label="task tokens", color="#457b9d")
    ax.bar(x + w / 2, s["median_probe_tokens"] / 1000.0, width=w, label="probe tokens", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in s["model_name"]], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Median tokens (k)")
    ax.legend()
    _save(fig, out)


def _plot_success_per_million(summary_all: pd.DataFrame, out: Path, title: str) -> None:
    s = summary_all.copy()
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, s["successes_per_million_tokens"], color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_model(m) for m in s["model_name"]], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Successes per 1M tokens (incl. probes)")
    for i, v in enumerate(s["successes_per_million_tokens"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    _save(fig, out)


def _plot_chunk_timeline(chunk_timing: pd.DataFrame, out: Path, title: str) -> None:
    if chunk_timing.empty:
        return
    ok = chunk_timing[chunk_timing["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["timestamp"] = pd.to_datetime(ok["timestamp"], errors="coerce")
    ok = ok.dropna(subset=["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for m, sub in ok.groupby("model_name"):
        sub = sub.sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["chunk_num"], marker="o", linewidth=1.4, label=_short_model(str(m)))
    ax.set_title(title)
    ax.set_ylabel("Chunk number")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, out)


def _write_report(paths: Paths, summary_all: pd.DataFrame, summary_clean: pd.DataFrame, probe_summary: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append(f"# SWE-PolyBench Campaign Closeout ({paths.run_tag})")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- merged csv: `{paths.merged_csv}`")
    lines.append(f"- summary all: `{paths.model_summary_all}`")
    lines.append(f"- summary clean noninfra: `{paths.model_summary_clean}`")
    lines.append(f"- probe summary: `{paths.probe_summary_model}`")
    lines.append(f"- figures: `{paths.fig_dir}`")
    lines.append("")
    lines.append("## Model Summary (All Rows)")
    lines.append(summary_all.to_markdown(index=False))
    lines.append("")
    lines.append("## Model Summary (Non-Infra Rows)")
    lines.append(summary_clean.to_markdown(index=False))
    lines.append("")
    lines.append("## Probe Summary")
    lines.append(probe_summary.to_markdown(index=False))
    lines.append("")
    paths.report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--models-json", default="benchmarks/models.openrouter.remaining5.json")
    args = ap.parse_args()

    paths = _paths(args.run_tag)
    _ensure_dir(paths.fig_dir)

    model_order = _get_model_order(Path(args.models_json))
    raw = _load_chunks(paths.run_dir)
    df = _derive(raw)
    df.to_csv(paths.merged_csv, index=False)

    summary_all = _model_summary(df)
    summary_all = _sort_by_model_order(summary_all, "model_name", model_order)
    summary_all.to_csv(paths.model_summary_all, index=False)

    clean = df[~df["infra_prefetch_failure"]].copy()
    summary_clean = _model_summary(clean)
    summary_clean = _sort_by_model_order(summary_clean, "model_name", model_order)
    summary_clean.to_csv(paths.model_summary_clean, index=False)

    comp_all = _failure_composition(df)
    comp_all = _sort_by_model_order(comp_all, "model_name", model_order)
    comp_all.to_csv(paths.failure_comp_all, index=False)

    comp_clean = _failure_composition(clean)
    comp_clean = _sort_by_model_order(comp_clean, "model_name", model_order)
    comp_clean.to_csv(paths.failure_comp_clean, index=False)

    err = (
        df[df["error"].str.len() > 0]
        .groupby(["model_name", "error_category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    err = _sort_by_model_order(err, "model_name", model_order)
    err.to_csv(paths.error_counts, index=False)

    probe = _extract_probe_scr(df)
    probe = _sort_by_model_order(probe, "model_name", model_order)
    probe.to_csv(paths.probe_per_run, index=False)

    probe_summary = _probe_summary_by_model(probe, df)
    probe_summary = _sort_by_model_order(probe_summary, "model_name", model_order)
    probe_summary.to_csv(paths.probe_summary_model, index=False)

    probe_success = (
        probe[probe["probe_scr_count"] > 0]
        .groupby(["model_name", "success"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            median_probe_scr_median=("probe_scr_median", "median"),
            median_probe_scr_max=("probe_scr_max", "median"),
        )
        .reset_index()
    )
    probe_success = _sort_by_model_order(probe_success, "model_name", model_order)
    probe_success.to_csv(paths.probe_summary_success, index=False)

    ct = _chunk_timing(paths.run_dir / "progress.tsv")
    if not ct.empty:
        ct.to_csv(paths.chunk_timing, index=False)

    _set_style()
    _plot_workload_panel(summary_all, paths.fig_dir / "fig_workload_summary_panel_all.png", "SWE-PolyBench workload summary (all rows)")
    _plot_workload_panel(
        summary_clean, paths.fig_dir / "fig_workload_summary_panel_clean_noninfra.png", "SWE-PolyBench workload summary (non-infra rows)"
    )
    _plot_failure_comp(comp_all, paths.fig_dir / "fig_failure_composition_all.png", "Failure composition by model (all rows)")
    _plot_failure_comp(
        comp_clean, paths.fig_dir / "fig_failure_composition_clean_noninfra.png", "Failure composition by model (non-infra rows)"
    )
    _plot_probe_dist(
        probe, probe_summary, paths.fig_dir / "fig_scr_probe_distribution.png", "SCR_probe distribution and coverage (campaign)"
    )
    _plot_probe_coverage(probe_summary, paths.fig_dir / "fig_probe_log_coverage.png", "Probe-log coverage by model")
    _plot_infra_rate(summary_all, paths.fig_dir / "fig_infra_prefetch_failure_rate.png", "Infrastructure prefetch failure rate")
    _plot_token_breakdown(summary_all, paths.fig_dir / "fig_token_breakdown_task_vs_probe.png", "Median token breakdown by model")
    _plot_success_per_million(
        summary_all,
        paths.fig_dir / "fig_successes_per_million_tokens.png",
        "Cost efficiency: successes per 1M tokens",
    )
    _plot_chunk_timeline(ct, paths.fig_dir / "fig_chunk_completion_timeline.png", "Chunk completion timeline")

    _write_report(paths, summary_all, summary_clean, probe_summary)

    print(f"[ok] merged: {paths.merged_csv}")
    print(f"[ok] summary all: {paths.model_summary_all}")
    print(f"[ok] summary clean: {paths.model_summary_clean}")
    print(f"[ok] probe summary: {paths.probe_summary_model}")
    print(f"[ok] figures dir: {paths.fig_dir}")
    print(f"[ok] report: {paths.report_md}")


if __name__ == "__main__":
    main()
