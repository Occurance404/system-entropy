import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROBE_EVENT_TYPES = {
    "periodic_probe",
    "perturbation_triggered",
    "proxy_probe",
    "proxy_shock_injected",
}

METRIC_LABELS = {
    "scr": "Semantic Collapse Ratio (SCR)",
    "current_entropy": "Entropy",
    "compression_ratio": "Compression Ratio",
    "ige": "Information Gain Efficiency (IGE)",
    "rdi": "Regressive Debt Index (RDI)",
}


def read_first_entry(path):
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def load_log(path):
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def select_logs(log_dir, log_files, scenario_ids, exclude_scenarios, include_all_runs):
    if log_files:
        paths = log_files
    else:
        paths = glob.glob(os.path.join(log_dir, "tb_monitor_*.jsonl"))

    candidates = []
    for path in paths:
        first_entry = read_first_entry(path)
        if not first_entry:
            continue
        scenario_id = first_entry.get("scenario_id", "unknown")
        if scenario_ids and scenario_id not in scenario_ids:
            continue
        if exclude_scenarios and scenario_id in exclude_scenarios:
            continue
        candidates.append(
            {
                "path": path,
                "scenario_id": scenario_id,
                "run_id": first_entry.get("run_id", ""),
                "mtime": os.path.getmtime(path),
            }
        )

    if include_all_runs:
        return candidates

    latest = {}
    for candidate in candidates:
        scenario_id = candidate["scenario_id"]
        if scenario_id not in latest or candidate["mtime"] > latest[scenario_id]["mtime"]:
            latest[scenario_id] = candidate
    return list(latest.values())


def build_metric_series(df, metric, probe_only):
    if df.empty:
        return pd.Series(dtype=float), 0

    df = df.copy()
    if "step_index" not in df.columns:
        df["step_index"] = range(1, len(df) + 1)
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce")
    df = df[df["step_index"].notna()]

    max_step = int(df["step_index"].max()) if not df.empty else 0
    if metric not in df.columns:
        return pd.Series(dtype=float), max_step

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    metric_df = df[df[metric].notna()]

    if probe_only and metric == "scr" and "event_type" in df.columns:
        metric_df = metric_df[metric_df["event_type"].isin(PROBE_EVENT_TYPES)]

    if metric_df.empty:
        return pd.Series(dtype=float), max_step

    series = metric_df.groupby("step_index")[metric].mean()
    return series, max_step


def format_label(entry, pretty_labels, include_run_id):
    label = entry["scenario_id"] or "unknown"
    if pretty_labels:
        label = label.replace("_", " ").title()
    if include_run_id:
        suffix = entry.get("run_id") or os.path.splitext(os.path.basename(entry["path"]))[0]
        label = f"{label} ({suffix[:8]})"
    return label


def build_heatmap_matrix(entries, metric, probe_only, max_steps):
    series_by_label = {}
    max_step = 0
    for entry in entries:
        df = load_log(entry["path"])
        series, entry_max_step = build_metric_series(df, metric, probe_only)
        label = entry["label"]
        series_by_label[label] = series
        max_step = max(max_step, entry_max_step)

    if max_steps is not None:
        max_step = min(max_step, max_steps)

    if max_step == 0:
        return None, None

    matrix = []
    for label in [entry["label"] for entry in entries]:
        row = np.full(max_step, np.nan)
        series = series_by_label.get(label)
        if series is not None and not series.empty:
            for step, value in series.items():
                step_index = int(step)
                if 1 <= step_index <= max_step:
                    row[step_index - 1] = value
        matrix.append(row)

    return np.array(matrix), max_step


def choose_tick_step(max_step):
    if max_step <= 50:
        return 5
    if max_step <= 100:
        return 10
    if max_step <= 200:
        return 20
    if max_step <= 300:
        return 25
    return 50


def plot_heatmap(matrix, entries, max_step, metric, output_path, title, vmin, vmax):
    values = matrix[~np.isnan(matrix)]
    if values.size == 0:
        print("No metric values found to plot.")
        return

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    diverging = vmin < 0 < vmax
    if diverging:
        cmap = sns.color_palette("vlag", as_cmap=True)
        center = 0
    else:
        cmap = sns.color_palette("rocket", as_cmap=True)
        center = None

    fig_width = max(12, min(36, max_step * 0.08))
    fig_height = max(3, 0.6 * len(entries) + 2)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        mask=np.isnan(matrix),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
        ax=ax,
    )

    ax.set_ylabel("Scenario")
    ax.set_xlabel("Step Index")
    ax.set_yticks(np.arange(len(entries)) + 0.5)
    ax.set_yticklabels([entry["label"] for entry in entries], rotation=0)

    tick_step = choose_tick_step(max_step)
    ticks = np.arange(0, max_step, tick_step)
    ax.set_xticks(ticks + 0.5)
    ax.set_xticklabels([str(t + 1) for t in ticks], rotation=0)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate an aggregate stability heatmap across scenarios.")
    parser.add_argument("--log_dir", default="logs/terminal_bench", help="Directory containing tb_monitor logs.")
    parser.add_argument("--log_files", nargs="*", help="Specific log files to include.")
    parser.add_argument("--scenario_ids", nargs="*", help="Only include these scenario IDs.")
    parser.add_argument("--exclude_scenarios", nargs="*", help="Exclude these scenario IDs.")
    parser.add_argument("--all_runs", action="store_true", help="Include all runs, not just the latest per scenario.")
    parser.add_argument("--metric", default="scr", help="Metric column to plot (default: scr).")
    parser.add_argument("--no_probe_only", action="store_true", help="Do not filter SCR to probe-only events.")
    parser.add_argument("--max_steps", type=int, help="Cap the x-axis at this many steps.")
    parser.add_argument("--pretty_labels", action="store_true", help="Pretty-print scenario names.")
    parser.add_argument("--scenario_order", nargs="*", help="Order scenarios by this list of scenario IDs.")
    parser.add_argument("--output", default="data/results/stability_heatmap.png", help="Output image path.")
    parser.add_argument("--title", default=None, help="Chart title (optional).")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for the color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for the color scale.")
    args = parser.parse_args()

    selected_logs = select_logs(
        log_dir=args.log_dir,
        log_files=args.log_files,
        scenario_ids=args.scenario_ids,
        exclude_scenarios=args.exclude_scenarios,
        include_all_runs=args.all_runs,
    )

    if not selected_logs:
        print("No log files found to plot.")
        return

    entries = []
    include_run_id = args.all_runs
    for entry in selected_logs:
        label = format_label(entry, args.pretty_labels, include_run_id)
        entries.append({**entry, "label": label})

    if args.scenario_order:
        order_map = {scenario_id: idx for idx, scenario_id in enumerate(args.scenario_order)}
        entries.sort(key=lambda item: (order_map.get(item["scenario_id"], len(order_map)), item["scenario_id"], item["mtime"]))
    else:
        entries.sort(key=lambda item: (item["scenario_id"], item["mtime"]))

    matrix, max_step = build_heatmap_matrix(
        entries=entries,
        metric=args.metric,
        probe_only=not args.no_probe_only,
        max_steps=args.max_steps,
    )

    if matrix is None:
        print("No data available after filtering.")
        return

    metric_name = METRIC_LABELS.get(args.metric, args.metric)
    title = args.title or f"Stability Heatmap ({metric_name})"
    plot_heatmap(
        matrix=matrix,
        entries=entries,
        max_step=max_step,
        metric=args.metric,
        output_path=args.output,
        title=title,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
