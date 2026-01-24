import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator

def load_latest_log(log_dir):
    # Find the latest log file
    list_of_files = glob.glob(f'{log_dir}/*.jsonl')
    if not list_of_files:
        print(f"No log files found in {log_dir}")
        return None, None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading log file: {latest_file}")
    
    data = []
    with open(latest_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Flatten metrics if present
            if "metrics" in entry and isinstance(entry["metrics"], dict):
                entry.update(entry["metrics"])
            data.append(entry)
    return pd.DataFrame(data), latest_file

def plot_metrics(df, output_path="data/results/experiment_summary.png"):
    if df is None or df.empty:
        print("No data to plot.")
        return

    df = df.copy()

    # Ensure all relevant columns exist; keep missing values as NaN/None (do not forward-fill).
    if 'step_index' not in df.columns:
        df['step_index'] = range(1, len(df) + 1)
    df['step_index'] = pd.to_numeric(df['step_index'], errors='coerce')
    df = df.sort_values('step_index')

    for col in ['current_entropy', 'scr', 'ige', 'cbf', 'rdi', 'compression_ratio']:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'event_type' not in df.columns:
        df['event_type'] = 'unknown_action'

    if 'panic_counter' not in df.columns:
        if 'orchestrator_state' in df.columns:
            df['panic_counter'] = df['orchestrator_state'].apply(
                lambda x: x.get('panic_counter', 0) if isinstance(x, dict) else 0
            )
        else:
            df['panic_counter'] = 0
    df['panic_counter'] = pd.to_numeric(df['panic_counter'], errors='coerce').fillna(0)

    event_series = df['event_type'].fillna('').astype(str)
    probe_mask = event_series.str.contains('probe', case=False) | event_series.isin(['perturbation_triggered', 'proxy_shock_injected'])
    main_events = df[~probe_mask]
    steps = main_events['step_index']

    shock_steps = df[event_series.isin(['perturbation_triggered', 'proxy_shock_injected'])]['step_index'].dropna().tolist()

    entropy_present = df['current_entropy'].notna() if 'current_entropy' in df.columns else pd.Series(False, index=df.index)
    non_probe = ~probe_mask
    entropy_coverage = float((entropy_present & non_probe).sum() / non_probe.sum()) if int(non_probe.sum()) > 0 else 0.0

    entropy_data = main_events[main_events['current_entropy'].notna()]
    panic_has_signal = main_events['panic_counter'].fillna(0).abs().max() > 0
    scr_events = df[event_series.isin(['perturbation_triggered', 'periodic_probe', 'proxy_probe', 'proxy_shock_injected'])]
    scr_data = scr_events[scr_events['scr'].notna()]
    ige_data = main_events[main_events['ige'].notna()]
    cbf_data = main_events[main_events['cbf'] > 0]
    rdi_data = main_events[main_events['rdi'] > 0]
    cr_data = main_events[main_events['compression_ratio'].notna()]

    panels = []
    if not entropy_data.empty:
        panels.append("entropy")
    if panic_has_signal or entropy_coverage > 0:
        panels.append("panic")
    if not scr_data.empty:
        panels.append("scr")
    if not ige_data.empty:
        panels.append("ige")
    if not cbf_data.empty or not rdi_data.empty:
        panels.append("quality")
    if not cr_data.empty:
        panels.append("compression")

    if not panels:
        print("No usable metrics to plot.")
        return

    fig_height = max(6, 2.6 * len(panels))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, fig_height), sharex=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    scenario_id = df['scenario_id'].dropna().iloc[0] if 'scenario_id' in df.columns and not df['scenario_id'].dropna().empty else 'unknown'
    coverage_label = f"Entropy coverage (non-probe): {entropy_coverage:.0%}"
    fig.suptitle(f"Run Summary - {scenario_id} | {coverage_label}", fontsize=12)

    def _shade_shocks(ax):
        for step in shock_steps:
            ax.axvline(step, color='red', alpha=0.15, linestyle='--', linewidth=1)

    def _set_axis_style(ax):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=9)
        _shade_shocks(ax)

    panel_idx = 0

    if "entropy" in panels:
        ax = axes[panel_idx]
        ax.plot(
            entropy_data['step_index'],
            entropy_data['current_entropy'],
            marker='o',
            linestyle='-',
            color='blue',
            label='Current Entropy',
            markersize=3,
        )
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy (Chosen-Token Surprisal Proxy)')
        ax.legend(fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    if "panic" in panels:
        ax = axes[panel_idx]
        ax.plot(steps, main_events['panic_counter'], marker='o', linestyle='-', color='orange', label='Panic Counter', markersize=3)
        ax.set_ylabel('Panic')
        ax.set_title('Panic Counter (Entropy/Loop-Based)')
        ax.legend(fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    if "scr" in panels:
        ax = axes[panel_idx]
        colors = ['red' if et in ('perturbation_triggered', 'proxy_shock_injected') else 'blue' for et in scr_data['event_type']]
        ax.bar(scr_data['step_index'], scr_data['scr'], color=colors, width=0.6)

        scr_max = float(scr_data['scr'].max()) if not scr_data.empty else 1.0
        ax.set_ylim(0, max(1.0, min(2.0, scr_max * 1.15)))
        ax.set_ylabel('SCR')
        ax.set_title('Semantic Collapse Ratio (Probes)')

        if not scr_data.empty:
            top_n = min(6, len(scr_data))
            top_rows = scr_data.nlargest(top_n, 'scr')
            for _, row in top_rows.iterrows():
                ax.text(row['step_index'], row['scr'] + 0.02, f"{row['scr']:.2f}", ha='center', fontsize=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='SCR (Perturbation/Shock)'),
            Patch(facecolor='blue', label='SCR (Periodic Probe)'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    if "ige" in panels:
        ax = axes[panel_idx]
        colors = ['green' if v > 0 else 'red' for v in ige_data['ige']]
        ax.bar(ige_data['step_index'], ige_data['ige'], color=colors, width=0.6, label='IGE')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('IGE')
        ax.set_title('Information Gain Efficiency')
        ax.legend(fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    if "quality" in panels:
        ax = axes[panel_idx]
        if not cbf_data.empty:
            ax.plot(cbf_data['step_index'], cbf_data['cbf'], color='purple', marker='s', linestyle='-', markersize=3, label='CBF')
        if not rdi_data.empty:
            ax.plot(rdi_data['step_index'], rdi_data['rdi'], color='brown', marker='o', linestyle='-', markersize=3, label='RDI')
        ax.set_ylabel('Score')
        ax.set_title('Code Quality Metrics (CBF / RDI)')
        ax.legend(fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    if "compression" in panels:
        ax = axes[panel_idx]
        ax.plot(cr_data['step_index'], cr_data['compression_ratio'], marker='s', linestyle='-', color='teal', label='Compression Ratio', markersize=3)
        ax.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Looping Threshold (<0.2)')
        ax.set_ylabel('Ratio')
        ax.set_title('Structural Health (Compression Ratio)')
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
        _set_axis_style(ax)
        panel_idx += 1

    axes[-1].set_xlabel('Simulation Step')
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results from the Entropic Stress-Test Simulation.")
    parser.add_argument("--log_file", type=str, help="Specific log file to visualize.")
    parser.add_argument("--log_dir", type=str, default="logs/rescue", help="Directory to search for latest log if file not specified.")
    args = parser.parse_args()

    if args.log_file:
        log_filename = args.log_file
        data = []
        with open(args.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if "metrics" in entry and isinstance(entry["metrics"], dict):
                    entry.update(entry["metrics"])
                data.append(entry)
        df = pd.DataFrame(data)
    else:
        df, log_filename = load_latest_log(args.log_dir)
    
    if df is not None:
        # Generate a unique output filename based on the log file
        base_name = os.path.basename(log_filename)
        # Remove extension
        base_name = os.path.splitext(base_name)[0]
        output_plot = f"data/results/summary_{base_name}.png"
            
        plot_metrics(df, output_path=output_plot)
