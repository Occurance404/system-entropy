import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

    # Main "step" events can share step_index with periodic_probe entries; exclude probes for time-series plots.
    main_events = df[~df['event_type'].isin(['periodic_probe'])]
    steps = main_events['step_index']
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True) # 6 subplots now
    
    # Plot 1: Current Entropy
    entropy_data = main_events[main_events['current_entropy'].notna()]
    axes[0].plot(
        entropy_data['step_index'],
        entropy_data['current_entropy'],
        marker='o',
        linestyle='-',
        color='blue',
        label='Current Entropy',
    )
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Agent Internal State Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Panic Counter
    axes[1].plot(steps, main_events['panic_counter'], marker='o', linestyle='-', color='orange', label='Panic Counter')
    axes[1].set_ylabel('Panic Level')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Semantic Collapse Ratio (SCR)
    probe_events = df[df['event_type'].isin(['perturbation_triggered', 'periodic_probe', 'proxy_probe', 'proxy_shock_injected'])]
    scr_data = probe_events[probe_events['scr'].notna()]
    if not scr_data.empty:
        colors = []
        for et in scr_data['event_type']:
            if et in ('perturbation_triggered', 'proxy_shock_injected'):
                colors.append('red')
            else:
                colors.append('blue')

        axes[2].bar(scr_data['step_index'], scr_data['scr'], color=colors, width=0.5)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='red', label='SCR (Perturbation/Shock)'),
            Patch(facecolor='blue', label='SCR (Periodic Probe)'),
        ]
        axes[2].legend(handles=legend_elements)

        for _, row in scr_data.iterrows():
            axes[2].text(row['step_index'], row['scr'] + 0.01, f"{row['scr']:.2f}", ha='center', fontsize=8)

    axes[2].set_ylabel('SCR Score')
    axes[2].set_title('Semantic Collapse Ratio (Perturbation vs. Periodic Probe)')
    if not scr_data.empty:
        scr_max = float(scr_data['scr'].max())
        axes[2].set_ylim(0, max(1.0, min(2.0, scr_max * 1.1)))
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Information Gain Efficiency (IGE)
    ige_data = main_events[main_events['ige'].notna()]
    if not ige_data.empty:
        colors = ['green' if v > 0 else 'red' for v in ige_data['ige']]
        axes[3].bar(ige_data['step_index'], ige_data['ige'], color=colors, width=0.5, label='IGE')
    axes[3].set_ylabel('IGE (Info Gain)')
    axes[3].set_title('Information Gain Efficiency (Entropy Delta / Cost)')
    axes[3].axhline(0, color='black', linewidth=0.8)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Plot 5: Cyclomatic Bloat Factor (CBF) & Regressive Debt Index (RDI)
    cbf_data = main_events[main_events['cbf'] > 0] # Only plot if CBF was calculated
    if not cbf_data.empty:
        axes[4].bar(cbf_data['step_index'], cbf_data['cbf'], color='purple', width=0.4, label='CBF (Code Bloat)', align='center')
        for i, row in cbf_data.iterrows():
            axes[4].text(row['step_index'], row['cbf'] + 0.1, f"{int(row['cbf'])}", ha='center')

    rdi_data = main_events[main_events['rdi'] > 0] # Only plot if RDI was calculated
    if not rdi_data.empty:
        # Offset slightly for RDI bars if CBF also exists at same step
        offset = -0.2 if not cbf_data.empty else 0
        axes[4].bar(rdi_data['step_index'] + offset, rdi_data['rdi'], color='brown', width=0.4, label='RDI (Regressive Debt)', align='center')
        for i, row in rdi_data.iterrows():
            axes[4].text(row['step_index'] + offset, row['rdi'] + 0.01, f"{row['rdi']:.2f}", ha='center')

    axes[4].set_ylabel('Complexity/Debt')
    axes[4].set_title('Code Quality Metrics')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    # Plot 6: Compression Ratio (Repetition Detector)
    cr_data = main_events[main_events['compression_ratio'].notna()]
    if not cr_data.empty:
        axes[5].plot(cr_data['step_index'], cr_data['compression_ratio'], marker='s', linestyle='-', color='teal', label='Compression Ratio')
        axes[5].axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Looping Threshold (<0.2)')
            
    axes[5].set_ylabel('Ratio (Compressed/Raw)')
    axes[5].set_title('Structural Health (Compression Ratio)')
    axes[5].set_xlabel('Simulation Step')
    axes[5].set_ylim(0, 1.2)
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results from the Entropic Stress-Test Simulation.")
    parser.add_argument("--log_file", type=str, help="Specific log file to visualize.")
    parser.add_argument("--log_dir", type=str, default="data/logs_rescue", help="Directory to search for latest log if file not specified.")
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
