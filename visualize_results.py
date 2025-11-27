import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_latest_log(log_dir="data/logs_rescue"):
    # Find the latest log file
    list_of_files = glob.glob(f'{log_dir}/*.jsonl')
    if not list_of_files:
        print("No log files found.")
        return None
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
    return pd.DataFrame(data)

def plot_metrics(df, output_path="data/results/experiment_summary.png"):
    if df is None or df.empty:
        print("No data to plot.")
        return

    # Ensure all relevant columns exist, fill missing with NaN for plotting clarity
    df['current_entropy'] = df['current_entropy'].fillna(method='ffill').fillna(method='bfill') # Fill forward/backward for continuous plot
    df['scr'] = df['scr'].fillna(0) # SCR only exists at perturbation, fill others with 0
    df['ige'] = df['ige'].fillna(0) # IGE only exists at tool_execution, fill others with 0
    df['cbf'] = df['cbf'].fillna(0) # CBF only exists at code writing, fill others with 0
    df['rdi'] = df['rdi'].fillna(0) # RDI only exists at code execution, fill others with 0

    steps = df['step_index']
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True) # 6 subplots now
    
    # Plot 1: Current Entropy
    axes[0].plot(steps, df['current_entropy'], marker='o', linestyle='-', color='blue', label='Current Entropy')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Agent Internal State Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Panic Counter
    axes[1].plot(steps, [s['panic_counter'] for s in df['orchestrator_state']], marker='o', linestyle='-', color='orange', label='Panic Counter')
    axes[1].set_ylabel('Panic Level')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Semantic Collapse Ratio (SCR)
    scr_data = df[df['event_type'] == 'perturbation_triggered']
    if not scr_data.empty:
        axes[2].bar(scr_data['step_index'], scr_data['scr'], color='red', width=0.5, label='SCR (Collapse)')
        for i, row in scr_data.iterrows():
            axes[2].text(row['step_index'], row['scr'] + 0.01, f"{row['scr']:.2f}", ha='center')
    axes[2].set_ylabel('SCR Score')
    axes[2].set_title('Semantic Collapse Ratio (at Perturbations)')
    axes[2].set_ylim(0, 1.0) # SCR is 0-1
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Plot 4: Information Gain Efficiency (IGE)
    ige_data = df[df['event_type'] == 'tool_execution']
    if not ige_data.empty:
        colors = ['green' if v > 0 else 'red' for v in ige_data['ige']]
        axes[3].bar(ige_data['step_index'], ige_data['ige'], color=colors, width=0.5, label='IGE')
    axes[3].set_ylabel('IGE (Info Gain)')
    axes[3].set_title('Information Gain Efficiency (Tool Usage)')
    axes[3].axhline(0, color='black', linewidth=0.8)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Plot 5: Cyclomatic Bloat Factor (CBF) & Regressive Debt Index (RDI)
    cbf_data = df[df['cbf'] > 0] # Only plot if CBF was calculated
    if not cbf_data.empty:
        axes[4].bar(cbf_data['step_index'], cbf_data['cbf'], color='purple', width=0.4, label='CBF (Code Bloat)', align='center')
        for i, row in cbf_data.iterrows():
            axes[4].text(row['step_index'], row['cbf'] + 0.1, f"{int(row['cbf'])}", ha='center')

    rdi_data = df[df['rdi'] > 0] # Only plot if RDI was calculated
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
    # Ensure compression_ratio column exists
    if 'compression_ratio' in df.columns:
        cr_data = df[df['compression_ratio'].notna()]
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
    parser.add_argument("--log_file", type=str, help="Specific log file to visualize. If not provided, the latest will be loaded.")
    args = parser.parse_args()

    if args.log_file:
        data = []
        with open(args.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Flatten metrics if present
                if "metrics" in entry and isinstance(entry["metrics"], dict):
                    entry.update(entry["metrics"])
                data.append(entry)
        df = pd.DataFrame(data)
    else:
        df = load_latest_log()
        
    plot_metrics(df)
