import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob

def load_monitor_log(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def load_latest_monitor_log(log_dir="data/logs_terminal_bench"):
    list_of_files = glob.glob(f'{log_dir}/tb_monitor_*.jsonl')
    if not list_of_files:
        print(f"No monitor log files found in {log_dir}.")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading monitor log file: {latest_file}")
    
    return load_monitor_log(latest_file)

def plot_swe_agent_metrics(df, output_path="data/results/mini_swe_agent_metrics.png"):
    if df is None or df.empty:
        print("No data to plot for mini-swe-agent.")
        return

    # Ensure all relevant columns exist, fill missing with NaN for plotting clarity
    df['entropy'] = df['entropy'].fillna(method='ffill').fillna(method='bfill')
    df['scr'] = df['scr'].fillna(0) # SCR only exists at perturbation, fill others with 0

    steps = range(1, len(df) + 1) # Use sequential steps since original step_index might not be continuous
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Entropy
    axes[0].plot(steps, df['entropy'], marker='o', linestyle='-', color='blue', label='Agent Entropy (per LLM Call)')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Mini-SWE-Agent: Entropy and SCR per LLM Call')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Semantic Collapse Ratio (SCR)
    # SCR is only calculated when branching probe is triggered (i.e., when scr is not 0)
    scr_data = df[df['scr'] > 0]
    if not scr_data.empty:
        axes[1].bar(scr_data.index + 1, scr_data['scr'], color='red', width=0.5, label='SCR (when triggered)')
        for i, row in scr_data.iterrows():
            axes[1].text(i + 1, row['scr'] + 0.01, f"{row['scr']:.2f}", ha='center', color='red')
    
    axes[1].set_ylabel('SCR Score')
    axes[1].set_xlabel('LLM Call Index')
    axes[1].set_ylim(0, df['scr'].max() * 1.2 if not df['scr'].empty else 1.0) # Adjust ylim for clarity
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Mini-SWE-Agent metrics plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize metrics from the LLM proxy monitor log for mini-swe-agent.")
    parser.add_argument("--log_file", type=str, help="Specific monitor log file to visualize. If not provided, the latest will be loaded.")
    args = parser.parse_args()

    if args.log_file:
        df = load_monitor_log(args.log_file)
    else:
        df = load_latest_monitor_log()
        
    plot_swe_agent_metrics(df)
