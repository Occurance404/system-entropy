import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def load_log(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_comparison(shock_log, baseline_log, output_path="data/results/entropy_comparison.png"):
    df_shock = load_log(shock_log)
    df_baseline = load_log(baseline_log)

    # Normalize/Fill for plotting
    df_shock['current_entropy'] = df_shock['current_entropy'].fillna(method='ffill').fillna(method='bfill')
    df_baseline['current_entropy'] = df_baseline['current_entropy'].fillna(method='ffill').fillna(method='bfill')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Shock
    ax.plot(df_shock['step_index'], df_shock['current_entropy'], 
            marker='o', linestyle='-', color='red', linewidth=2, label='Shock Scenario (Dynamic Constraints)')
    
    # Plot Baseline
    ax.plot(df_baseline['step_index'], df_baseline['current_entropy'], 
            marker='x', linestyle='--', color='blue', linewidth=2, label='Baseline (Linear Task)')

    # Highlight Perturbation Points (Steps 4 and 7 in Shock)
    perturbations = df_shock[df_shock['event_type'] == 'perturbation_triggered']
    for _, row in perturbations.iterrows():
        ax.axvline(x=row['step_index'], color='red', linestyle=':', alpha=0.5)
        ax.text(row['step_index'], ax.get_ylim()[1]*0.9, f"Shock\nSCR: {row['scr']:.2f}", 
                color='red', ha='center', fontweight='bold')

    ax.set_title('Entropic Dynamics: Baseline vs. Shock Scenario', fontsize=14)
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Agent Entropy (Uncertainty)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shock", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    
    plot_comparison(args.shock, args.baseline)
