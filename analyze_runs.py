import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

def load_logs(log_file: str) -> pd.DataFrame:
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Flatten step metrics if nested
                if "step_index" in entry: # Engine log style
                    data.append(entry)
                elif "response_obj" in entry: # Proxy log style
                     # Extract relevant proxy metrics
                     flat = {
                         "timestamp": entry.get("timestamp"),
                         "model": entry.get("model_name"),
                         "entropy": entry.get("entropy_from_logprobs"),
                     }
                     data.append(flat)
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(data)

def analyze_run(df: pd.DataFrame, output_dir: str):
    print(f"Analyzing {len(df)} steps...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Entropy over Time
    if "current_entropy" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["step_index"], df["current_entropy"], label="Entropy")
        plt.xlabel("Step")
        plt.ylabel("Entropy")
        plt.title("Agent Entropy over Time")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "entropy_time.png"))
        print(f"Saved entropy_time.png to {output_dir}")
        
        print(f"Avg Entropy: {df['current_entropy'].mean():.4f}")
        print(f"Max Entropy: {df['current_entropy'].max():.4f}")

    # 2. IGE Analysis
    if "ige" in df.columns:
        # Filter out None
        ige_data = df[df["ige"].notnull()]
        if not ige_data.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(ige_data["step_index"], ige_data["ige"])
            plt.xlabel("Step")
            plt.ylabel("IGE")
            plt.title("Information Gain Efficiency per Tool Use")
            plt.savefig(os.path.join(output_dir, "ige_bars.png"))
            print(f"Saved ige_bars.png to {output_dir}")
            print(f"Avg IGE: {ige_data['ige'].mean():.4f}")

    # 3. Panic Analysis
    if "panic_counter" in df.columns:
        max_panic = df["panic_counter"].max()
        print(f"Max Panic Level Reached: {max_panic}")
        
    # 4. Event Types
    if "event_type" in df.columns:
        print("\nEvent Distribution:")
        print(df["event_type"].value_counts())

    # 5. RDI Analysis
    if "rdi" in df.columns:
        rdi_data = df[df["rdi"].notnull()]
        if not rdi_data.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(rdi_data["step_index"], rdi_data["rdi"], label="Regressive Debt (RDI)", color="red")
            plt.xlabel("Step")
            plt.ylabel("RDI")
            plt.title("Regressive Debt Index over Time")
            plt.legend()
            plt.savefig(os.path.join(output_dir, "rdi_time.png"))
            print(f"Saved rdi_time.png to {output_dir}")

    # 6. Load Drift Summary
    if "run_id" in df.columns:
        run_id = df["run_id"].iloc[0]
        summary_path = f"data/logs_terminal_bench/run_{run_id}_summary.json"
        if os.path.exists(summary_path):
            print("\n--- Drift Summary ---")
            with open(summary_path, "r") as f:
                summary = json.load(f)
                print(json.dumps(summary, indent=2))
        else:
            print(f"\n(No drift summary found at {summary_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation logs.")
    parser.add_argument("log_file", help="Path to JSONL log file")
    parser.add_argument("--output", default="analysis_results", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"File not found: {args.log_file}")
        exit(1)
        
    df = load_logs(args.log_file)
    if df.empty:
        print("No valid data found in log file.")
    else:
        analyze_run(df, args.output)
