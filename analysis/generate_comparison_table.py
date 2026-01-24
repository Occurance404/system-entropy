import os
import json
import glob
import pandas as pd
import numpy as np

def analyze_logs(log_dir="logs/terminal_bench"):
    print(f"Scanning {log_dir}...")
    log_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    
    summary_data = []
    
    for log_file in log_files:
        try:
            filename = os.path.basename(log_file)
            scenario_id = "unknown"
            steps = 0
            entropies = []
            scrs = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        scenario_id = entry.get("scenario_id", scenario_id)
                        steps = max(steps, entry.get("step_index", 0))
                        
                        # Extract metrics
                        ent = entry.get("current_entropy")
                        if ent is not None: entropies.append(ent)
                        
                        scr = entry.get("scr")
                        if scr is not None and scr > 0: scrs.append(scr)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Filter out empty or broken runs
            if steps < 5: continue
            
            summary_data.append({
                "Log File": filename,
                "Scenario": scenario_id,
                "Total Steps": steps,
                "Avg Entropy": np.mean(entropies) if entropies else 0.0,
                "Max Entropy": np.max(entropies) if entropies else 0.0,
                "Avg SCR": np.mean(scrs) if scrs else 0.0,
                "Peak SCR": np.max(scrs) if scrs else 0.0
            })
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    df = pd.DataFrame(summary_data)
    
    # Sort by scenario and steps
    if not df.empty:
        df = df.sort_values(by=["Scenario", "Total Steps"], ascending=[True, False])
        
        output_path = "data/results/final_comparison.csv"
        df.to_csv(output_path, index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Comparison saved to: {output_path}")
        print("\nTop 5 Runs:")
        print(df[["Scenario", "Total Steps", "Avg Entropy", "Peak SCR"]].head(5).to_string(index=False))
    else:
        print("No valid logs found.")

if __name__ == "__main__":
    analyze_logs()
