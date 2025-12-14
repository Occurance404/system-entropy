import os
import subprocess
import time
import json
import glob
from datetime import datetime

def run_scale_experiment(num_runs=5, output_csv="data/results/scale_experiment_results_v2.csv"):
    print(f"--- Starting Scale Experiment (Slow Mode): {num_runs} runs ---")
    
    # 1. Setup Results File
    os.makedirs("data/results", exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("run_id,step,entropy,scr,is_shocked\n")
        
    for run_i in range(num_runs):
        print(f"\n[Run {run_i+1}/{num_runs}] Launching Experiment...")
        
        # 2. Run the Shock Experiment
        # We use subprocess to run the bash script. 
        try:
            # Construct a single bash command to activate venv and run the script
            # using run_rescue_experiment.py with rescue DISABLED (raw data)
            bash_command = (
                f"source .venv/bin/activate && "
                f"python run_rescue_experiment.py --scenario_id file_organizer_shock --max_steps 8" 
            )
            subprocess.run(["/bin/bash", "-c", bash_command], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run {run_i+1} failed: {e}")
            # Don't continue, try to sleep and recover
            
        # 3. Harvest Data from the Monitor Log
        # The runner writes to data/logs_rescue/sim_baseline_file_organizer_shock_YYYYMMDD_HHMMSS.jsonl
        list_of_files = glob.glob('data/logs_rescue/*.jsonl')
        if not list_of_files:
            print("Warning: No monitor logs found.")
            continue
            
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Harvesting data from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            step_count = 0
            for line in f:
                try:
                    entry = json.loads(line)
                    # Extract metrics from the nested dictionary
                    metrics = entry.get("metrics", {})
                    
                    step_index = entry.get("step_index", 0)
                    entropy = metrics.get("current_entropy")
                    scr = metrics.get("scr")
                    
                    # Handle Nones for CSV safety
                    if entropy is None: entropy = 0.0
                    if scr is None: scr = 0.0
                    
                    # Determine if this step was shocked (Scenario defines shock at Step 3)
                    is_shocked = step_index >= 3
                    
                    # Append to CSV
                    with open(output_csv, "a") as out_f:
                        out_f.write(f"{run_i+1},{step_index},{entropy},{scr},{is_shocked}\n")
                        
                except json.JSONDecodeError:
                    continue
        
        # 4. Rate Limit Cooldown
        print("Sleeping for 45s to reset API Rate Limits...")
        time.sleep(45)

    print(f"--- Scale Experiment Complete. Data saved to {output_csv} ---")

if __name__ == "__main__":
    run_scale_experiment()
