import os
import subprocess
import time
import json
import glob
from datetime import datetime

def run_scale_experiment(num_runs=50, output_csv="data/results/scale_experiment_results.csv"):
    print(f"--- Starting Scale Experiment: {num_runs} runs ---")
    
    # 1. Setup Results File
    os.makedirs("data/results", exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("run_id,step,entropy,scr,is_shocked\n")
        
    for run_i in range(num_runs):
        print(f"\n[Run {run_i+1}/{num_runs}] Launching Experiment...")
        
        # 2. Run the Shock Experiment
        # We use subprocess to run the bash script. 
        # Ensure we capture stdout/stderr to avoid clutter, or let it print to monitor progress.
        try:
            # Construct a single bash command to activate venv and run the script
            # This is the most robust way to ensure the venv is active for the subprocess.
            bash_command = (
                f"source .venv/bin/activate && "
                f"./run_shock_experiment.sh"
            )
            subprocess.run(["/bin/bash", "-c", bash_command], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run {run_i+1} failed: {e}")
            continue
            
        # 3. Harvest Data from the Monitor Log
        # The monitor writes to data/logs_terminal_bench/tb_monitor_YYYYMMDD_HHMMSS.jsonl
        # We need to find the *latest* file created.
        list_of_files = glob.glob('data/logs_terminal_bench/*.jsonl')
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
                    # Extract metrics
                    entropy = entry.get("entropy", 0.0)
                    scr = entry.get("scr", 0.0)
                    if scr is None: scr = 0.0
                    
                    # Determine if this step was shocked
                    # We know shock hits at step 3 (from run_shock_experiment.sh)
                    step_count += 1
                    is_shocked = step_count >= 3
                    
                    # Append to CSV
                    with open(output_csv, "a") as out_f:
                        out_f.write(f"{run_i+1},{step_count},{entropy},{scr},{is_shocked}\n")
                        
                except json.JSONDecodeError:
                    continue
        
        # 4. Cleanup (Optional - maybe keep logs for deep dive?)
        # For now, we keep them.
        
        # Sleep briefly to allow ports to clear
        time.sleep(2)

    print(f"--- Scale Experiment Complete. Data saved to {output_csv} ---")

if __name__ == "__main__":
    run_scale_experiment()
