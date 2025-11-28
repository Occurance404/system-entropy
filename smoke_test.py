import os
import json
import glob
from src.orchestrator.engine import Orchestrator
from src.agent.mock_agent import ScriptedAgent
from src.services.metrics import EmbeddingMetricService
from src.monitor.terminal_bench_monitor import get_monitor
import uuid

def run_smoke_test():
    print("--- Starting Smoke Test ---")
    
    # Clean logs
    log_dir = "data/logs_terminal_bench"
    if os.path.exists(log_dir):
        for f in glob.glob(f"{log_dir}/*.jsonl"):
            os.remove(f)
    
    scenario_id = "drug_filter_baseline"
    
    # Mock Agent & Service
    metric_service = EmbeddingMetricService()
    agent = ScriptedAgent(model_name="SmokeTestAgent")
    tb_monitor = get_monitor()
    run_id = str(uuid.uuid4())
    
    orchestrator = Orchestrator(
        scenario_id=scenario_id,
        agent=agent,
        metric_service=metric_service,
        run_id=run_id,
        metrics_monitor=tb_monitor
    )
    
    # Run 2 steps
    orchestrator.step()
    orchestrator.step()
    
    print("--- Simulation Steps Complete ---")
    
    # Verify Log
    log_files = glob.glob(f"{log_dir}/*.jsonl")
    if not log_files:
        print("FAIL: No log file created.")
        exit(1)
        
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Verifying log file: {latest_log}")
    
    required_keys = ["run_id", "scenario_id", "model", "step_index", "event_type", "current_entropy", "panic_counter"]
    
    with open(latest_log, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            print(f"FAIL: Expected at least 2 log entries, found {len(lines)}")
            exit(1)
            
        for line in lines:
            entry = json.loads(line)
            for k in required_keys:
                if k not in entry:
                    print(f"FAIL: Missing key '{k}' in log entry: {entry}")
                    exit(1)
            
            if entry["run_id"] != run_id:
                print(f"FAIL: run_id mismatch. Expected {run_id}, got {entry['run_id']}")
                exit(1)

    print("PASS: Smoke Test passed successfully.")

if __name__ == "__main__":
    run_smoke_test()
