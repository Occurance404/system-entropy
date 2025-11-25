import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from dotenv import dotenv_values 

from src.orchestrator.engine import Orchestrator
from src.monitor.probe import StateMonitor
from src.agent.real_agent import OpenAICompatibleAgent # Ensure this is imported for real runs

def run_real_simulation(scenario_id: str, max_steps: int):
    # Load environment variables from .env file
    config = dotenv_values(".env")
    
    # Check for configuration
    api_key = config.get("VLLM_API_KEY")
    base_url = config.get("VLLM_BASE_URL")
    model_name = config.get("VLLM_MODEL_NAME", "deepseek-chat") 
    
    if not api_key:
        print("ERROR: VLLM_API_KEY not set in .env file or environment.")
        print("Please set VLLM_API_KEY to your OpenAI or vLLM API key.")
        sys.exit(1)
        
    if not base_url:
        print("WARNING: VLLM_BASE_URL not set in .env file or environment.")
        print("Defaulting to http://localhost:8000/v1 (common for local vLLM).")
        base_url = "http://localhost:8000/v1"
    
    print(f"--- Starting REAL Simulation: {scenario_id} ---")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    
    # 1. Initialize Components
    monitor = StateMonitor()
    try:
        agent = OpenAICompatibleAgent(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        )
    except Exception as e:
        print(f"Failed to initialize Real Agent: {e}")
        sys.exit(1)
        
    try:
        orchestrator = Orchestrator(scenario_id=scenario_id, agent=agent, monitor=monitor)
    except ValueError as e:
        print(f"Error initializing Orchestrator: {e}")
        sys.exit(1)
    
    # 2. Setup Logging
    log_dir = "data/logs_real"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/real_sim_{scenario_id}_{timestamp}.jsonl"
    
    print(f"Logging to: {log_file}")
    
    with open(log_file, "w") as f:
        # 3. Simulation Loop
        for i in range(max_steps):
            print(f"\n[Step {i+1}] Executing...")
            
            try:
                # Run Orchestrator Step
                step_result_dict = orchestrator.step() # Orchestrator now returns a dict of metrics
                
                # Enrich Log - log all metrics directly
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "step_index": step_result_dict.get("step_index", i + 1),
                    "orchestrator_state": {
                        "panic_counter": orchestrator.panic_counter,
                        "entropy_threshold": orchestrator.entropy_threshold
                    },
                    "event_type": step_result_dict.get("event_type", "unknown"),
                    "current_entropy": step_result_dict.get("current_entropy"),
                    "ige": step_result_dict.get("ige"),
                    "scr": step_result_dict.get("scr"),
                    "cbf": step_result_dict.get("cbf"),
                    "rdi": step_result_dict.get("rdi"),
                    "raw_result": step_result_dict # Keep original result for debugging
                }
                
                # Write to file
                f.write(json.dumps(log_entry) + "\n")
                f.flush() 
                
                # Console Output
                print(f"  Event Type: {log_entry['event_type']}")
                print(f"  Current Entropy: {log_entry['current_entropy']:.4f}" if log_entry['current_entropy'] is not None else "  Current Entropy: N/A")
                
                if log_entry['event_type'] == 'perturbation_triggered':
                    scr = log_entry['scr']
                    print(f"  >>> PERTURBATION DETECTED! Triggering Branching Probe.")
                    print(f"  >>> Semantic Collapse Ratio (SCR): {scr:.4f}")
                    print(f"  >>> Branches Generated: {len(log_entry['raw_result']['probe_metrics']['branches'])}")
                
                if log_entry['event_type'] == 'tool_execution':
                    print(f"  Tool: {log_entry['raw_result']['tool']}")
                    print(f"  IGE (Info Gain): {log_entry['ige']:.4f}" if log_entry['ige'] is not None else "  IGE (Info Gain): N/A")
                    print(f"  CBF (Code Bloat): {log_entry['cbf']}" if log_entry['cbf'] is not None else "  CBF (Code Bloat): N/A")
                    print(f"  RDI (Regressive Debt): {log_entry['rdi']}" if log_entry['rdi'] is not None else "  RDI (Regressive Debt): N/A")
                
                if log_entry['event_type'] == 'intervention':
                    print(f"  !!! INTERVENTION TRIGGERED !!! Reason: {log_entry['raw_result'].get('reason')}")
                    break 

            except Exception as e:
                print(f"CRITICAL ERROR at Step {i+1}: {e}")
                import traceback
                traceback.print_exc()
                break

    print(f"\n--- Simulation Complete. Check {log_file} for details ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Entropic Stress-Test Simulation.")
    parser.add_argument("--scenario_id", type=str, default="drug_filter_shock", help="ID of the scenario to run (defined in src/scenarios/definitions.py)")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to run the simulation")
    
    args = parser.parse_args()
    
    run_real_simulation(scenario_id=args.scenario_id, max_steps=args.max_steps)
