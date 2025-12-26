import json
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import uuid
from datetime import datetime
from dotenv import dotenv_values 

from src.orchestrator.engine import Orchestrator
from src.monitor.terminal_bench_monitor import get_monitor
from src.agent.real_agent import OpenAICompatibleAgent 

def run_real_simulation(scenario_id: str, max_steps: int, cheap: bool = False):
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
    
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")

    # 1. Initialize Components
    tb_monitor = get_monitor() # Unified Logger
    
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
        orchestrator = Orchestrator(
            scenario_id=scenario_id, 
            agent=agent, 
            run_id=run_id,
            metrics_monitor=tb_monitor,
            enable_validation=cheap,
            stop_on_success=cheap,
            enable_branching_probes=not cheap,
        )
    except ValueError as e:
        print(f"Error initializing Orchestrator: {e}")
        sys.exit(1)
    
    # 3. Simulation Loop
    for i in range(max_steps):
        print(f"\n[Step {i+1}] Executing...")
        
        try:
            # Run Orchestrator Step (logs internally via tb_monitor)
            step_result_dict = orchestrator.step() 
            
            # Console Output
            event_type = step_result_dict.get("event_type", "unknown")
            print(f"  Event Type: {event_type}")
            print(f"  Current Entropy: {step_result_dict.get('current_entropy', 'N/A')}")
            
            if event_type == 'perturbation_triggered':
                scr = step_result_dict.get('scr')
                print(f"  >>> PERTURBATION DETECTED! Triggering Branching Probe.")
                if scr is None:
                    print("  >>> Semantic Collapse Ratio (SCR): N/A (embeddings unavailable)")
                else:
                    print(f"  >>> Semantic Collapse Ratio (SCR): {scr:.4f}")
            
            if event_type == 'tool_execution':
                print(f"  Tool: {step_result_dict.get('tool')}")
                print(f"  IGE (Info Gain): {step_result_dict.get('ige', 'N/A')}")
                print(f"  CBF (Code Bloat): {step_result_dict.get('cbf', 'N/A')}")
                print(f"  RDI (Regressive Debt): {step_result_dict.get('rdi', 'N/A')}")
            
            if event_type == 'intervention':
                print(f"  !!! INTERVENTION TRIGGERED !!!")
                break 
            if step_result_dict.get("task_complete") or event_type == "task_complete":
                print("\n[TASK COMPLETE] Validator signaled success.")
                break

        except Exception as e:
            print(f"CRITICAL ERROR at Step {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break

    # 4. Compute and Save Drift Summary
    print("\n--- Computing Drift Metrics ---")
    summary = orchestrator.compute_drift_summary()
    summary_file = os.path.join(orchestrator.run_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    print(f"Max Drift: {summary['max_drift']:.4f}")

    print(f"\n--- Simulation Complete. Logs in data/logs_terminal_bench/ ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Entropic Stress-Test Simulation.")
    parser.add_argument("--scenario_id", type=str, default="drug_filter_shock", help="ID of the scenario to run (defined in src/scenarios/definitions.py)")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to run the simulation")
    parser.add_argument("--cheap", action="store_true", help="Disable expensive probes and stop on validator success.")
    
    args = parser.parse_args()
    
    run_real_simulation(scenario_id=args.scenario_id, max_steps=args.max_steps, cheap=args.cheap)
