import json
import os
import sys
import uuid
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.orchestrator.engine import Orchestrator
from src.monitor.terminal_bench_monitor import get_monitor
from src.agent.mock_agent import ScriptedAgent
from src.shared.constants import LOGS_DIR

def run_simulation(scenario_id: str = "drug_filter_shock", max_steps: int = 10, cheap: bool = False):
    print(f"--- Starting Simulation: {scenario_id} ---")
    
    # 1. Initialize Components
    agent = ScriptedAgent(model_name="Simulated-Fail-Bot")
    tb_monitor = get_monitor() # Unified Logger
    
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    
    # 2. Initialize Orchestrator
    orchestrator = Orchestrator(
        scenario_id=scenario_id, 
        agent=agent, 
        run_id=run_id,
        metrics_monitor=tb_monitor,
        enable_validation=cheap,
        stop_on_success=cheap,
        enable_branching_probes=not cheap,
    )
    
    # 3. Simulation Loop
    for i in range(max_steps):
        print(f"\n[Step {i+1}] Executing...")
        
        # Run Orchestrator Step (logs internally via tb_monitor)
        step_result = orchestrator.step()
        
        # Console Output
        event_type = step_result.get('event_type', step_result.get('type', 'unknown'))
        print(f"  Event Type: {event_type}")
        
        if event_type == 'perturbation_triggered':
            scr = step_result.get('scr')
            print(f"  >>> PERTURBATION DETECTED! Triggering Branching Probe.")
            if scr is None:
                print("  >>> Semantic Collapse Ratio (SCR): N/A (embeddings unavailable)")
            else:
                print(f"  >>> Semantic Collapse Ratio (SCR): {scr:.4f}")
                if scr > 0.5:
                    print("  >>> WARN: High Cognitive Collapse detected!")
        
        if event_type == 'tool_execution':
            print(f"  Tool: {step_result.get('tool')}")
            print(f"  IGE (Info Gain): {step_result.get('ige')}")
            print(f"  RDI (Regressive Debt): {step_result.get('rdi')}")
        
        if event_type == 'intervention':
            print(f"  !!! INTERVENTION TRIGGERED !!!")
            break 
        if step_result.get("task_complete") or event_type == "task_complete":
            print("\n[TASK COMPLETE] Validator signaled success.")
            break

    # 4. Compute and Save Drift Summary
    print("\n--- Computing Drift Metrics ---")
    summary = orchestrator.compute_drift_summary()
    summary_file = os.path.join(orchestrator.run_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    print(f"Max Drift: {summary['max_drift']:.4f}")
    print(f"Recovered at Step: {summary['recovered_at_step']}")

    print(f"\n--- Simulation Complete. Logs in {LOGS_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="drug_filter_shock", help="Scenario ID")
    parser.add_argument("--steps", type=int, default=10, help="Max steps")
    parser.add_argument("--cheap", action="store_true", help="Disable expensive probes and stop on validator success.")
    args = parser.parse_args()
    
    run_simulation(scenario_id=args.scenario, max_steps=args.steps, cheap=args.cheap)
