import json
import os
import uuid
import argparse
from datetime import datetime
from src.orchestrator.engine import Orchestrator
from src.monitor.probe import StateMonitor
from src.monitor.terminal_bench_monitor import get_monitor
from src.agent.mock_agent import ScriptedAgent

def run_simulation(scenario_id: str = "drug_filter_shock", max_steps: int = 10):
    print(f"--- Starting Simulation: {scenario_id} ---")
    
    # 1. Initialize Components
    monitor = StateMonitor()
    agent = ScriptedAgent(model_name="Simulated-Fail-Bot")
    tb_monitor = get_monitor() # Unified Logger
    
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    
    # 2. Initialize Orchestrator
    orchestrator = Orchestrator(
        scenario_id=scenario_id, 
        agent=agent, 
        monitor=monitor,
        run_id=run_id,
        metrics_monitor=tb_monitor
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
            scr = step_result.get('scr', 0.0)
            print(f"  >>> PERTURBATION DETECTED! Triggering Branching Probe.")
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

    # 4. Compute and Save Drift Summary
    print("\n--- Computing Drift Metrics ---")
    summary = orchestrator.compute_drift_summary()
    summary_file = f"data/logs_terminal_bench/run_{run_id}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    print(f"Max Drift: {summary['max_drift']:.4f}")
    print(f"Recovered at Step: {summary['recovered_at_step']}")

    print(f"\n--- Simulation Complete. Logs in data/logs_terminal_bench/ ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="drug_filter_shock", help="Scenario ID")
    parser.add_argument("--steps", type=int, default=10, help="Max steps")
    args = parser.parse_args()
    
    run_simulation(scenario_id=args.scenario, max_steps=args.steps)
