import json
import os
from datetime import datetime
from src.orchestrator.engine import Orchestrator
from src.monitor.probe import StateMonitor
from src.agent.mock_agent import ScriptedAgent

def run_simulation(scenario_id: str = "drug_filter_shock", max_steps: int = 10):
    print(f"--- Starting Simulation: {scenario_id} ---")
    
    # 1. Initialize Components
    monitor = StateMonitor()
    agent = ScriptedAgent(model_name="Simulated-Fail-Bot")
    orchestrator = Orchestrator(scenario_id=scenario_id, agent=agent, monitor=monitor)
    
    # 2. Setup Logging
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/sim_{scenario_id}_{timestamp}.jsonl"
    
    print(f"Logging to: {log_file}")
    
    with open(log_file, "w") as f:
        # 3. Simulation Loop
        for i in range(max_steps):
            print(f"\n[Step {i+1}] Executing...")
            
            # Run Orchestrator Step
            step_result = orchestrator.step()
            
            # Enrich Log
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step_index": i + 1,
                "orchestrator_state": {
                    "panic_counter": orchestrator.panic_counter,
                    "entropy_threshold": orchestrator.entropy_threshold
                },
                "result": step_result
            }
            
            # Write to file
            f.write(json.dumps(log_entry) + "\n")
            
            # Console Output
            print(f"  Event Type: {step_result['type']}")
            if step_result['type'] == 'perturbation_triggered':
                scr = step_result['probe_metrics']['scr']
                print(f"  >>> PERTURBATION DETECTED! Triggering Branching Probe.")
                print(f"  >>> Semantic Collapse Ratio (SCR): {scr:.4f}")
                if scr > 0.5:
                    print("  >>> WARN: High Cognitive Collapse detected!")
            
            if step_result['type'] == 'tool_execution':
                print(f"  Tool: {step_result['tool']}")
                print(f"  IGE (Info Gain): {step_result['ige']:.4f}")
            
            if step_result['type'] == 'intervention':
                print(f"  !!! INTERVENTION TRIGGERED !!! Reason: {step_result.get('reason')}")
                break # End simulation on intervention for this demo

    print(f"\n--- Simulation Complete. Check {log_file} for details ---")

if __name__ == "__main__":
    run_simulation()
