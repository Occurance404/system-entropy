import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import dotenv_values 

from src.orchestrator.engine import Orchestrator
from src.services.metrics import EmbeddingMetricService
from src.agent.real_agent import OpenAICompatibleAgent

def run_rescue_experiment(scenario_id: str, max_steps: int, enable_rescue: bool):
    # Load environment variables
    config = dotenv_values(".env")
    
    api_key = config.get("VLLM_API_KEY")
    base_url = config.get("VLLM_BASE_URL")
    primary_model = config.get("VLLM_MODEL_NAME", "deepseek-chat") 
    rescue_model = config.get("RESCUE_MODEL_NAME") 
    
    if not api_key:
        print("ERROR: VLLM_API_KEY not set.")
        sys.exit(1)
        
    if not rescue_model:
        print("WARNING: RESCUE_MODEL_NAME not set. Using 'gpt-4' as placeholder default.")
        rescue_model = "gpt-4" # Placeholder default

    print(f"--- Starting RESCUE Experiment: {scenario_id} ---")
    print(f"Primary Model: {primary_model}")
    print(f"Rescue Protocol Enabled: {enable_rescue}")
    if enable_rescue:
        print(f"Rescue Model: {rescue_model}")
    
    # 1. Initialize Components
    metric_service = EmbeddingMetricService()
    
    try:
        print("Initializing Primary Agent...")
        primary_agent = OpenAICompatibleAgent(
            model_name=primary_model,
            base_url=base_url,
            api_key=api_key
        )
        
        if enable_rescue:
            print("Initializing Rescue Agent...")
            rescue_agent = OpenAICompatibleAgent(
                model_name=rescue_model,
                base_url=config.get("RESCUE_BASE_URL", base_url),
                api_key=config.get("RESCUE_API_KEY", api_key)
            )
    except Exception as e:
        print(f"Failed to initialize Agents: {e}")
        sys.exit(1)
        
    try:
        orchestrator = Orchestrator(scenario_id=scenario_id, agent=primary_agent, metric_service=metric_service)
    except ValueError as e:
        print(f"Error initializing Orchestrator: {e}")
        sys.exit(1)
    
    # 2. Setup Logging
    log_dir = "data/logs_rescue"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rescue_tag = "rescued" if enable_rescue else "baseline"
    log_file = f"{log_dir}/sim_{rescue_tag}_{scenario_id}_{timestamp}.jsonl"
    
    print(f"Logging to: {log_file}")
    
    rescued = False
    
    with open(log_file, "w") as f:
        # 3. Simulation Loop
        for i in range(max_steps):
            print(f"\n[Step {i+1}] Executing...")
            
            try:
                step_result_dict = orchestrator.step()
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "step_index": step_result_dict.get("step_index", i + 1),
                    "orchestrator_state": {
                        "panic_counter": orchestrator.panic_counter,
                        "current_agent": orchestrator.agent.model_name,
                        "rescue_enabled": enable_rescue
                    },
                    "event_type": step_result_dict.get("event_type", "unknown"),
                    "metrics": step_result_dict
                }
                
                f.write(json.dumps(log_entry) + "\n")
                f.flush()
                
                print(f"  Event Type: {log_entry['event_type']}")
                
                if log_entry['event_type'] == 'perturbation_triggered':
                    print(f"  >>> PERTURBATION DETECTED! SCR: {log_entry['metrics'].get('scr', 'N/A')}")

                if log_entry['event_type'] == 'intervention':
                    print(f"  !!! PANIC DETECTED !!! Reason: {log_entry['metrics'].get('reason')}")
                    
                    if enable_rescue:
                        if not rescued:
                            print(f"  >>> INITIATING RESCUE PROTOCOL: Switching to {rescue_model} <<<")
                            orchestrator.switch_agent(rescue_agent)
                            rescued = True
                        else:
                            print("  >>> Rescue already attempted. Simulation failing despite rescue.")
                            break
                    else:
                        print("  >>> Rescue Disabled. Continuing with Primary Agent to observe collapse.")

            except Exception as e:
                print(f"CRITICAL ERROR at Step {i+1}: {e}")
                break

    print(f"\n--- Experiment Complete. Check {log_file} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Entropic Stress-Test Rescue Experiment.")
    parser.add_argument("--scenario_id", type=str, default="drug_filter_shock", help="ID of the scenario")
    parser.add_argument("--max_steps", type=int, default=15, help="Max steps")
    parser.add_argument("--enable_rescue", action="store_true", help="Enable the Rescue Agent protocol.")
    
    args = parser.parse_args()
    
    run_rescue_experiment(scenario_id=args.scenario_id, max_steps=args.max_steps, enable_rescue=args.enable_rescue)
