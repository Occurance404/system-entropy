import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import dotenv_values 

from src.orchestrator.engine import Orchestrator
from src.services.metrics import EmbeddingMetricService
from src.agent.real_agent import OpenAICompatibleAgent
# Fallback to mock if needed, but user requested SOTA
from src.agent.mock_agent import ScriptedAgent 

def run_hard_mode_experiment(scenario_id: str, max_steps: int, probe_interval: int, model_name: str = None):
    # Load environment variables
    config = dotenv_values(".env")
    
    api_key = config.get("VLLM_API_KEY") or os.getenv("VLLM_API_KEY")
    base_url = config.get("VLLM_BASE_URL") or os.getenv("VLLM_BASE_URL")
    
    # Default to user provided model or env var or deepseek-chat
    primary_model = model_name or config.get("VLLM_MODEL_NAME", "deepseek-chat") 
    
    print(f"--- Starting HARD MODE Experiment: {scenario_id} ---")
    print(f"Model: {primary_model}")
    print(f"Intervention/Rescue: DISABLED")
    print(f"Silent Probe Interval: Every {probe_interval} steps")
    
    # 1. Initialize Components
    metric_service = EmbeddingMetricService()
    
    agent = None
    if api_key:
        try:
            print("Initializing OpenAICompatibleAgent...")
            agent = OpenAICompatibleAgent(
                model_name=primary_model,
                base_url=base_url,
                api_key=api_key
            )
        except Exception as e:
            print(f"Failed to initialize Real Agent: {e}")
    
    if not agent:
        print("WARNING: No API Key found or Agent init failed. Falling back to ScriptedAgent (Mock).")
        agent = ScriptedAgent(model_name="mock-agent")

    try:
        # Initialize Orchestrator with intervention DISABLED and periodic probing ENABLED
        orchestrator = Orchestrator(
            scenario_id=scenario_id, 
            agent=agent, 
            metric_service=metric_service,
            enable_intervention=False, # Strict "No Rescue" rule
            probe_interval=probe_interval
        )
    except ValueError as e:
        print(f"Error initializing Orchestrator: {e}")
        sys.exit(1)
    
    # 2. Setup Logging
    log_dir = "data/logs_hard_mode"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize primary_model name for use in file path
    sanitized_model_name = primary_model.replace("/", "_").replace(":", "_")
    log_file = f"{log_dir}/{scenario_id}_{sanitized_model_name}_{timestamp}.jsonl"
    
    print(f"Logging to: {log_file}")
    
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
                        "intervention_enabled": False
                    },
                    "event_type": step_result_dict.get("event_type", "unknown"),
                    "metrics": step_result_dict
                }
                
                f.write(json.dumps(log_entry) + "\n")
                f.flush()
                
                event = log_entry['event_type']
                print(f"  Event Type: {event}")
                
                if event == 'periodic_probe':
                     print(f"  (Silent Probe) SCR: {log_entry['metrics'].get('scr', 'N/A')}")
                elif event == 'panic_detected':
                     print(f"  !!! PANIC DETECTED (Ignored) !!! Entropy: {log_entry['metrics'].get('current_entropy')}")

            except Exception as e:
                print(f"CRITICAL ERROR at Step {i+1}: {e}")
                import traceback
                traceback.print_exc()
                break

    print(f"\n--- Hard Mode Experiment Complete. Check {log_file} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hard Mode (No Rescue) Experiment.")
    parser.add_argument("--scenario_id", type=str, required=True, help="ID of the scenario (e.g., hard_coding_challenge)")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps to allow")
    parser.add_argument("--probe_interval", type=int, default=3, help="Steps between silent SCR probes")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    
    args = parser.parse_args()
    
    run_hard_mode_experiment(
        scenario_id=args.scenario_id, 
        max_steps=args.max_steps, 
        probe_interval=args.probe_interval,
        model_name=args.model
    )
