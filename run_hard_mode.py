import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import dotenv_values 

from src.orchestrator.engine import Orchestrator
from src.services.metrics import EmbeddingMetricService
from src.agent.real_agent import OpenAICompatibleAgent
from src.monitor.terminal_bench_monitor import TerminalBenchMonitor
from src.agent.mock_agent import ScriptedAgent 

def run_hard_mode_experiment(
    scenario_id: str,
    max_steps: int,
    probe_interval: int,
    model_name: str = None,
    autonomous: bool = False,
    cheap: bool = False,
):
    # Load environment variables
    config = dotenv_values(".env")
    
    api_key = config.get("VLLM_API_KEY") or os.getenv("VLLM_API_KEY")
    base_url = config.get("VLLM_BASE_URL") or os.getenv("VLLM_BASE_URL")
    
    # Default to user provided model or env var or deepseek-chat
    primary_model = model_name or config.get("VLLM_MODEL_NAME", "deepseek-chat") 
    
    print(f"--- Starting HARD MODE Experiment: {scenario_id} ---")
    print(f"Model: {primary_model}")
    print(f"Intervention/Rescue: DISABLED")
    if cheap:
        print("Mode: CHEAP (Probes disabled, validation enabled)")
        probe_interval = 0
    print(f"Silent Probe Interval: Every {probe_interval} steps")
    if autonomous:
        print("Mode: AUTONOMOUS (Will stop on 'Task Complete')")
    
    # 1. Initialize Components
    metric_service = EmbeddingMetricService()
    
    # Initialize Monitor (Logs to data/logs_terminal_bench/)
    metrics_monitor = TerminalBenchMonitor()
    print(f"Logging via Monitor to: {metrics_monitor.log_file}")
    
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
            metrics_monitor=metrics_monitor, # Pass the monitor here
            enable_intervention=False, # Strict "No Rescue" rule
            probe_interval=probe_interval,
            enable_validation=cheap,
            stop_on_success=cheap,
            enable_branching_probes=not cheap,
        )
    except ValueError as e:
        print(f"Error initializing Orchestrator: {e}")
        sys.exit(1)
    
    # 2. Simulation Loop
    stop_phrases = ["task is complete", "final summary", "mission accomplished", "completing the task"]
    
    for i in range(max_steps):
        print(f"\n[Step {i+1}] Executing...")
        
        try:
            step_result_dict = orchestrator.step()
            
            # Console Feedback
            event = step_result_dict.get('event_type', 'unknown')
            print(f"  Event Type: {event}")
            
            if event == 'periodic_probe':
                    print(f"  (Silent Probe) SCR: {step_result_dict.get('scr', 'N/A')}")
            elif event == 'panic_detected':
                    print(f"  !!! PANIC DETECTED (Ignored) !!! Entropy: {step_result_dict.get('current_entropy')}")
            
            # Autonomous Stopping Logic
            if autonomous and event == 'llm_reply':
                content = step_result_dict.get('content', '')
                if isinstance(content, str):
                    content_lower = content.lower()
                    if any(phrase in content_lower for phrase in stop_phrases):
                        print("\n[AUTONOMOUS STOP] Agent signaled task completion.")
                        print(f"Reason: Found stop phrase in: '{content[:100]}...'")
                        break
            if step_result_dict.get("task_complete") or event == "task_complete":
                print("\n[TASK COMPLETE] Validator signaled success.")
                break

        except Exception as e:
            print(f"CRITICAL ERROR at Step {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"\n--- Hard Mode Experiment Complete. Check {metrics_monitor.log_file} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hard Mode (No Rescue) Experiment.")
    parser.add_argument("--scenario_id", type=str, required=True, help="ID of the scenario (e.g., hard_coding_challenge)")
    parser.add_argument("--max_steps", type=int, default=50, help="Max safety limit. Set to 500+ for open-ended tasks.")
    parser.add_argument("--probe_interval", type=int, default=5, help="Steps between silent SCR probes")
    parser.add_argument("--model", type=str, default=None, help="Model name to use")
    parser.add_argument("--autonomous", action="store_true", help="Stop automatically when agent says 'Task is complete'.")
    parser.add_argument("--cheap", action="store_true", help="Disable probes and stop on validator success.")
    
    args = parser.parse_args()
    
    run_hard_mode_experiment(
        scenario_id=args.scenario_id, 
        max_steps=args.max_steps, 
        probe_interval=args.probe_interval,
        model_name=args.model,
        autonomous=args.autonomous,
        cheap=args.cheap,
    )
