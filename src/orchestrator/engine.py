import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.scenarios.definitions import SCENARIOS
from src.scenarios.setup_ops import SCENARIO_SETUP_MAP
from src.connectors.tb_connect import TerminalBenchConnector
from src.services.metrics import EmbeddingMetricService
from src.interfaces import AgentProtocol, MetricServiceProtocol
from src.tools.registry import ToolRegistry

class Orchestrator:
    """
    Module 1: The Orchestrator (The Controller) - v2.2 Refactor
    
    Decoupled architecture using Protocols and Tool Registry.
    """
    
    def __init__(self, 
                 scenario_id: str, 
                 agent: AgentProtocol, 
                 metric_service: Optional[MetricServiceProtocol] = None, 
                 entropy_mean: float = None, 
                 entropy_std: float = None,
                 connector: Any = None,
                 run_id: str = None,
                 metrics_monitor: Any = None):
        
        self.scenario_id = scenario_id
        self.scenario = self._load_scenario(scenario_id)
        if not self.scenario:
            raise ValueError(f"Scenario with ID '{scenario_id}' not found.")
            
        # --- ENVIRONMENT SETUP ---
        project_root = os.path.abspath(os.getcwd())
        self.sandbox_path = os.path.join(project_root, "data", f"sandbox_{scenario_id}")
        
        if scenario_id in SCENARIO_SETUP_MAP:
            print(f"Orchestrator: Running environment setup for {scenario_id}...")
            SCENARIO_SETUP_MAP[scenario_id](self.sandbox_path)
        else:
            os.makedirs(self.sandbox_path, exist_ok=True)
            
        self.agent = agent
        self.run_id = run_id or str(uuid.uuid4())
        self.metrics_monitor = metrics_monitor
        
        # Dependency Injection: Metric Service
        if metric_service:
            self.metric_service = metric_service
        else:
            print("Orchestrator: No metric_service provided, instantiating default EmbeddingMetricService.")
            self.metric_service = EmbeddingMetricService()
            
        # Dependency Injection: Tool Registry
        self.tool_registry = ToolRegistry()
            
        self.step_count = 0
        self.history: List[Dict] = []
        self.panic_counter = 0
        self.panic_threshold = 3
        self.entropy_threshold = 0.8 
        
        self.entropy_mean = entropy_mean
        self.entropy_std = entropy_std
        self.z_score_threshold = 2.0 
        
        self.last_tool_context: Optional[Dict] = None
        
        # Initialize Sandbox Connector
        if connector:
            self.connector = connector
        else:
            print("Initializing TerminalBench Sandbox...")
            self.connector = TerminalBenchConnector(scenario_id)
            self.connector.start()
            
        # RDI Series Tracking
        self.rdi_series: List[Optional[float]] = []
        self.recovered_at_step: Optional[int] = None
        self.stability_counter = 0
        self.recovery_threshold = 2
        
        # Determine Ground Truth for RDI (Text only, embedding happens in service)
        self.ground_truth_text = self.scenario.get("ground_truth_goal") or self.scenario.get("initial_prompt", "")
        
    def switch_agent(self, new_agent: AgentProtocol):
        """Swaps the current agent."""
        print(f"Orchestrator: Switching agent from {getattr(self.agent, 'model_name', 'Unknown')} to {getattr(new_agent, 'model_name', 'Unknown')}.")
        self.agent = new_agent
        self.panic_counter = 0

    def _load_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        for s in SCENARIOS:
            if s["id"] == scenario_id:
                return s
        return None

    def compute_drift_summary(self) -> Dict[str, Any]:
        valid_rdi = [x for x in self.rdi_series if x is not None]
        max_drift = max(valid_rdi) if valid_rdi else 0.0
        drift_auc = sum(valid_rdi)
        
        post_recovery_mean = None
        if self.recovered_at_step is not None and self.recovered_at_step <= len(self.rdi_series):
            start_idx = self.recovered_at_step - 1
            post_series = [x for x in self.rdi_series[start_idx:] if x is not None]
            if post_series:
                post_recovery_mean = sum(post_series) / len(post_series)
                
        return {
            "max_drift": max_drift,
            "drift_auc": drift_auc,
            "recovered_at_step": self.recovered_at_step,
            "post_recovery_drift_mean": post_recovery_mean
        }

    def step(self) -> Dict:
        """
        Advances the simulation by one step.
        """
        self.step_count += 1
        
        step_metrics = {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "step_index": self.step_count,
            "model": getattr(self.agent, "model_name", "unknown"),
            "current_entropy": None, 
            "ige": None,
            "scr": None,
            "cbf": None,
            "rdi": None, 
            "compression_ratio": None,
            "event_type": None,
            "panic_counter": self.panic_counter,
            "tool": None,
            "timestamp": datetime.now().isoformat()
        }

        perturbation_instruction = self._check_perturbation_triggers()
        if perturbation_instruction:
            self.recovered_at_step = None
            self.stability_counter = 0
            
            step_metrics["event_type"] = "perturbation_triggered"
            
            # Probe using Service
            probe_results = self._run_branching_probe(perturbation_instruction)
            step_metrics["scr"] = probe_results["scr"]
            
            self.rdi_series.append(None)
            
            if self.metrics_monitor:
                self.metrics_monitor.log_step(
                    run_id=self.run_id,
                    scenario_id=self.scenario_id,
                    model_name=step_metrics["model"],
                    step_index=self.step_count,
                    event_type="perturbation_triggered",
                    prompt=perturbation_instruction,
                    scr=step_metrics["scr"],
                    panic_counter=self.panic_counter
                )

            return {**step_metrics, **{"type": "perturbation_triggered", "perturbation": perturbation_instruction, "probe_metrics": probe_results}}
        
        # Get Agent's next action
        agent_action_intent = self.agent.get_next_action(self.history)
        
        # Calculate Entropy via Service
        current_entropy = self.metric_service.calculate_entropy(agent_action_intent.get("logprobs", []))
        step_metrics["current_entropy"] = current_entropy
        
        # IGE Calculation
        if self.last_tool_context:
            h_pre = self.last_tool_context['h_pre']
            token_cost = self.last_tool_context['token_cost']
            ige = self.metric_service.calculate_ige(h_pre, current_entropy, token_cost)
            step_metrics["ige"] = ige
            self.last_tool_context = None
            
        # Calculate RDI via Service
        current_content = agent_action_intent.get("content", "")
        if not isinstance(current_content, str):
            current_content = str(current_content)
            
        step_metrics["rdi"] = self.metric_service.calculate_rdi(current_content, self.ground_truth_text)
        self.rdi_series.append(step_metrics["rdi"])
        
        # Intervention Check
        if self._check_panic(current_entropy):
            step_metrics["event_type"] = "intervention"
            step_metrics["panic_counter"] = self.panic_counter
            self.recovered_at_step = None
            self.stability_counter = 0
            
            if self.metrics_monitor:
                self.metrics_monitor.log_step(
                     run_id=self.run_id,
                     scenario_id=self.scenario_id,
                     model_name=step_metrics["model"],
                     step_index=self.step_count,
                     event_type="intervention",
                     current_entropy=current_entropy,
                     panic_counter=self.panic_counter,
                     rdi=step_metrics["rdi"]
                )
                
            self.intervene()
            return {**step_metrics, **{"type": "intervention", "reason": "persistent_panic", "step": self.step_count}}
        
        step_metrics["panic_counter"] = self.panic_counter
        
        if self.panic_counter == 0:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            
        if self.stability_counter >= self.recovery_threshold and self.recovered_at_step is None:
             self.recovered_at_step = self.step_count

        # Handle Agent's Intent
        tool_result = None
        if agent_action_intent["type"] == "tool_use":
            tool_name = agent_action_intent["tool"]
            tool_args = agent_action_intent["content"]
            step_metrics["tool"] = tool_name
            
            token_count = len(agent_action_intent.get("logprobs", []))
            if token_count == 0:
                token_count = len(str(tool_args)) // 4 + 1
            self.last_tool_context = {
                "h_pre": current_entropy,
                "token_cost": token_count
            }

            tool_result, cbf_value, _ = self._execute_tool_and_measure(agent_action_intent)
            step_metrics["event_type"] = "tool_execution"
            step_metrics["cbf"] = cbf_value
            
            self.history.append({"role": "user", "content": f"Used tool: {tool_name} with args: {tool_args}"})
            self.history.append({"role": "tool_output", "content": tool_result})
            
        elif agent_action_intent["type"] == "llm_reply":
            step_metrics["event_type"] = "llm_reply"
            cr = self.metric_service.calculate_compression_ratio(agent_action_intent["content"])
            step_metrics["compression_ratio"] = cr
            self.history.append({"role": "user", "content": agent_action_intent["content"]})
        else:
             step_metrics["event_type"] = "unknown_action"

        # Logging
        if self.metrics_monitor:
            def branching_wrapper():
                 branches = self.agent.generate_multiple(self.history, n=5)
                 return [b.get("content", "") for b in branches]

            self.metrics_monitor.log_step(
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                model_name=step_metrics["model"],
                step_index=self.step_count,
                event_type=step_metrics["event_type"],
                prompt=str(self.history[-1]["content"]) if self.history else "",
                current_entropy=step_metrics["current_entropy"],
                ige=step_metrics["ige"],
                scr=step_metrics["scr"], 
                cbf=step_metrics["cbf"],
                rdi=step_metrics["rdi"],
                panic_counter=self.panic_counter,
                tool=step_metrics["tool"],
                compression_ratio=step_metrics["compression_ratio"],
                branching_func=branching_wrapper 
            )

        if tool_result:
             return {**step_metrics, **{"type": "tool_execution", "tool": tool_name, "result": tool_result}}
        elif step_metrics["event_type"] == "llm_reply":
             return {**step_metrics, **{"type": "llm_reply", "content": agent_action_intent["content"]}}
        else:
             return {**step_metrics, **{"type": "unknown_action"}}

    def _run_branching_probe(self, perturbation_instruction: str) -> Dict:
        """
        Executes the Branching Probe via Agent and calculates SCR via Service.
        """
        probe_history = self.history + [{"role": "user", "content": perturbation_instruction}]
        branches = self.agent.generate_multiple(probe_history, n=5)
        branch_texts = [b.get("content", "") for b in branches]
        
        scr = self.metric_service.calculate_scr(branch_texts)
        return {"scr": scr, "branches": branch_texts}

    def _execute_tool_and_measure(self, action_intent: Dict) -> tuple:
        """
        Executes tool via ToolRegistry.
        """
        tool_name = action_intent["tool"]
        tool_args = action_intent["content"] 
        
        tool = self.tool_registry.get_tool(tool_name)
        
        if tool:
            result, cbf = tool.execute(tool_args, self.connector)
            return result, cbf, None
        else:
            return f"Unknown tool: {tool_name}", None, None

    def _check_panic(self, entropy: float) -> bool:
        triggered = False
        
        if self.entropy_mean is not None and self.entropy_std is not None and self.entropy_std > 0:
            z_score = (entropy - self.entropy_mean) / self.entropy_std
            triggered = z_score > self.z_score_threshold
        else:
            triggered = entropy > self.entropy_threshold
            
        if triggered:
            self.panic_counter += 1
        else:
            self.panic_counter = 0
            
        return self.panic_counter >= self.panic_threshold

    def _check_perturbation_triggers(self) -> Optional[str]:
        for p in self.scenario.get("perturbations", []):
            if p["step"] == self.step_count:
                return p["instruction"]
        return None

    def intervene(self):
        print(f"Intervention Triggered at step {self.step_count} due to persistent panic!")
        self.panic_counter = 0
        self.history.append({"role": "system", "content": "Intervention: You seem stuck. Please reassess your goal and try a different approach."})
