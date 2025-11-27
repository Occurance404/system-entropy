import uuid
import os
import abc
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.scenarios.definitions import SCENARIOS
from src.scenarios.setup_ops import SCENARIO_SETUP_MAP
from sentence_transformers import SentenceTransformer
from src.connectors.tb_connect import TerminalBenchConnector

class Orchestrator:
    """
    Module 1: The Orchestrator (The Controller) - v2.0 Refactor
    
    Manages the experiment lifecycle, specifically:
    1. State Machine (Linear Run vs. Branching Probe)
    2. External Tool Execution (Sandboxing)
    3. Perturbation Injection
    """
    
    def __init__(self, 
                 scenario_id: str, 
                 agent: Any, 
                 monitor: Any, 
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
        # Ensure the sandbox data directory is populated before starting the connector
        project_root = os.path.abspath(os.getcwd())
        self.sandbox_path = os.path.join(project_root, "data", f"sandbox_{scenario_id}")
        
        if scenario_id in SCENARIO_SETUP_MAP:
            print(f"Orchestrator: Running environment setup for {scenario_id}...")
            SCENARIO_SETUP_MAP[scenario_id](self.sandbox_path)
        else:
            print(f"Orchestrator: No specific setup found for {scenario_id}. Ensuring directory exists.")
            os.makedirs(self.sandbox_path, exist_ok=True)
        # -------------------------
            
        self.agent = agent
        self.monitor = monitor # This is the StateMonitor (calculator)
        self.metrics_monitor = metrics_monitor # This is the TerminalBenchMonitor (logger)
        self.run_id = run_id or str(uuid.uuid4())
        
        self.step_count = 0
        self.history: List[Dict] = []
        self.panic_counter = 0
        self.panic_threshold = 3
        self.entropy_threshold = 0.8 
        
        # Baseline stats for Z-score panic detection
        self.entropy_mean = entropy_mean
        self.entropy_std = entropy_std
        self.z_score_threshold = 2.0 
        
        # Context for IGE calculation
        self.last_tool_context: Optional[Dict] = None
        
        # Initialize local embedding model for SCR and RDI calculation
        print("Loading embedding model for SCR/RDI metrics...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("Embedding model loaded.") 
            
            # Calculate Ground Truth Embedding for RDI
            # Prefer specific ground truth goal, fallback to initial prompt
            truth_text = self.scenario.get("ground_truth_goal") or self.scenario.get("initial_prompt", "")
            self.ground_truth_embedding = self.embedding_model.encode(truth_text).tolist() if truth_text else None
            
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            self.embedding_model = None
            self.ground_truth_embedding = None
        
        # Initialize Sandbox Connector (Dependency Injection)
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
        self.recovery_threshold = 2 # Consecutive stable steps to mark recovery
        
    def switch_agent(self, new_agent: Any):
        """Swaps the current agent with a new one (e.g., for Model Handoff)."""
        print(f"Orchestrator: Switching agent from {self.agent.model_name if hasattr(self.agent, 'model_name') else 'Unknown'} to {new_agent.model_name if hasattr(new_agent, 'model_name') else 'Unknown'}.")
        self.agent = new_agent
        self.panic_counter = 0

    def _load_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Loads a scripted scenario from definitions.py."""
        for s in SCENARIOS:
            if s["id"] == scenario_id:
                return s
        return None

    def compute_drift_summary(self) -> Dict[str, Any]:
        """Computes aggregate drift metrics for the run."""
        valid_rdi = [x for x in self.rdi_series if x is not None]
        
        max_drift = max(valid_rdi) if valid_rdi else 0.0
        drift_auc = sum(valid_rdi) # Simple sum as AUC proxy for step-wise data
        
        post_recovery_mean = None
        if self.recovered_at_step is not None and self.recovered_at_step <= len(self.rdi_series):
            # Slicing 1-based step index to 0-based list index: step 1 is index 0
            # recovered_at_step is the step number (1-based)
            # indices: recovered_at_step - 1 to end
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
        Implements the State Machine:
        1. Check Perturbation (Trigger Probe?)
        2. Get Agent Action (Intent)
        3. Execute Tool (if needed) & Measure IGE (Post-Hoc)
        4. Calculate RDI
        5. Log via TerminalBenchMonitor
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
            # Reset Recovery State on Perturbation
            self.recovered_at_step = None
            self.stability_counter = 0
            
            step_metrics["event_type"] = "perturbation_triggered"
            probe_results = self._run_branching_probe(perturbation_instruction)
            step_metrics["scr"] = probe_results["scr"]
            
            # Append None to RDI series for this step (or keep previous?)
            # Logic says track per step. Perturbation step has no RDI? 
            # It interrupts the agent. Let's append None.
            self.rdi_series.append(None)
            
            # Log Perturbation
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
        
        current_entropy = self.monitor.calculate_entropy_from_logprobs(agent_action_intent.get("logprobs", []))
        step_metrics["current_entropy"] = current_entropy
        
        # Delayed IGE Calculation (from previous step's tool use)
        if self.last_tool_context:
            h_pre = self.last_tool_context['h_pre']
            token_cost = self.last_tool_context['token_cost']
            ige = self.monitor.calculate_information_gain_efficiency(h_pre, current_entropy, token_cost)
            step_metrics["ige"] = ige
            self.last_tool_context = None
            
        # Calculate RDI (Regressive Debt Index)
        current_content = agent_action_intent.get("content", "")
        # Ensure content is a string for embedding
        if not isinstance(current_content, str):
            current_content = str(current_content)
            
        if self.embedding_model and self.ground_truth_embedding and current_content.strip():
            try:
                current_plan_embedding = self.embedding_model.encode(current_content).tolist()
                rdi = self.monitor.check_goal_deviance(current_plan_embedding, self.ground_truth_embedding)
                step_metrics["rdi"] = rdi
            except Exception as e:
                print(f"Warning: RDI calculation failed: {e}")
                step_metrics["rdi"] = None
        
        # Track RDI
        self.rdi_series.append(step_metrics["rdi"])
        
        # Intervention Check (Hysteresis)
        if self._check_panic(current_entropy):
            step_metrics["event_type"] = "intervention"
            step_metrics["panic_counter"] = self.panic_counter # Update metric with new counter value
            
            # Reset Recovery State on Intervention
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
        
        # Update metric if panic check passed but didn't trigger intervention
        step_metrics["panic_counter"] = self.panic_counter
        
        # Recovery Detection
        # If we are stable (panic_counter == 0), we count up.
        # Note: _check_panic sets panic_counter=0 if entropy <= threshold
        if self.panic_counter == 0:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            
        if self.stability_counter >= self.recovery_threshold and self.recovered_at_step is None:
             # Mark recovery
             self.recovered_at_step = self.step_count

        # Handle Agent's Intent (Tool Use or Reply)
        tool_result = None
        if agent_action_intent["type"] == "tool_use":
            tool_name = agent_action_intent["tool"]
            tool_args = agent_action_intent["content"]
            step_metrics["tool"] = tool_name
            
            # Store context for NEXT step's IGE calculation
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
            # rdi is already calculated from intent
            
            # Update history
            self.history.append({"role": "user", "content": f"Used tool: {tool_name} with args: {tool_args}"})
            self.history.append({"role": "tool_output", "content": tool_result})
            
        elif agent_action_intent["type"] == "llm_reply":
            step_metrics["event_type"] = "llm_reply"
            
            # Calculate Compression Ratio
            cr = self.monitor.calculate_compression_ratio(agent_action_intent["content"])
            step_metrics["compression_ratio"] = cr
            
            self.history.append({"role": "user", "content": agent_action_intent["content"]})
        else:
             step_metrics["event_type"] = "unknown_action"

        # Final Logging via Metrics Monitor
        if self.metrics_monitor:
            # Provide branching function for potential SCR calculation
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
                scr=step_metrics["scr"], # Might be None here, calculated inside if branching triggered
                cbf=step_metrics["cbf"],
                rdi=step_metrics["rdi"],
                panic_counter=self.panic_counter,
                tool=step_metrics["tool"],
                compression_ratio=step_metrics["compression_ratio"],
                branching_func=branching_wrapper # Monitor can decide to call this
            )

        if tool_result:
             return {**step_metrics, **{"type": "tool_execution", "tool": tool_name, "result": tool_result}}
        elif step_metrics["event_type"] == "llm_reply":
             return {**step_metrics, **{"type": "llm_reply", "content": agent_action_intent["content"]}}
        else:
             return {**step_metrics, **{"type": "unknown_action"}}

    def _run_branching_probe(self, perturbation_instruction: str) -> Dict:
        """
        Executes the Branching Probe (Top-N generation) to measure Semantic Collapse.
        Uses REAL embeddings to calculate SCR.
        """
        # Inject perturbation into history for the branching probe
        probe_history = self.history + [{"role": "user", "content": perturbation_instruction}]
        branches = self.agent.generate_multiple(probe_history, n=5)
        
        # Extract text content from branches
        branch_texts = [b.get("content", "") for b in branches]
        
        # Generate REAL embeddings
        if branch_texts and self.embedding_model:
            embeddings = self.embedding_model.encode(branch_texts)
            embeddings_list = [e.tolist() for e in embeddings]
            scr = self.monitor.calculate_semantic_collapse_ratio(embeddings_list)
        else:
            scr = 0.0
            
        return {"scr": scr, "branches": branch_texts}

    def _execute_tool_and_measure(self, action_intent: Dict) -> tuple:
        """
        Executes tool in sandbox, calculates CBF (if code).
        Returns: (tool_result, cbf_value, rdi_value_placeholder)
        """
        tool_name = action_intent["tool"]
        tool_args = action_intent["content"] 
        
        tool_result = ""
        cbf_value = None
        
        if tool_name == "read_file":
            path = tool_args.get('path', '')
            tool_result = self.connector.read_file(path)
            
        elif tool_name == "write_file":
            path = tool_args.get('path', '')
            content = tool_args.get('content', '')
            success = self.connector.write_file(path, content)
            tool_result = "File written successfully." if success else "Failed to write file."
            
            if path.endswith('.py'):
                cbf_value = self.monitor.measure_cyclomatic_complexity(content)
                
        elif tool_name == "execute_python":
            script_path = tool_args.get('script_path', '')
            exit_code, output = self.connector.execute_command(f"python3 {script_path}")
            tool_result = f"Exit Code: {exit_code}\nOutput:\n{output}"
            
        elif tool_name == "search_web":
            tool_result = "Search is currently disabled in the sandbox environment."
        
        elif tool_name == "run_shell":
            command = tool_args.get('command', '')
            exit_code, output = self.connector.execute_command(command)
            tool_result = f"Exit Code: {exit_code}\nOutput:\n{output}"
        
        else:
            tool_result = f"Unknown tool: {tool_name}"
        
        return tool_result, cbf_value, None

    def _check_panic(self, entropy: float) -> bool:
        """Updates panic counter and returns True if threshold exceeded."""
        triggered = False
        
        if self.entropy_mean is not None and self.entropy_std is not None and self.entropy_std > 0:
            # Z-score approach
            z_score = (entropy - self.entropy_mean) / self.entropy_std
            triggered = z_score > self.z_score_threshold
        else:
            # Absolute threshold approach
            triggered = entropy > self.entropy_threshold
            
        if triggered:
            self.panic_counter += 1
        else:
            self.panic_counter = 0
            
        return self.panic_counter >= self.panic_threshold

    def _check_perturbation_triggers(self) -> Optional[str]:
        """Monitors step count and returns perturbation instruction if triggered."""
        for p in self.scenario.get("perturbations", []):
            if p["step"] == self.step_count:
                return p["instruction"]
        return None

    def intervene(self):
        """Resets Agent to last valid state if looping is detected."""
        print(f"Intervention Triggered at step {self.step_count} due to persistent panic!")
        self.panic_counter = 0
        self.history.append({"role": "system", "content": "Intervention: You seem stuck. Please reassess your goal and try a different approach."})
