import abc
from typing import List, Dict, Any, Optional
from src.scenarios.definitions import SCENARIOS
from sentence_transformers import SentenceTransformer

class Orchestrator:
    """
    Module 1: The Orchestrator (The Controller) - v2.0 Refactor
    
    Manages the experiment lifecycle, specifically:
    1. State Machine (Linear Run vs. Branching Probe)
    2. External Tool Execution (Sandboxing)
    3. Perturbation Injection
    """
    
    def __init__(self, scenario_id: str, agent: Any, monitor: Any):
        self.scenario_id = scenario_id
        self.scenario = self._load_scenario(scenario_id)
        if not self.scenario:
            raise ValueError(f"Scenario with ID '{scenario_id}' not found.")
            
        self.agent = agent
        self.monitor = monitor
        self.step_count = 0
        self.history: List[Dict] = []
        self.panic_counter = 0
        self.panic_threshold = 3
        self.entropy_threshold = 0.8 
        
        # Initialize local embedding model for SCR calculation
        print("Loading embedding model for SCR metrics...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.") 
        
    def _load_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Loads a scripted scenario from definitions.py."""
        for s in SCENARIOS:
            if s["id"] == scenario_id:
                return s
        return None

    def step(self) -> Dict:
        """
        Advances the simulation by one step.
        Implements the State Machine:
        1. Check Perturbation (Trigger Probe?)
        2. Get Agent Action (Intent)
        3. Execute Tool (if needed) & Measure IGE
        """
        self.step_count += 1
        
        step_metrics = {
            "step_index": self.step_count,
            "current_entropy": None, # Will be filled after agent response
            "ige": None,
            "scr": None,
            "cbf": None,
            "rdi": None, # Placeholder for Regressive Debt Index
            "event_type": None
        }

        perturbation_instruction = self._check_perturbation_triggers()
        if perturbation_instruction:
            step_metrics["event_type"] = "perturbation_triggered"
            probe_results = self._run_branching_probe(perturbation_instruction)
            step_metrics["scr"] = probe_results["scr"]
            # Returning perturbation details and probe metrics
            return {**step_metrics, **{"perturbation": perturbation_instruction, "probe_metrics": probe_results}}
        
        # Get Agent's next action
        agent_action_intent = self.agent.get_next_action(self.history)
        
        current_entropy = self.monitor.calculate_entropy_from_logprobs(agent_action_intent.get("logprobs", []))
        step_metrics["current_entropy"] = current_entropy
        
        # Intervention Check (Hysteresis)
        if self._check_panic(current_entropy):
            step_metrics["event_type"] = "intervention"
            return {**step_metrics, **{"type": "intervention", "reason": "persistent_panic", "step": self.step_count}}

        # Handle Agent's Intent (Tool Use or Reply)
        if agent_action_intent["type"] == "tool_use":
            tool_name = agent_action_intent["tool"]
            tool_args = agent_action_intent["content"]

            tool_result, ige, cbf_value, rdi_value = self._execute_tool_and_measure(agent_action_intent, current_entropy)
            step_metrics["event_type"] = "tool_execution"
            step_metrics["ige"] = ige
            step_metrics["cbf"] = cbf_value
            step_metrics["rdi"] = rdi_value # From tool_execution if relevant
            
            # Update history with tool action and result
            self.history.append({"role": "user", "content": f"Used tool: {tool_name} with args: {tool_args}"})
            self.history.append({"role": "tool_output", "content": tool_result})
            
            return {**step_metrics, **{"type": "tool_execution", "tool": tool_name, "result": tool_result}}
            
        elif agent_action_intent["type"] == "llm_reply":
            step_metrics["event_type"] = "llm_reply"
            self.history.append({"role": "user", "content": agent_action_intent["content"]})
            return {**step_metrics, **{"type": "llm_reply", "content": agent_action_intent["content"]}}
        
        step_metrics["event_type"] = "unknown_action"
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
        if branch_texts:
            # Using local embedding model initialized in __init__
            embeddings = self.embedding_model.encode(branch_texts)
            # Convert to list of lists for the monitor
            embeddings_list = [e.tolist() for e in embeddings]
        else:
            embeddings_list = []
            
        scr = self.monitor.calculate_semantic_collapse_ratio(embeddings_list)
        return {"scr": scr, "branches": branch_texts}

    def _execute_tool_and_measure(self, action_intent: Dict, h_pre: float) -> tuple:
        """
        Executes tool in sandbox, calculates IGE, CBF (if code), and RDI (placeholder).
        Returns: (tool_result, ige, cbf_value, rdi_value)
        """
        tool_name = action_intent["tool"]
        tool_args = action_intent["content"] 
        
        mock_tool_output = f"Mock output for {tool_name} with args {tool_args}. Result: Success!"
        cbf_value = None
        rdi_value = None # Placeholder for RDI

        if tool_name == "read_file":
            mock_tool_output = "File content: " + str(tool_args.get('path', ''))
        elif tool_name == "write_file":
            mock_tool_output = "File written successfully."
            # If writing a Python file, calculate CBF
            if isinstance(tool_args, dict) and 'path' in tool_args and tool_args['path'].endswith('.py'):
                code_content = tool_args.get('content', '')
                cbf_value = self.monitor.measure_cyclomatic_complexity(code_content)
        elif tool_name == "execute_python":
            mock_tool_output = "Python script executed. Mock output."
            # RDI check could happen here, or if agent generates/runs tests
            rdi_value = 0.0 # Mock RDI, needs actual test comparison
        elif tool_name == "search_web":
            mock_tool_output = "Search results for: " + str(tool_args.get('query', ''))
        
        # Simulate h_post from a subsequent agent thought after tool result
        mock_post_logprobs = [-0.2, -0.3] # Assume some logprobs for next thought
        h_post = self.monitor.calculate_entropy_from_logprobs(mock_post_logprobs)
        
        token_cost = 10 # Mock cost of the tool call
        ige = self.monitor.calculate_information_gain_efficiency(h_pre, h_post, token_cost)
        
        return mock_tool_output, ige, cbf_value, rdi_value

    def _check_panic(self, entropy: float) -> bool:
        """Updates panic counter and returns True if threshold exceeded."""
        if entropy > self.entropy_threshold:
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
        # Implement actual reset logic here
        pass
