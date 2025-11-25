import abc
from typing import List, Dict, Any

class AgentWrapper(abc.ABC):
    """
    Module 2: The Agent Wrapper (The Subject) - v2.0 Refactor
    
    Abstracts the internal model (e.g., Qwen2.5, GPT-4o).
    Key Change: The Agent is a Text Generator, NOT an Executor.
    It outputs INTENT (Action), which the Orchestrator executes.
    """
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.context_window: List[Dict] = []
        
    @abc.abstractmethod
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Decides the next step based on history.
        Returns:
            {
                "type": "tool_use" | "reply",
                "content": str | Dict (tool args),
                "logprobs": List[float] # Critical for Entropy Metric
            }
        """
        pass

    @abc.abstractmethod
    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        """
        Generates N divergent responses (Branching Probe).
        Used by Orchestrator during Perturbation to measure Semantic Collapse.
        """
        pass

class VLLMAgent(AgentWrapper):
    """Implementation for local open-weights models via vLLM."""
    
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        # Placeholder: In real impl, this calls vllm.generate
        # and parses the output to detect tool usage.
        return {
            "type": "reply", 
            "content": "placeholder response", 
            "logprobs": [-0.1, -0.5] # Mock logprobs
        }

    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        # Placeholder for Branching Probe
        return [
            {
                "type": "reply", 
                "content": f"branch_{i}", 
                "logprobs": [-0.1 * i]
            } for i in range(n)
        ]
