import math
import random
from typing import List, Dict, Any
from src.agent.wrapper import AgentWrapper

class ScriptedAgent(AgentWrapper):
    """
    A mock agent that follows a script but reacts to perturbations
    by simulating 'confusion' (High Entropy) to test the framework's metrics.
    """
    def __init__(self, model_name: str = "Scripted-Sim-7B"):
        super().__init__(model_name)
        self.step = 0
        
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        self.step += 1
        
        # --- SIMULATION LOGIC ---
        
        # Step 1-3: Normal Operation (Low Entropy, High Confidence)
        if self.step <= 3:
            return {
                "type": "tool_use",
                "tool": "read_file" if self.step == 1 else "write_file",
                "content": {"path": "script.py", "content": "print('hello')"},
                # Mock Distribution: Top-1 is 95%, others negligible.
                "logprobs": [[math.log(0.95), math.log(0.02), math.log(0.02), math.log(0.01)]]
            }
            
        # Step 4: The Perturbation hits (The Orchestrator injects it).
        # The Agent sees the change and gets "confused".
        if self.step == 4:
            # Simulating a "Thought" about the perturbation with higher uncertainty
            return {
                "type": "llm_reply",
                "content": "Wait, the requirements changed. I need to check the API docs...",
                # Mock Distribution: Flat (33% each) -> High Entropy
                "logprobs": [[math.log(0.33), math.log(0.33), math.log(0.33)]] * 5 # 5 tokens
            }

        # Step 5: "Thrashing" (Using tools but high entropy - Bad IGE)
        if self.step == 5:
            return {
                "type": "tool_use",
                "tool": "search_web",
                "content": {"query": "molecular mass api"},
                # Mock Distribution: Flat
                "logprobs": [[math.log(0.25), math.log(0.25), math.log(0.25), math.log(0.25)]]
            }

        # Step 6-8: "Panic" (Entropy stays high triggers Intervention)
        if self.step >= 6:
            return {
                "type": "llm_reply",
                "content": "I am not sure what to do. The API is confusing.",
                # Mock Distribution: Very Flat
                "logprobs": [[math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.2)]] * 3
            }
            
        return {"type": "llm_reply", "content": "...", "logprobs": []}

    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        """
        Simulates Branching Probe.
        If called during normal times, branches are similar.
        If called during perturbation, branches are divergent (Collapse).
        """
        # We infer context from self.step (simplified)
        is_crisis = self.step >= 4
        
        branches = []
        for i in range(n):
            if is_crisis:
                # DIVERGENT thoughts (High Semantic Collapse)
                content = f"Choice {i}: {['Delete everything', 'Sleep', 'Try API', 'Cry', 'Reboot'][i]}"
            else:
                # COHERENT thoughts (Low Semantic Collapse)
                content = f"Choice {i}: I should write the code in slightly different way {i}"
            
            branches.append({
                "type": "thought",
                "content": content,
                "logprobs": [math.log(0.5)]
            })
            
        return branches
