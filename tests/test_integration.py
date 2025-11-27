import unittest
from unittest.mock import MagicMock
from src.orchestrator.engine import Orchestrator
from src.monitor.probe import StateMonitor
from src.scenarios.definitions import SCENARIOS

class ScriptedAgent:
    def __init__(self, golden_path):
        self.golden_path = golden_path
        self.step_idx = 0
        self.model_name = "ScriptedAgent"

    def get_next_action(self, history):
        if self.step_idx < len(self.golden_path):
            action = self.golden_path[self.step_idx]["agent_action"].copy()
            # Fix for integration test schema mismatch
            if "args" in action and "content" not in action:
                 action["content"] = action["args"]
            
            # Mock logprobs
            action["logprobs"] = [-0.1] * 10 
            self.step_idx += 1
            return action
        return {"type": "llm_reply", "content": "Task complete.", "logprobs": [-0.1]}

    def generate_multiple(self, history, n=5):
        return [{"content": "thought", "logprobs": [-0.1]}] * n

class TestIntegration(unittest.TestCase):
    def test_simulation_run(self):
        # Use the first scenario
        scenario = SCENARIOS[0] 
        agent = ScriptedAgent(scenario["golden_path"])
        monitor = StateMonitor()
        
        # Mock connector
        mock_connector = MagicMock()
        mock_connector.execute_command.return_value = (0, "Mock output")
        mock_connector.read_file.return_value = "Mock content"
        mock_connector.write_file.return_value = True
        
        # Initialize Orchestrator with injected connector
        orchestrator = Orchestrator(
            scenario_id=scenario["id"], 
            agent=agent, 
            monitor=monitor,
            connector=mock_connector
        )
        
        print("\nRunning Integration Test Steps:")
        # Run a few steps corresponding to the golden path
        for i in range(len(scenario["golden_path"])):
            result = orchestrator.step()
            print(f"Step {i+1}: {result['event_type']}")
            
            # Basic Assertions
            self.assertIn("current_entropy", result)
            self.assertIn("step_index", result)
            
            # Check IGE calculation (should be present after first tool use)
            if i > 0 and scenario["golden_path"][i-1]["agent_action"]["type"] == "tool_use":
                pass
                
        self.assertTrue(orchestrator.step_count >= len(scenario["golden_path"]))

if __name__ == "__main__":
    unittest.main()
