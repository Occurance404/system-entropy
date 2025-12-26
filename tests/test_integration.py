import unittest
from unittest.mock import MagicMock
from src.orchestrator.engine import Orchestrator
from src.scenarios.definitions import SCENARIOS

class ScriptedAgent:
    def __init__(self, golden_path):
        self.golden_path = golden_path
        self.step_idx = 0
        self.model_name = "ScriptedAgent"

    def get_next_action(self, history):
        if self.step_idx < len(self.golden_path):
            # Access Pydantic model fields directly, converting to dict for copy()
            step_data = self.golden_path[self.step_idx]
            action = step_data.agent_action.model_dump().copy()
            
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
        agent = ScriptedAgent(scenario.golden_path)
        metric_service = MagicMock()
        metric_service.calculate_entropy.return_value = 0.5
        metric_service.calculate_scr.return_value = 0.1
        metric_service.calculate_rdi.return_value = 0.0
        metric_service.calculate_compression_ratio.return_value = 0.5
        metric_service.calculate_ige.side_effect = lambda h_pre, h_post, cost: ((h_pre - h_post) / cost) if cost else 0.0
        
        # Mock connector
        mock_connector = MagicMock()
        mock_connector.execute_command.return_value = (0, "Mock output")
        mock_connector.read_file.return_value = "Mock content"
        mock_connector.write_file.return_value = True
        
        # Mock metrics_monitor with a log_step method
        mock_metrics_monitor = MagicMock()
        mock_metrics_monitor.log_step.return_value = None # log_step typically doesn't return anything
        
        # Initialize Orchestrator with injected connector and mock metrics monitor
        orchestrator = Orchestrator(
            scenario_id=scenario.id, 
            agent=agent, 
            metric_service=metric_service,
            metrics_monitor=mock_metrics_monitor,
            connector=mock_connector
        )
        
        print("\nRunning Integration Test Steps:")
        # Run a few steps corresponding to the golden path
        for i in range(len(scenario.golden_path)):
            result = orchestrator.step()
            print(f"Step {i+1}: {result['event_type']}")
            
            # Basic Assertions
            self.assertIn("current_entropy", result)
            self.assertIn("step_index", result)
            
            # Check IGE calculation (should be present after first tool use)
            if i > 0 and scenario.golden_path[i-1].agent_action.type == "tool_use":
                pass
                
        self.assertTrue(orchestrator.step_count >= len(scenario.golden_path))

if __name__ == "__main__":
    unittest.main()
