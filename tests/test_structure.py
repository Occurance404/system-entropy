import pytest
import math
import numpy as np
from src.orchestrator.engine import Orchestrator
from src.agent.wrapper import VLLMAgent
from src.monitor.probe import StateMonitor
from src.scenarios.definitions import SCENARIOS

@pytest.fixture
def mock_monitor():
    monitor = StateMonitor()
    # Mock some methods if needed for orchestrator tests
    monitor.calculate_entropy_from_logprobs = lambda logprobs: 0.5 # Constant mock entropy
    monitor.calculate_semantic_collapse_ratio = lambda embeddings: 0.1 # Constant mock SCR
    monitor.calculate_information_gain_efficiency = lambda h_pre, h_post, cost: 0.05 # Constant mock IGE
    return monitor

@pytest.fixture
def mock_agent():
    agent = VLLMAgent(model_name="MockAgent")
    # Mock the agent's behavior for controlled testing
    agent.get_next_action = lambda history: {"type": "llm_reply", "content": "mock reply", "logprobs": [math.log(0.5), math.log(0.5)]}
    agent.generate_multiple = lambda history, n: [{"type": "llm_reply", "content": f"branch_{i}", "logprobs": [math.log(0.5)]} for i in range(n)]
    return agent

def test_orchestrator_init_and_scenario_load(mock_agent, mock_monitor):
    # Test loading a known scenario
    orch = Orchestrator(scenario_id="drug_filter_baseline", agent=mock_agent, monitor=mock_monitor)
    assert orch.scenario["id"] == "drug_filter_baseline"
    assert orch.step_count == 0
    assert orch.agent == mock_agent
    assert orch.monitor == mock_monitor

    # Test loading a non-existent scenario
    with pytest.raises(ValueError, match="Scenario with ID 'non_existent' not found."):
        Orchestrator(scenario_id="non_existent", agent=mock_agent, monitor=mock_monitor)

def test_orchestrator_step_no_perturbation(mock_agent, mock_monitor):
    orch = Orchestrator(scenario_id="drug_filter_baseline", agent=mock_agent, monitor=mock_monitor)
    result = orch.step()
    assert result["type"] == "llm_reply" # Assuming mock agent always returns llm_reply

def test_orchestrator_perturbation_trigger_and_probe(mock_agent, mock_monitor):
    # Temporarily modify scenario to trigger perturbation at step 1
    original_perturbations = SCENARIOS[1]["perturbations"]
    SCENARIOS[1]["perturbations"] = [{"step": 1, "type": "test", "instruction": "Test perturbation"}]

    orch = Orchestrator(scenario_id="drug_filter_shock", agent=mock_agent, monitor=mock_monitor)
    orch.step_count = 0 # Ensure perturbation triggers at step 1
    result = orch.step()
    
    assert result["type"] == "perturbation_triggered"
    assert result["perturbation"] == "Test perturbation"
    assert "probe_metrics" in result
    assert "scr" in result["probe_metrics"]
    
    # Restore original perturbations
    SCENARIOS[1]["perturbations"] = original_perturbations

def test_orchestrator_hysteresis_intervention(mock_agent, mock_monitor, capsys):
    # Set high entropy threshold and ensure mock agent provides high entropy
    mock_monitor.calculate_entropy_from_logprobs = lambda logprobs: 1.0 # High entropy
    orch = Orchestrator(scenario_id="drug_filter_baseline", agent=mock_agent, monitor=mock_monitor)
    orch.entropy_threshold = 0.5
    orch.panic_threshold = 3

    # Step 1: Panic counter = 1
    orch.step() 
    assert orch.panic_counter == 1

    # Step 2: Panic counter = 2
    orch.step()
    assert orch.panic_counter == 2
    
    # Step 3: Panic counter = 3, trigger intervention
    result = orch.step()
    assert result["type"] == "intervention"
    assert "persistent_panic" in result["reason"]
    
    captured = capsys.readouterr()
    assert "Intervention Triggered" in captured.out

def test_vllmagent_get_next_action(mock_agent):
    action = mock_agent.get_next_action([])
    assert action["type"] == "llm_reply"
    assert "logprobs" in action

def test_vllmagent_generate_multiple(mock_agent):
    branches = mock_agent.generate_multiple([], n=3)
    assert len(branches) == 3
    assert all("branch_" in b["content"] for b in branches)
    assert all("logprobs" in b for b in branches)
