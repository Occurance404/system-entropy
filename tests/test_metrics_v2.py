import pytest
import math
import numpy as np
from src.monitor.probe import StateMonitor

@pytest.fixture
def monitor():
    return StateMonitor()

def test_entropy_calculation(monitor):
    """
    Tests the new Normalized Top-K Entropy calculation.
    """
    # Case 1: Certainty
    # Distribution: [1.0] -> Logprob: [0.0]
    # H = -1.0 * ln(1.0) = 0.0
    logprobs_certain = [[0.0]] 
    assert monitor.calculate_entropy(logprobs_certain) == 0.0
    
    # Case 2: Coin Flip (Maximum Entropy for 2 choices)
    # Distribution: [0.5, 0.5] -> Logprobs: [ln(0.5), ln(0.5)]
    # H = -(0.5 * ln(0.5) + 0.5 * ln(0.5)) = 0.693...
    lp = math.log(0.5)
    logprobs_uncertain = [[lp, lp]]
    entropy = monitor.calculate_entropy(logprobs_uncertain)
    assert 0.69 < entropy < 0.70

    # Case 3: Sequence of varying entropy
    # Token 1: Certain (0.0)
    # Token 2: Uncertain (0.693)
    # Avg Entropy = 0.3465
    logprobs_seq = [[0.0], [lp, lp]]
    entropy_seq = monitor.calculate_entropy(logprobs_seq)
    assert 0.34 < entropy_seq < 0.35

def test_sanitize_code_block(monitor):
    raw = "Here is the code:\n```python\ndef foo(): pass\n```\nHope it helps."
    clean = monitor._sanitize_code_block(raw)
    assert clean == "def foo(): pass"
    
    raw_no_lang = "```\nprint('hi')\n```"
    clean_no_lang = monitor._sanitize_code_block(raw_no_lang)
    assert clean_no_lang == "print('hi')"

def test_ige_calculation(monitor):
    # H_pre = 0.8, H_post = 0.2, Cost = 10
    # IGE = (0.8 - 0.2) / 10 = 0.06
    ige = monitor.calculate_information_gain_efficiency(0.8, 0.2, 10)
    assert ige == pytest.approx(0.06)
    
    # Negative IGE (Confusion increased)
    # H_pre = 0.2, H_post = 0.8
    # IGE = (0.2 - 0.8) / 10 = -0.06
    ige_neg = monitor.calculate_information_gain_efficiency(0.2, 0.8, 10)
    assert ige_neg == pytest.approx(-0.06)

def test_semantic_collapse_ratio(monitor):
    # Identical vectors (Distance 0)
    embeddings_perfect = [[1.0, 0.0], [1.0, 0.0]]
    scr = monitor.calculate_semantic_collapse_ratio(embeddings_perfect)
    assert scr == 0.0
    
    # Orthogonal vectors (Distance 1.0)
    embeddings_diff = [[1.0, 0.0], [0.0, 1.0]]
    scr_diff = monitor.calculate_semantic_collapse_ratio(embeddings_diff)
    assert scr_diff == pytest.approx(1.0)