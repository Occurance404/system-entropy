import os
import json
import math
from datetime import datetime
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

from src.shared.constants import LOG_SCHEMA

class TerminalBenchMonitor:
    def __init__(self):
        self.log_file = f"data/logs_terminal_bench/tb_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Load embedding model for SCR (Semantic Collapse Ratio)
        try:
            print("Initializing Monitor: Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Monitor: Embedding model loaded.")
        except Exception as e:
            print(f"Monitor Error: Failed to load embedding model: {e}")
            self.embedding_model = None

    def calculate_entropy(self, token_distributions):
        """
        Calculates Shannon Entropy from a list of Top-K token distributions.
        Fixes Scientific Validity: Uses sum(-p * log(p)) over the normalized Top-K distribution.
        Distinguishes between 'Certain' (Peaked) and 'Confused' (Flat) states.
        """
        if not token_distributions:
            return 0.0
        
        total_entropy = 0.0
        count = 0
        
        for logprobs in token_distributions:
            if not logprobs: continue
            
            # 1. Convert logprobs to probabilities
            # Handle potential -inf or very small numbers
            probs = []
            for lp in logprobs:
                if lp > -100: # localized truncation for underflow
                    probs.append(math.exp(lp))
                else:
                    probs.append(0.0)
            
            sum_p = sum(probs)
            if sum_p <= 0: continue
            
            # 2. Normalize to create a valid probability distribution for Top-K
            # This approximates the "Local Entropy" given the choice is within Top-K
            norm_probs = [p / sum_p for p in probs]
            
            # 3. Calculate Shannon Entropy: H = -sum(p * ln(p))
            h = 0.0
            for p in norm_probs:
                if p > 0:
                    h -= p * math.log(p)
            
            total_entropy += h
            count += 1
            
        # Average per token to be length-invariant
        return total_entropy / count if count > 0 else 0.0

    def calculate_scr(self, texts):
        """Calculates Semantic Collapse Ratio (Average Pairwise Cosine Distance)."""
        if not self.embedding_model or len(texts) < 2:
            return 0.0
        
        embeddings = self.embedding_model.encode(texts)
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine(embeddings[i], embeddings[j])
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0

    def log_step(self, 
                 run_id: str,
                 scenario_id: str,
                 model_name: str,
                 step_index: int,
                 event_type: str,
                 prompt: str = "",
                 response_obj: Optional[dict] = None,
                 current_entropy: float = 0.0,
                 ige: Optional[float] = None,
                 scr: Optional[float] = None,
                 cbf: Optional[int] = None,
                 rdi: Optional[float] = None,
                 panic_counter: int = 0,
                 tool: Optional[str] = None,
                 compression_ratio: Optional[float] = None,
                 branching_func=None):
        """
        Main logging hook.
        Uses the unified LOG_SCHEMA.
        branching_func: A callable that generates N divergent responses (for SCR).
        """
        try:
            # If response_obj is provided, we can refine entropy calculation
            # But ideally, the caller passes 'current_entropy' already calculated
            
            # Branching Probe Logic (if requested and function provided)
            branches = []
            if branching_func and scr is None:
                print("Monitor: Triggering Branching Probe...")
                branches = branching_func() 
                scr = self.calculate_scr(branches)

            # Construct Log Entry matching LOG_SCHEMA
            entry = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "scenario_id": scenario_id,
                "model": model_name,
                "step_index": step_index,
                "event_type": event_type,
                "current_entropy": current_entropy,
                "ige": ige,
                "scr": scr,
                "cbf": cbf,
                "rdi": rdi,
                "panic_counter": panic_counter,
                "tool": tool,
                "compression_ratio": compression_ratio,
                # Extra debug info allowed
                "prompt_snippet": prompt[:100] if prompt else "",
                "branches_count": len(branches)
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
            # print(f"Monitor: Logged step {step_index}. Entropy: {current_entropy:.4f}")

        except Exception as e:
            print(f"Monitor Error during logging: {e}")

# Global instance
_MONITOR = None

def get_monitor():
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = TerminalBenchMonitor()
    return _MONITOR
