import os
import json
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Re-use logic from your existing StateMonitor
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

    def log_step(self, model_name, prompt, messages, response_obj, branching_func=None):
        """
        Main logging hook.
        branching_func: A callable that generates N divergent responses (for SCR).
        """
        try:
            # 1. Extract Main Metrics
            # Note: LiteLLM response object structure
            choice = response_obj["choices"][0]
            content = choice["message"]["content"]
            
            # Extract logprobs if available (need to ask LiteLLM to return them)
            token_distributions = []
            if "logprobs" in choice and choice["logprobs"]:
                 # Handle OpenAI-style logprobs structure
                 if "content" in choice["logprobs"]:
                     # Iterate over tokens in the sequence
                     for t in choice["logprobs"]["content"]:
                         # Extract Top-K candidates for this token position
                         if "top_logprobs" in t and t["top_logprobs"]:
                             candidates = [c["logprob"] for c in t["top_logprobs"]]
                             token_distributions.append(candidates)
                         elif "logprob" in t:
                             # Fallback (should not happen with top_logprobs=5)
                             token_distributions.append([t["logprob"]])
            
            entropy = self.calculate_entropy(token_distributions)

            # 2. Branching Probe (Optional - expensive)
            # We might only do this if entropy is high, or every N steps
            scr = None
            branches = []
            if branching_func:
                print("Monitor: Triggering Branching Probe...")
                branches = branching_func() # Expects list of strings
                scr = self.calculate_scr(branches)

            # 3. Log Entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_snippet": prompt[:100],
                "response_length": len(content) if content else 0,
                "entropy": entropy,
                "scr": scr,
                "branches_count": len(branches)
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
            print(f"Monitor: Logged step. Entropy: {entropy:.4f}, SCR: {scr}")

        except Exception as e:
            print(f"Monitor Error during logging: {e}")

# Global instance
_MONITOR = None

def get_monitor():
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = TerminalBenchMonitor()
    return _MONITOR
