import math
import ast
import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import cosine

class StateMonitor:
    """
    Module 3: The State Monitor (The Probe) - v2.0
    
    Runs parallel to the agent to measure cognitive and behavioral metrics.
    Now supports:
    1. Information Gain Efficiency (IGE)
    2. Semantic Collapse Ratio (SCR)
    3. Standard Entropy Metrics
    """
    
    def __init__(self):
        self.metrics_log = []

    def calculate_information_gain_efficiency(self, h_pre: float, h_post: float, token_cost: int) -> float:
        """
        Calculates Information Gain Efficiency (IGE).
        IGE = (H_pre - H_post) / Token_Cost
        
        Positive IGE -> Uncertainty Reduced (Good).
        Negative/Zero IGE -> Uncertainty Stuck or Increased (Thrashing).
        """
        if token_cost <= 0:
            return 0.0
        
        delta_h = h_pre - h_post
        return float(delta_h / token_cost)

    def calculate_semantic_collapse_ratio(self, embeddings: List[List[float]]) -> float:
        """
        Calculates Semantic Collapse Ratio (SCR) based on pairwise cosine distance
        of generated Top-K thoughts.
        
        SCR = Avg(CosineDistance(Ei, Ej))
        
        High Distance -> High Confusion/Collapse (Agent is considering wildly different things).
        Low Distance -> Low Confusion (Agent is confident, even if wrong).
        """
        if len(embeddings) < 2:
            return 0.0
            
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine distance = 1 - cosine_similarity
                # Scipy cosine returns distance directly
                dist = cosine(embeddings[i], embeddings[j])
                distances.append(dist)
                
        if not distances:
            return 0.0
            
        return float(np.mean(distances))

    def calculate_entropy(self, token_distributions: List[List[float]]) -> float:
        """
        Calculates Shannon Entropy from a list of Top-K token distributions.
        Synced with TerminalBenchMonitor.
        """
        if not token_distributions:
            return 0.0
        
        total_entropy = 0.0
        count = 0
        
        for logprobs in token_distributions:
            # Validation for Mock data safety
            if not isinstance(logprobs, list): continue 
            if not logprobs: continue
            
            # 1. Convert logprobs to probabilities
            probs = []
            for lp in logprobs:
                # localized truncation
                if lp > -100:
                    probs.append(math.exp(lp))
                else:
                    probs.append(0.0)
            
            sum_p = sum(probs)
            if sum_p <= 0: continue
            
            # 2. Normalize
            norm_probs = [p / sum_p for p in probs]
            
            # 3. Calculate H
            h = 0.0
            for p in norm_probs:
                if p > 0:
                    h -= p * math.log(p)
            
            total_entropy += h
            count += 1
            
        return total_entropy / count if count > 0 else 0.0

    def _sanitize_code_block(self, raw_text: str) -> str:
        """
        Extracts pure Python code from Markdown wrappers or chatter.
        """
        # 1. Try to find markdown blocks
        if "```python" in raw_text:
            parts = raw_text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        if "```" in raw_text:
             parts = raw_text.split("```")
             if len(parts) > 1:
                return parts[1].strip()
                
        # 2. Fallback: return raw text if no markdown found
        # (The AST parser might still fail, but we tried)
        return raw_text

    def measure_cyclomatic_complexity(self, code_snippet: str) -> int:
        """
        Parses Python code AST to measure structural complexity (CBF).
        Includes sanitization to handle LLM output.
        """
        clean_code = self._sanitize_code_block(code_snippet)
        try:
            tree = ast.parse(clean_code)
            # Basic complexity counting: Functions + Loops + Conditionals + 1
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity += 1
            return complexity
        except SyntaxError:
            return -1 # Indicates parsing failure (broken code)

    def check_goal_deviance(self, current_plan_embedding, ground_truth_embedding):
        """
        Compares embeddings to check for drift.
        """
        pass
