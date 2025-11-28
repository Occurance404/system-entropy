import math
import ast
import zlib
import numpy as np
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from src.interfaces import MetricServiceProtocol

class EmbeddingMetricService:
    """
    Implementation of MetricServiceProtocol.
    Decouples the Embedding Model and Math from the Orchestrator.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        print(f"MetricService: Loading embedding model {model_name}...")
        try:
            self.embedding_model = SentenceTransformer(model_name, device=device)
            print("MetricService: Embedding model loaded.")
        except Exception as e:
            print(f"MetricService: Failed to load model: {e}")
            self.embedding_model = None

    def calculate_scr(self, branches: List[str]) -> float:
        """
        Calculates Semantic Collapse Ratio (SCR) using embeddings.
        """
        if not self.embedding_model or not branches:
            return 0.0
        
        # Encode branches
        try:
            embeddings = self.embedding_model.encode(branches)
            # Convert to list of lists if necessary
            embeddings_list = [e.tolist() for e in embeddings]
            return self._calculate_pairwise_distance(embeddings_list)
        except Exception as e:
            print(f"MetricService Error (SCR): {e}")
            return 0.0

    def calculate_rdi(self, current_content: str, ground_truth_text: str) -> Optional[float]:
        """
        Calculates Regressive Debt Index (RDI) by comparing embeddings.
        """
        if not self.embedding_model:
            return None
            
        if not current_content.strip() or not ground_truth_text:
            return None
            
        try:
            # Encode individually
            current_emb = self.embedding_model.encode(current_content).tolist()
            truth_emb = self.embedding_model.encode(ground_truth_text).tolist()
            return cosine(current_emb, truth_emb)
        except Exception as e:
            print(f"MetricService Error (RDI): {e}")
            return None

    def calculate_entropy(self, logprobs: List[Any]) -> float:
        """
        Calculates a proxy for Entropy (Surprisal).
        H ~ - (1/N) * Sum(log(p_chosen))
        """
        if not logprobs:
            return 0.0
        
        clean_logprobs = []
        for lp in logprobs:
            if isinstance(lp, list):
                if lp: clean_logprobs.append(lp[0])
                else: clean_logprobs.append(0.0)
            elif isinstance(lp, (int, float)):
                clean_logprobs.append(lp)
            else:
                clean_logprobs.append(0.0)
                
        if not clean_logprobs:
            return 0.0
            
        total_surprisal = sum(-lp for lp in clean_logprobs)
        return total_surprisal / len(clean_logprobs)

    def calculate_ige(self, h_pre: float, h_post: float, token_cost: int) -> float:
        """Calculates Information Gain Efficiency."""
        if token_cost <= 0:
            return 0.0
        return float((h_pre - h_post) / token_cost)

    def calculate_compression_ratio(self, text: str) -> float:
        """Calculates the Compression Ratio."""
        if not text:
            return 1.0
        encoded = text.encode("utf-8")
        if len(encoded) == 0:
            return 1.0
        compressed = zlib.compress(encoded)
        return len(compressed) / len(encoded)

    def measure_cbf(self, code_snippet: str) -> int:
        """Measures Cyclomatic Complexity (CBF)."""
        clean_code = self._sanitize_code_block(code_snippet)
        try:
            tree = ast.parse(clean_code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity += 1
            return complexity
        except SyntaxError:
            return -1

    def _calculate_pairwise_distance(self, embeddings: List[List[float]]) -> float:
        """Internal helper for SCR math."""
        if len(embeddings) < 2:
            return 0.0
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine(embeddings[i], embeddings[j])
                distances.append(dist)
        if not distances:
            return 0.0
        return float(np.mean(distances))

    def _sanitize_code_block(self, raw_text: str) -> str:
        """Internal helper for code sanitization."""
        if "```python" in raw_text:
            parts = raw_text.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in raw_text:
             parts = raw_text.split("```")
             if len(parts) > 1:
                return parts[1].strip()
        return raw_text
