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
    
    Implements a Singleton pattern for the heavy embedding model to avoid 
    reloading it during repeated instantiations (e.g. in parameter sweeps).
    """
    
    _shared_model = None
    _model_cache_key = None

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        # Singleton Logic: Only load if not already loaded or if config (name/device) changes
        current_key = (model_name, device)
        if EmbeddingMetricService._shared_model is None or EmbeddingMetricService._model_cache_key != current_key:
            print(f"MetricService: Loading embedding model {model_name} on {device} (Singleton)...")
            try:
                EmbeddingMetricService._shared_model = SentenceTransformer(model_name, device=device)
                EmbeddingMetricService._model_cache_key = current_key
                print("MetricService: Embedding model loaded.")
            except Exception as e:
                print(f"MetricService: Failed to load model: {e}")
                EmbeddingMetricService._shared_model = None
        
        self.embedding_model = EmbeddingMetricService._shared_model

    def calculate_scr(self, branches: List[str]) -> Optional[float]:
        """
        Calculates Semantic Collapse Ratio (SCR) using embeddings.
        
        NOTE: Scientifically, this measures 'Semantic Divergence' or 'Instability'.
        - Higher Value: High Divergence (Agent is hallucinating/creative).
        - Zero Value: Total Collapse (Agent is looping/repeating exact text).
        - None: Metric unavailable (Model not loaded).
        
        Math: Mean Pairwise Cosine Distance (Range: [0, 2]).
        """
        if not self.embedding_model:
            return None
            
        if not branches:
            return None
        
        # Encode branches
        try:
            embeddings = self.embedding_model.encode(branches)
            # Convert to list of lists if necessary
            embeddings_list = [e.tolist() for e in embeddings]
            return self._calculate_pairwise_distance(embeddings_list)
        except Exception as e:
            print(f"MetricService Error (SCR): {e}")
            return None

    def calculate_rdi(self, current_content: str, ground_truth_text: str) -> Optional[float]:
        """
        Calculates Regressive Debt Index (RDI) by comparing embeddings.
        
        NOTE: This measures 'Semantic Drift' from the ground truth.
        - It uses Cosine Distance (0 to 1).
        - It does NOT measure completeness (length/coverage), only angular alignment.
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

    def calculate_entropy(self, logprobs: List[Any]) -> Optional[float]:
        """
        Calculates a proxy for Entropy (Surprisal).
        H ~ - (1/N) * Sum(log(p_chosen))
        
        NOTE: Units are 'Nats' (Natural Logarithm) if using standard OpenAI logprobs (ln).
        To convert to Bits: Multiply by 1.44 (1 / ln(2)).
        """
        if not logprobs:
            return None

        clean_logprobs: List[float] = []
        for lp in logprobs:
            if isinstance(lp, list):
                if not lp:
                    continue
                candidate = lp[0]
                if isinstance(candidate, (int, float)) and math.isfinite(candidate):
                    clean_logprobs.append(float(candidate))
                continue

            if isinstance(lp, (int, float)) and math.isfinite(lp):
                clean_logprobs.append(float(lp))

        if not clean_logprobs:
            return None

        # Sanity checks: log(p) should be <= 0 and typically < 0.
        # Some providers return placeholder zeros when logprobs are unsupported; treat as missing.
        if any(lp > 0.0 for lp in clean_logprobs):
            return None
        if min(clean_logprobs) > -1e-3:
            return None

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
