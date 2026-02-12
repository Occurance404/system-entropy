import os
import re
import hashlib
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
        """
        Embedding backend selection (env overrides):
        - SCR_EMBEDDING_BACKEND: auto|st|hash (default: auto)
        - SCR_EMBEDDING_MODEL: overrides model_name (default: all-MiniLM-L6-v2)
        - SCR_EMBEDDING_DEVICE: overrides device (default: cpu)
        - SCR_LOCAL_FILES_ONLY: 1/0; if 1, never downloads (default: 1)
        - SCR_HASH_DIM: dimension for hashing fallback (default: 1024)
        """
        backend = (os.getenv("SCR_EMBEDDING_BACKEND") or "auto").strip().lower()
        model_name = (os.getenv("SCR_EMBEDDING_MODEL") or model_name).strip()
        device = (os.getenv("SCR_EMBEDDING_DEVICE") or device).strip()
        local_files_only = (os.getenv("SCR_LOCAL_FILES_ONLY") or "1").strip().lower() in ("1", "true", "yes", "on")

        self.model_name = model_name
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.embedding_backend = "hash"
        self.hash_dim = self._parse_int_env("SCR_HASH_DIM", default=1024, min_value=128, max_value=16384)

        if backend in ("hash", "bow", "lexical"):
            self.embedding_model = None
            self.embedding_backend = "hash"
            self.model_name = None
            return

        # Singleton Logic: Only load if not already loaded or if config (name/device/files-only) changes
        current_key = (model_name, device, bool(local_files_only))
        if EmbeddingMetricService._shared_model is None or EmbeddingMetricService._model_cache_key != current_key:
            print(
                f"MetricService: Loading embedding model {model_name} on {device} "
                f"(local_files_only={local_files_only}) (Singleton)..."
            )
            try:
                EmbeddingMetricService._shared_model = SentenceTransformer(
                    model_name,
                    device=device,
                    local_files_only=local_files_only,
                )
                EmbeddingMetricService._model_cache_key = current_key
                print("MetricService: Embedding model loaded.")
            except Exception as e:
                print(f"MetricService: Failed to load SentenceTransformer model: {e}")
                EmbeddingMetricService._shared_model = None

        self.embedding_model = EmbeddingMetricService._shared_model
        if self.embedding_model is not None:
            self.embedding_backend = "st"

    def calculate_scr(self, branches: List[str]) -> Optional[float]:
        """
        Calculates Semantic Collapse Ratio (SCR) using embeddings.
        
        NOTE: Scientifically, this measures 'Semantic Divergence' or 'Instability'.
        - Higher Value: High Divergence (Agent is hallucinating/creative).
        - Zero Value: Total Collapse (Agent is looping/repeating exact text).
        - None: Metric unavailable (Model not loaded).
        
        Math: Mean Pairwise Cosine Distance (Range: [0, 2]).
        """
        if not branches:
            return None
        # Treat empty/whitespace-only branches as unusable probe outputs.
        # We need at least two non-empty branches to compute pairwise distance.
        cleaned_branches = [str(b).strip() for b in branches if str(b).strip()]
        if len(cleaned_branches) < 2:
            return None

        if self.embedding_model is not None:
            # Encode branches via SentenceTransformer
            try:
                embeddings = self.embedding_model.encode(cleaned_branches)
                embeddings_list = [e.tolist() for e in embeddings]
                return self._calculate_pairwise_distance(embeddings_list)
            except Exception as e:
                print(f"MetricService Error (SCR/SentenceTransformer): {e}")
                return None

        # Offline fallback: lexical hashing embeddings (always available, deterministic).
        try:
            embeddings = self._hash_embed_many(cleaned_branches, dim=self.hash_dim)
            return self._calculate_pairwise_distance(embeddings)
        except Exception as e:
            print(f"MetricService Error (SCR/HashingFallback): {e}")
            return None

    def calculate_rdi(self, current_content: str, ground_truth_text: str) -> Optional[float]:
        """
        Calculates Regressive Debt Index (RDI) by comparing embeddings.
        
        NOTE: This measures 'Semantic Drift' from the ground truth.
        - It uses Cosine Distance (0 to 1).
        - It does NOT measure completeness (length/coverage), only angular alignment.
        """
        if not current_content.strip() or not ground_truth_text:
            return None

        if self.embedding_model is not None:
            try:
                current_emb = self.embedding_model.encode(current_content).tolist()
                truth_emb = self.embedding_model.encode(ground_truth_text).tolist()
                return self._safe_cosine_distance(current_emb, truth_emb)
            except Exception as e:
                print(f"MetricService Error (RDI/SentenceTransformer): {e}")
                return None

        # Offline fallback: lexical hashing embeddings (0..2); keep cosine distance semantics.
        try:
            current_emb = self._hash_embed_one(current_content, dim=self.hash_dim)
            truth_emb = self._hash_embed_one(ground_truth_text, dim=self.hash_dim)
            return self._safe_cosine_distance(current_emb, truth_emb)
        except Exception as e:
            print(f"MetricService Error (RDI/HashingFallback): {e}")
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
                dist = self._safe_cosine_distance(embeddings[i], embeddings[j])
                if dist is not None and math.isfinite(dist):
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

    def _safe_cosine_distance(self, a: List[float], b: List[float]) -> Optional[float]:
        try:
            va = np.asarray(a, dtype=np.float32)
            vb = np.asarray(b, dtype=np.float32)
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            denom = na * nb
            if denom <= 0.0 or not math.isfinite(denom):
                return 0.0
            sim = float(np.dot(va, vb) / denom)
            # cosine distance in [0, 2] for general vectors; clip numerical noise.
            dist = 1.0 - sim
            if not math.isfinite(dist):
                return None
            return float(max(0.0, min(2.0, dist)))
        except Exception:
            # Fallback to scipy if anything odd happens.
            try:
                dist = float(cosine(a, b))
                if not math.isfinite(dist):
                    return None
                return float(max(0.0, min(2.0, dist)))
            except Exception:
                return None

    def _parse_int_env(self, key: str, *, default: int, min_value: int, max_value: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            value = int(str(raw).strip())
        except Exception:
            return default
        return max(min_value, min(max_value, value))

    _token_re = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

    def _hash_embed_one(self, text: str, *, dim: int) -> List[float]:
        vec = np.zeros(dim, dtype=np.float32)
        if not text:
            return vec.tolist()
        tokens = self._token_re.findall(text.lower())
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            # Stable across runs/Python versions.
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "big") % dim
            vec[idx] += 1.0
        vec = np.log1p(vec)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0 and math.isfinite(norm):
            vec /= norm
        return vec.tolist()

    def _hash_embed_many(self, texts: List[str], *, dim: int) -> List[List[float]]:
        return [self._hash_embed_one(t, dim=dim) for t in texts]
