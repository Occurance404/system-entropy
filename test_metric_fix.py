import time
from src.services.metrics import EmbeddingMetricService

print("--- TEST: Singleton & Metrics Audit ---")

# 1. Test Singleton Loading
start_t = time.time()
s1 = EmbeddingMetricService()
t1 = time.time() - start_t
print(f"First Load Time: {t1:.4f}s")

start_t = time.time()
s2 = EmbeddingMetricService()
t2 = time.time() - start_t
print(f"Second Load Time: {t2:.4f}s")

if t2 < 1.0 and t1 > 1.0: # Assuming first load takes at least 1s
    print("PASS: Singleton Pattern is working (Second load was instant).")
else:
    # If first load was fast (cache), this might be a false negative, but we'll see.
    print(f"WARN: Singleton check ambiguous. (First: {t1:.2f}s, Second: {t2:.2f}s)")
    if s1.embedding_model is s2.embedding_model:
        print("PASS: Object identity confirmed (s1.model is s2.model).")
    else:
        print("FAIL: Object identity mismatch!")

# 2. Test SCR (Divergence) Logic
print("\n--- TEST: SCR Logic ---")
branches_identical = ["The cat sat on the mat."] * 5
branches_divergent = [
    "The cat sat on the mat.",
    "A dog chased a ball.",
    "SpaceX launched a rocket.",
    "I like ice cream.",
    "Python is a programming language."
]

scr_identical = s1.calculate_scr(branches_identical)
scr_divergent = s1.calculate_scr(branches_divergent)

if scr_identical is None or scr_divergent is None:
    print("SCR unavailable (embedding model not loaded).")
    print("Set up sentence-transformers model cache or enable network access to download it.")
    exit(0)

print(f"SCR (Identical Inputs): {scr_identical:.4f} (Expected ~0.0)")
print(f"SCR (Divergent Inputs): {scr_divergent:.4f} (Expected > 0.5)")

if scr_identical < 1e-6 and scr_divergent > 0.3:
    print("PASS: SCR correctly measures DIVERGENCE (0=Collapse, High=Chaos).")
else:
    print("FAIL: SCR logic does not match expectations.")

print("\n--- TEST COMPLETE ---")
