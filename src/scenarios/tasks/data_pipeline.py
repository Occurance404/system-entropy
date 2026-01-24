import json
import hashlib
import os
from datetime import datetime, timedelta

def _stable_int(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")

def _stable_choice(seed: int, key: str, options: list[str]) -> str:
    if not options:
        return ""
    return options[_stable_int(seed, key) % len(options)]

def _stable_weighted_choice(seed: int, key: str, options: list[str], weights: list[int]) -> str:
    if not options:
        return ""
    if len(options) != len(weights) or not weights:
        return _stable_choice(seed, key, options)
    total = int(sum(max(0, int(w)) for w in weights))
    if total <= 0:
        return _stable_choice(seed, key, options)
    r = _stable_int(seed, key) % total
    acc = 0
    for opt, w in zip(options, weights):
        w = max(0, int(w))
        acc += w
        if r < acc:
            return opt
    return options[-1]

def _stable_uuid_short(seed: int, i: int) -> str:
    digest = hashlib.sha256(f"{seed}:uuid:{i}".encode("utf-8")).hexdigest()
    return digest[:8]

def setup_environment(base_path="data/sandbox_task_1", seed: int | None = None):
    """
    Generates a realistic server log file.
    """
    import os
    os.makedirs(base_path, exist_ok=True)

    if seed is None:
        # Default seed keeps the dataset stable across runs (paper-friendly).
        seed = int(os.getenv("SCENARIO_SEED") or 0)

    log_file_path = os.path.join(base_path, "server_logs.json")
    print(f"Generating Data Pipeline logs at {log_file_path}...")
    
    logs = []
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    services = ["auth-service", "payment-gateway", "database-shard-01", "frontend-ui"]
    level_weights = [50, 30, 15, 5]

    # Fixed reference time for reproducibility (validator does not depend on exact timestamps).
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    
    # Generate 50 log entries
    for i in range(50):
        level = _stable_weighted_choice(seed, f"level:{i}", levels, level_weights)
        service = _stable_choice(seed, f"service:{i}", services)
        
        # Standard ISO timestamp for Phase 1
        ts = (t0 + timedelta(minutes=i)).isoformat()
        
        entry = {
            "timestamp": ts,
            "level": level,
            "service": service,
            "message": (
                f"Transaction {_stable_uuid_short(seed, i)} processed."
                if level == "INFO"
                else f"Connection timeout in {service}."
            ),
        }
        
        # Inject explicit errors for the task
        if i == 15 or i == 42:
            entry["level"] = "ERROR"
            entry["message"] = "Critical failure: Database deadlock detected."
            
        logs.append(entry)
        
    with open(log_file_path, "w") as f:
        json.dump(logs, f, indent=2)
        
    print("Environment Ready. Created server_logs.json")

if __name__ == "__main__":
    setup_environment()
