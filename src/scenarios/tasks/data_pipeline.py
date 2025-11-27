import json
import random
import time
from datetime import datetime, timedelta

def setup_environment(base_path="data/sandbox_task_1"):
    """
    Generates a realistic server log file.
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    log_file_path = os.path.join(base_path, "server_logs.json")
    print(f"Generating Data Pipeline logs at {log_file_path}...")
    
    logs = []
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    services = ["auth-service", "payment-gateway", "database-shard-01", "frontend-ui"]
    
    # Generate 50 log entries
    for i in range(50):
        level = random.choices(levels, weights=[50, 30, 15, 5])[0]
        service = random.choice(services)
        
        # Standard ISO timestamp for Phase 1
        ts = (datetime.now() - timedelta(minutes=50-i)).isoformat()
        
        entry = {
            "timestamp": ts,
            "level": level,
            "service": service,
            "message": f"Transaction {uuid_short()} processed." if level == "INFO" else f"Connection timeout in {service}."
        }
        
        # Inject explicit errors for the task
        if i == 15 or i == 42:
            entry["level"] = "ERROR"
            entry["message"] = "Critical failure: Database deadlock detected."
            
        logs.append(entry)
        
    with open(log_file_path, "w") as f:
        json.dump(logs, f, indent=2)
        
    print("Environment Ready. Created server_logs.json")

def uuid_short():
    import uuid
    return str(uuid.uuid4())[:8]

if __name__ == "__main__":
    setup_environment()
