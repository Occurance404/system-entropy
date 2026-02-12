import os

# Directory Paths
DATA_DIR = os.getenv("EXPERIMENT_DATA_DIR", "data")
LOGS_ROOT = os.getenv("EXPERIMENT_LOGS_ROOT", "logs")
LOGS_DIR = os.getenv("EXPERIMENT_TB_LOG_DIR", os.path.join(LOGS_ROOT, "terminal_bench"))
RESULTS_DIR = os.getenv("EXPERIMENT_RESULTS_DIR", os.path.join(DATA_DIR, "results"))
RUN_ARTIFACTS_DIR = os.getenv("EXPERIMENT_RUN_ARTIFACTS_DIR", os.path.join(DATA_DIR, "run_artifacts"))

# Shared logging fields
LOG_SCHEMA = [
    "run_id",
    "scenario_id",
    "model",
    "step_index",
    "event_type",
    "current_entropy",
    "ige",
    "scr",
    "cbf",
    "rdi",
    "panic_counter",
    "tool",
    "compression_ratio",
    "task_complete",
    "agent_done_claimed",
    "validation_passed",
    "validation_score",
    "validation_details",
    "ai_verifier_verdict",
    "ai_verifier_confidence",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "timestamp"
]

# Event types
EVENT_TYPE_TOOL_EXECUTION = "tool_execution"
EVENT_TYPE_LLM_REPLY = "llm_reply"
EVENT_TYPE_PERTURBATION = "perturbation_triggered"
EVENT_TYPE_INTERVENTION = "intervention"
EVENT_TYPE_UNKNOWN = "unknown_action"
