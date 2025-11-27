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
    "timestamp"
]

# Event types
EVENT_TYPE_TOOL_EXECUTION = "tool_execution"
EVENT_TYPE_LLM_REPLY = "llm_reply"
EVENT_TYPE_PERTURBATION = "perturbation_triggered"
EVENT_TYPE_INTERVENTION = "intervention"
EVENT_TYPE_UNKNOWN = "unknown_action"
