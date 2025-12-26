import os
import json
import math
import uuid
from datetime import datetime
from typing import Optional, Any
import numpy as np

from src.shared.constants import LOG_SCHEMA, LOGS_DIR
from src.services.metrics import EmbeddingMetricService

class TerminalBenchMonitor:
    def __init__(self):
        self.log_file = os.path.join(
            LOGS_DIR, f"tb_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        self._default_run_id = str(uuid.uuid4())
        self._default_scenario_id = os.getenv("TB_TASK_ID") or "unknown"
        self._auto_step_index = 0
        
        # Use the Unified Singleton Metric Service
        try:
            print("Initializing Monitor: Linking to MetricService...")
            self.metric_service = EmbeddingMetricService()
        except Exception as e:
            print(f"Monitor Error: Failed to link MetricService: {e}")
            self.metric_service = None

    def log_step(self, 
                 run_id: Optional[str] = None,
                 scenario_id: Optional[str] = None,
                 model_name: str = "unknown",
                 step_index: Optional[int] = None,
                 event_type: str = "llm_call",
                 prompt: str = "",
                 messages: Optional[list] = None,
                 response_obj: Optional[dict] = None,
                 current_entropy: Optional[float] = None,
                 ige: Optional[float] = None,
                 scr: Optional[float] = None,
                 cbf: Optional[int] = None,
                 rdi: Optional[float] = None,
                 panic_counter: int = 0,
                 tool: Optional[str] = None,
                 compression_ratio: Optional[float] = None,
                 task_complete: Optional[bool] = None,
                 validation_passed: Optional[bool] = None,
                 validation_score: Optional[float] = None,
                 validation_details: Optional[str] = None,
                 prompt_tokens: Optional[int] = None,
                 completion_tokens: Optional[int] = None,
                 total_tokens: Optional[int] = None,
                 branches_count: Optional[int] = None,
                 branching_func=None):
        """
        Main logging hook.
        Uses the unified LOG_SCHEMA.
        branching_func: A callable that generates N divergent responses (for SCR).
        """
        try:
            if run_id is None:
                run_id = self._default_run_id
            if scenario_id is None:
                scenario_id = self._default_scenario_id
            if step_index is None:
                self._auto_step_index += 1
                step_index = self._auto_step_index

            # Branching Probe Logic (Fallback if SCR not provided by Orchestrator)
            branches = []
            scr_probe_event_types = {"perturbation_triggered", "periodic_probe", "proxy_probe", "proxy_shock_injected"}
            if branching_func and scr is None and self.metric_service and event_type in scr_probe_event_types:
                # Only trigger if explicitly requested and missing
                # print("Monitor: Triggering Branching Probe (Fallback)...")
                branches = branching_func() 
                scr = self.metric_service.calculate_scr(branches)

            if isinstance(current_entropy, float) and (math.isnan(current_entropy) or math.isinf(current_entropy)):
                current_entropy = None
            if isinstance(scr, float) and (math.isnan(scr) or math.isinf(scr)):
                scr = None
            if isinstance(ige, float) and (math.isnan(ige) or math.isinf(ige)):
                ige = None
            if isinstance(rdi, float) and (math.isnan(rdi) or math.isinf(rdi)):
                rdi = None
            if isinstance(compression_ratio, float) and (
                math.isnan(compression_ratio) or math.isinf(compression_ratio)
            ):
                compression_ratio = None
            if isinstance(validation_score, float) and (
                math.isnan(validation_score) or math.isinf(validation_score)
            ):
                validation_score = None

            # Construct Log Entry matching LOG_SCHEMA
            entry = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "scenario_id": scenario_id,
                "model": model_name,
                "step_index": step_index,
                "event_type": event_type,
                "current_entropy": current_entropy,
                "ige": ige,
                "scr": scr,
                "cbf": cbf,
                "rdi": rdi,
                "panic_counter": panic_counter,
                "tool": tool,
                "compression_ratio": compression_ratio,
                "task_complete": task_complete,
                "validation_passed": validation_passed,
                "validation_score": validation_score,
                "validation_details": (validation_details[:500] if isinstance(validation_details, str) else None),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_snippet": prompt[:100] if prompt else "",
                "branches_count": branches_count if isinstance(branches_count, int) else len(branches)
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            print(f"Monitor Error during logging: {e}")

# Global instance
_MONITOR = None

def get_monitor():
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = TerminalBenchMonitor()
    return _MONITOR
