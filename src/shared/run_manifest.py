from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def get_git_sha(repo_root: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_manifest(
    *,
    run_id: str,
    scenario_id: str,
    model_name: str,
    sandbox_path: str,
    log_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    env_allowlist = [
        "SANDBOX_BACKEND",
        "SANDBOX_PER_RUN",
        "SANDBOX_PYTHON",
        "RESET_SANDBOX",
        "SCENARIO_SEED",
        "REQUEST_TOOLS",
        "REQUEST_LOGPROBS",
        "VALIDATION_FEEDBACK",
        "MAX_COMPLETION_TOKENS",
        "PROBE_MAX_TOKENS",
        "VLLM_BASE_URL",
        "VLLM_MODEL_NAME",
        "RESCUE_BASE_URL",
        "RESCUE_MODEL_NAME",
        "SCR_EMBEDDING_BACKEND",
        "SCR_EMBEDDING_MODEL",
        "SCR_EMBEDDING_DEVICE",
        "SCR_LOCAL_FILES_ONLY",
        "SCR_HASH_DIM",
    ]
    env_snapshot = {k: os.getenv(k) for k in env_allowlist if os.getenv(k) is not None}
    return {
        "schema_version": 1,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "model_name": model_name,
        "started_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "sandbox_path": sandbox_path,
        "log_file": log_file,
        "config": config or {},
        "env": env_snapshot,
    }
