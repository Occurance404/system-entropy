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
    }

