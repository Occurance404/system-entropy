from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shared.run_manifest import get_git_sha


ENV_SNAPSHOT_KEYS = [
    "VLLM_BASE_URL",
    "VLLM_MODEL_NAME",
    "RESCUE_BASE_URL",
    "RESCUE_MODEL_NAME",
    "SANDBOX_BACKEND",
    "SANDBOX_PER_RUN",
    "SANDBOX_PYTHON",
    "SCENARIO_SEED",
    "REQUEST_TOOLS",
    "REQUEST_LOGPROBS",
    "VALIDATION_FEEDBACK",
    "MAX_COMPLETION_TOKENS",
    "PROBE_MAX_TOKENS",
    "AI_VERIFIER",
    "AI_VERIFIER_INTERVAL",
    "AI_VERIFIER_CONFIDENCE",
    "AI_VERIFIER_FEEDBACK",
    "SCR_EMBEDDING_BACKEND",
    "SCR_LOCAL_FILES_ONLY",
    "EXPERIMENT_SANDBOX_ROOT",
]


@dataclass(frozen=True)
class SessionPaths:
    root: str
    logs_root: str
    tb_logs: str
    rescue_logs: str
    results: str
    run_artifacts: str
    sandboxes: str
    metadata: str
    console_log: str


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "experiment"


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _make_paths(base_dir: str, session_name: str) -> SessionPaths:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{ts}_{_slugify(session_name)}"
    root = _mkdir(os.path.join(base_dir, session_id))
    logs_root = _mkdir(os.path.join(root, "logs"))
    tb_logs = _mkdir(os.path.join(logs_root, "terminal_bench"))
    rescue_logs = _mkdir(os.path.join(logs_root, "rescue"))
    results = _mkdir(os.path.join(root, "results"))
    run_artifacts = _mkdir(os.path.join(root, "run_artifacts"))
    sandboxes = _mkdir(os.path.join(root, "sandboxes"))
    metadata = _mkdir(os.path.join(root, "metadata"))
    console_log = os.path.join(root, "console.log")
    return SessionPaths(
        root=root,
        logs_root=logs_root,
        tb_logs=tb_logs,
        rescue_logs=rescue_logs,
        results=results,
        run_artifacts=run_artifacts,
        sandboxes=sandboxes,
        metadata=metadata,
        console_log=console_log,
    )


def _snapshot_env() -> Dict[str, str]:
    return {k: os.getenv(k) for k in ENV_SNAPSHOT_KEYS if os.getenv(k) is not None}


def _write_json(path: str, payload: Dict) -> None:
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _update_latest_pointer(base_dir: str, session_id: str, session_path: str) -> None:
    _mkdir(base_dir)
    latest_txt = os.path.join(base_dir, "LATEST")
    latest_json = os.path.join(base_dir, "LATEST.json")
    latest_link = os.path.join(base_dir, "latest")

    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(f"{session_id}\n")

    _write_json(
        latest_json,
        {
            "session_id": session_id,
            "session_path": os.path.abspath(session_path),
            "updated_at": datetime.now().isoformat(),
        },
    )

    try:
        if os.path.lexists(latest_link):
            if os.path.islink(latest_link) or os.path.isfile(latest_link):
                os.unlink(latest_link)
            elif os.path.isdir(latest_link):
                return
        os.symlink(session_id, latest_link)
    except Exception:
        # Symlinks may be unavailable; keep text/json pointers as source of truth.
        pass


def _write_readme(paths: SessionPaths, meta: Dict, manifest_paths: List[str], log_paths: List[str]) -> None:
    readme_path = os.path.join(paths.root, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Session\n\n")
        f.write(f"- Session ID: `{meta['session_id']}`\n")
        f.write(f"- Started: `{meta['started_at']}`\n")
        f.write(f"- Finished: `{meta.get('finished_at')}`\n")
        f.write(f"- Exit code: `{meta.get('exit_code')}`\n")
        f.write(f"- Git SHA: `{meta.get('git_sha')}`\n")
        f.write(f"- Command: `{meta['command']}`\n")
        if meta.get("notes"):
            f.write(f"- Notes: `{meta['notes']}`\n")
        f.write("\n## Paths\n")
        f.write(f"- Console log: `{os.path.relpath(paths.console_log, paths.root)}`\n")
        f.write(f"- TerminalBench logs: `{os.path.relpath(paths.tb_logs, paths.root)}`\n")
        f.write(f"- Rescue logs: `{os.path.relpath(paths.rescue_logs, paths.root)}`\n")
        f.write(f"- Run artifacts: `{os.path.relpath(paths.run_artifacts, paths.root)}`\n")
        f.write(f"- Results: `{os.path.relpath(paths.results, paths.root)}`\n")
        f.write(f"- Sandboxes: `{os.path.relpath(paths.sandboxes, paths.root)}`\n")
        f.write("\n## Outputs\n")
        f.write(f"- Manifests found: `{len(manifest_paths)}`\n")
        f.write(f"- Log files found: `{len(log_paths)}`\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an experiment in an isolated session folder with automatic metadata and output routing.\n"
            "Use '--' before the actual experiment command."
        )
    )
    parser.add_argument("--name", required=True, help="Short session name (used in folder name).")
    parser.add_argument("--notes", default="", help="Optional notes stored in metadata/README.")
    parser.add_argument("--base-dir", default="data/experiments", help="Root directory for experiment sessions.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute, e.g. -- .venv/bin/python ...")
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        print("ERROR: Missing command. Example:", file=sys.stderr)
        print(
            "  .venv/bin/python scripts/run_experiment_session.py --name rescue_real -- "
            ".venv/bin/python experiments/run_rescue_experiment.py --scenario_id drug_filter_shock --max_steps 20",
            file=sys.stderr,
        )
        return 2

    paths = _make_paths(args.base_dir, args.name)
    session_id = os.path.basename(paths.root)
    command_str = shlex.join(command)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENT_SESSION_ID": session_id,
            "EXPERIMENT_SESSION_DIR": paths.root,
            "EXPERIMENT_LOGS_ROOT": paths.logs_root,
            "EXPERIMENT_TB_LOG_DIR": paths.tb_logs,
            "EXPERIMENT_RESULTS_DIR": paths.results,
            "EXPERIMENT_RUN_ARTIFACTS_DIR": paths.run_artifacts,
            "EXPERIMENT_SANDBOX_ROOT": paths.sandboxes,
        }
    )

    started_at = datetime.now().isoformat()
    metadata_path = os.path.join(paths.metadata, "session.json")
    metadata = {
        "session_id": session_id,
        "session_dir": os.path.abspath(paths.root),
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "exit_code": None,
        "command": command_str,
        "cwd": os.getcwd(),
        "git_sha": get_git_sha(REPO_ROOT),
        "notes": args.notes,
        "env": _snapshot_env(),
        "overrides": {
            "EXPERIMENT_LOGS_ROOT": paths.logs_root,
            "EXPERIMENT_TB_LOG_DIR": paths.tb_logs,
            "EXPERIMENT_RESULTS_DIR": paths.results,
            "EXPERIMENT_RUN_ARTIFACTS_DIR": paths.run_artifacts,
            "EXPERIMENT_SANDBOX_ROOT": paths.sandboxes,
        },
    }
    _write_json(metadata_path, metadata)

    print(f"[session] {session_id}")
    print(f"[dir] {paths.root}")
    print(f"[cmd] {command_str}")
    print("")

    exit_code = 0
    with open(paths.console_log, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
        process.wait()
        exit_code = int(process.returncode or 0)

    finished_at = datetime.now().isoformat()
    manifest_paths = sorted(glob.glob(os.path.join(paths.run_artifacts, "*", "manifest.json")))
    log_paths = sorted(glob.glob(os.path.join(paths.logs_root, "**", "*.jsonl"), recursive=True))

    metadata.update(
        {
            "finished_at": finished_at,
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "manifest_count": len(manifest_paths),
            "log_file_count": len(log_paths),
        }
    )
    _write_json(metadata_path, metadata)

    index_payload = {
        "session_id": session_id,
        "created_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "manifests": [os.path.relpath(p, paths.root) for p in manifest_paths],
        "logs": [os.path.relpath(p, paths.root) for p in log_paths],
        "console_log": os.path.relpath(paths.console_log, paths.root),
    }
    _write_json(os.path.join(paths.metadata, "index.json"), index_payload)
    _write_readme(paths, metadata, manifest_paths, log_paths)
    _update_latest_pointer(args.base_dir, session_id, paths.root)

    print("")
    print(f"[done] exit={exit_code}")
    print(f"[summary] {os.path.join(paths.metadata, 'index.json')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
