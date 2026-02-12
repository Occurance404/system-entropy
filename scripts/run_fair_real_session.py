from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Dict, Tuple

from dotenv import dotenv_values
from openai import OpenAI


def _python_bin() -> str:
    local = os.path.join(".venv", "bin", "python")
    return local if os.path.exists(local) else "python3"


def _resolve_env(key: str, dotenv: Dict[str, str]) -> str:
    return (os.getenv(key) or dotenv.get(key) or "").strip()


def _check_models_endpoint(base_url: str) -> Tuple[bool, str]:
    if not base_url.startswith(("http://", "https://")):
        return False, "Base URL must start with http:// or https://"
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer probe"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # 401/403 still proves endpoint reachability.
        if e.code in (401, 403):
            return True, f"HTTP {e.code} (reachable)"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def _supports_logprobs(client: OpenAI, model: str) -> Tuple[bool, str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply exactly with: ok"}],
            max_tokens=2,
            temperature=0,
            logprobs=True,
            top_logprobs=1,
        )
        choice = resp.choices[0]
        lp = getattr(getattr(choice, "logprobs", None), "content", None)
        if lp:
            return True, ""
        return False, "No logprobs.content returned"
    except Exception as e:
        return False, str(e)


def _supports_tools(client: OpenAI, model: str) -> Tuple[bool, str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "returns pong",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call ping tool."}],
            max_tokens=8,
            temperature=0,
            tools=tools,
            tool_choice="auto",
        )
        return True, ""
    except Exception as e:
        return False, str(e)


def _build_inner_command(
    py: str,
    mode: str,
    scenario_id: str,
    max_steps: int,
    with_probes: bool,
    stop_on_agent_done: bool,
) -> list[str]:
    if mode in ("rescue_baseline", "rescue"):
        cmd = [
            py,
            "experiments/run_rescue_experiment.py",
            "--scenario_id",
            scenario_id,
            "--max_steps",
            str(max_steps),
            "--enable_validation",
            "--stop_on_success",
        ]
        if stop_on_agent_done:
            cmd.append("--stop_on_agent_done")
        if mode == "rescue":
            cmd.append("--enable_rescue")
        if not with_probes:
            cmd.append("--disable_probes")
        return cmd

    if mode == "hard":
        cmd = [
            py,
            "experiments/run_hard_mode.py",
            "--scenario_id",
            scenario_id,
            "--max_steps",
            str(max_steps),
            "--require_real_agent",
        ]
        if stop_on_agent_done:
            cmd.append("--autonomous")
        if with_probes:
            cmd.extend(["--probe_interval", "5"])
        else:
            cmd.append("--cheap")
        return cmd

    if mode == "simulate":
        cmd = [
            py,
            "experiments/simulate_real.py",
            "--scenario_id",
            scenario_id,
            "--max_steps",
            str(max_steps),
        ]
        if stop_on_agent_done:
            cmd.append("--stop_on_agent_done")
        if not with_probes:
            cmd.append("--cheap")
        return cmd

    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fair real-agent experiment session with preflight checks and stability defaults."
    )
    parser.add_argument(
        "--mode",
        choices=["rescue_baseline", "rescue", "hard", "simulate"],
        default="rescue_baseline",
        help="Experiment runner to launch.",
    )
    parser.add_argument("--scenario_id", default="drug_filter_shock", help="Scenario ID.")
    parser.add_argument("--max_steps", type=int, default=40, help="Max steps.")
    parser.add_argument("--model", default=None, help="Override VLLM_MODEL_NAME for this run.")
    parser.add_argument("--name", default=None, help="Session name (default auto-generated).")
    parser.add_argument("--notes", default="fair real-agent run", help="Session notes.")
    parser.add_argument("--base-dir", default="data/experiments", help="Session root directory.")
    parser.add_argument("--with-probes", action="store_true", help="Keep branching probes enabled.")
    parser.add_argument(
        "--stop-on-agent-done",
        action="store_true",
        help="Enable model self-reported completion stopping (off by default to avoid false positives).",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip endpoint/capability probe checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command/env and exit.")
    args = parser.parse_args()

    dotenv = dotenv_values(".env")
    api_key = _resolve_env("VLLM_API_KEY", dotenv)
    base_url = _resolve_env("VLLM_BASE_URL", dotenv)
    model_name = (args.model or _resolve_env("VLLM_MODEL_NAME", dotenv)).strip()

    if not api_key:
        print("ERROR: VLLM_API_KEY is not set in env or .env")
        return 2
    if not base_url:
        print("ERROR: VLLM_BASE_URL is not set in env or .env")
        return 2
    if not model_name:
        print("ERROR: VLLM_MODEL_NAME is not set in env or .env")
        return 2

    tools_mode = "auto"
    logprobs_mode = "auto"

    if not args.skip_preflight:
        reachable, reach_note = _check_models_endpoint(base_url)
        print(f"[preflight] models endpoint: {'OK' if reachable else 'FAIL'} ({reach_note})")
        if not reachable:
            print("ERROR: Endpoint unreachable. This is infrastructure failure, not agent failure.")
            return 2

        client = OpenAI(base_url=base_url, api_key=api_key)
        tools_ok, tools_note = _supports_tools(client, model_name)
        logprobs_ok, logprobs_note = _supports_logprobs(client, model_name)

        tools_mode = "on" if tools_ok else "off"
        logprobs_mode = "on" if logprobs_ok else "off"

        print(f"[preflight] tools: {'on' if tools_ok else 'off'} ({tools_note or 'supported'})")
        print(f"[preflight] logprobs: {'on' if logprobs_ok else 'off'} ({logprobs_note or 'supported'})")

    py = _python_bin()
    session_name = args.name or f"fair_{args.mode}_{args.scenario_id}"
    inner_cmd = _build_inner_command(
        py,
        args.mode,
        args.scenario_id,
        args.max_steps,
        args.with_probes,
        args.stop_on_agent_done,
    )

    wrapper_cmd = [
        py,
        "scripts/run_experiment_session.py",
        "--name",
        session_name,
        "--base-dir",
        args.base_dir,
        "--notes",
        args.notes,
        "--",
    ] + inner_cmd

    env = os.environ.copy()
    env.update(
        {
            "VLLM_BASE_URL": base_url,
            "VLLM_API_KEY": api_key,
            "VLLM_MODEL_NAME": model_name,
            "REQUEST_TOOLS": tools_mode,
            "REQUEST_LOGPROBS": logprobs_mode,
            "SANDBOX_BACKEND": os.getenv("SANDBOX_BACKEND", "local"),
            "SANDBOX_PER_RUN": os.getenv("SANDBOX_PER_RUN", "1"),
            "SCENARIO_SEED": os.getenv("SCENARIO_SEED", "0"),
            "VALIDATION_FEEDBACK": os.getenv("VALIDATION_FEEDBACK", "on"),
            "MAX_COMPLETION_TOKENS": os.getenv("MAX_COMPLETION_TOKENS", "1024"),
            "PROBE_MAX_TOKENS": os.getenv("PROBE_MAX_TOKENS", "192"),
            "AI_VERIFIER": os.getenv("AI_VERIFIER", "on"),
            "AI_VERIFIER_INTERVAL": os.getenv("AI_VERIFIER_INTERVAL", "5"),
            "AI_VERIFIER_CONFIDENCE": os.getenv("AI_VERIFIER_CONFIDENCE", "0.8"),
            "AI_VERIFIER_FEEDBACK": os.getenv("AI_VERIFIER_FEEDBACK", "on"),
        }
    )

    print("[profile] mode:", args.mode)
    print("[profile] scenario:", args.scenario_id)
    print("[profile] model:", model_name)
    print("[profile] REQUEST_TOOLS:", env["REQUEST_TOOLS"])
    print("[profile] REQUEST_LOGPROBS:", env["REQUEST_LOGPROBS"])
    print("[profile] SANDBOX_BACKEND:", env["SANDBOX_BACKEND"])
    print("[profile] stop_on_agent_done:", bool(args.stop_on_agent_done))
    print("[profile] AI_VERIFIER:", env["AI_VERIFIER"])
    print("[profile] AI_VERIFIER_INTERVAL:", env["AI_VERIFIER_INTERVAL"])
    print("[profile] AI_VERIFIER_CONFIDENCE:", env["AI_VERIFIER_CONFIDENCE"])

    if args.dry_run:
        print("[dry-run] command:")
        print(" ".join(wrapper_cmd))
        return 0

    result = subprocess.run(wrapper_cmd, env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
