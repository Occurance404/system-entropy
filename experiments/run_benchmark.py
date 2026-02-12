import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import dotenv_values

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.real_agent import OpenAICompatibleAgent
from src.monitor.terminal_bench_monitor import TerminalBenchMonitor
from src.orchestrator.engine import Orchestrator
from src.scenarios.validation_ops import validate_scenario
from src.services.metrics import EmbeddingMetricService
from src.shared.constants import RESULTS_DIR


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_env(key: str, dotenv: Dict[str, str]) -> Optional[str]:
    return os.getenv(key) or dotenv.get(key)


def _classify_error(error: Exception) -> str:
    msg = str(error).lower()
    if "api key" in msg or "unauthorized" in msg or "401" in msg:
        return "auth"
    if "connection" in msg or "timeout" in msg or "name resolution" in msg or "dns" in msg:
        return "infrastructure"
    if "does not support tools" in msg or "tool_choice" in msg or "logprobs" in msg:
        return "capability_mismatch"
    if "missing api key env var" in msg:
        return "configuration"
    return "unknown"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.2


def _parse_model_spec(raw: Dict[str, Any], dotenv: Dict[str, str]) -> ModelSpec:
    name = str(raw["name"])
    model = str(raw["model"])
    base_url = str(raw.get("base_url") or _resolve_env("VLLM_BASE_URL", dotenv) or "")
    api_key_env = str(raw.get("api_key_env") or "VLLM_API_KEY")
    temperature = float(raw.get("temperature", 0.2))
    if not base_url:
        raise ValueError(f"ModelSpec '{name}' missing base_url (or VLLM_BASE_URL).")
    return ModelSpec(
        name=name,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature,
    )


def _run_one(
    *,
    model: ModelSpec,
    scenario_id: str,
    max_steps: int,
    rep_index: int,
    dotenv: Dict[str, str],
    enable_branching_probes: bool,
    probe_interval: int,
    probe_branch_count: int,
    enable_intervention: bool,
    validation_interval: int,
    stop_on_agent_done: bool,
) -> Dict[str, Any]:
    api_key = _resolve_env(model.api_key_env, dotenv)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {model.api_key_env}")

    run_id = str(uuid.uuid4())
    metric_service = EmbeddingMetricService()
    metrics_monitor = TerminalBenchMonitor()
    metric_backend = getattr(metric_service, "embedding_backend", None)
    metric_model = getattr(metric_service, "model_name", None)
    metric_device = getattr(metric_service, "device", None)
    metric_local_files_only = getattr(metric_service, "local_files_only", None)
    metric_hash_dim = getattr(metric_service, "hash_dim", None)

    agent = OpenAICompatibleAgent(
        model_name=model.model,
        base_url=model.base_url,
        api_key=api_key,
        temperature=model.temperature,
    )

    orchestrator = Orchestrator(
        scenario_id=scenario_id,
        agent=agent,
        metric_service=metric_service,
        run_id=run_id,
        metrics_monitor=metrics_monitor,
        enable_intervention=enable_intervention,
        enable_validation=True,
        validation_interval=validation_interval,
        stop_on_success=True,
        stop_on_agent_done=stop_on_agent_done,
        enable_branching_probes=enable_branching_probes,
        probe_interval=probe_interval,
        probe_branch_count=probe_branch_count,
    )

    try:
        total_tokens = 0
        probe_total_tokens = 0
        last_event_type = None
        agent_done_claim_count = 0
        first_agent_done_claim_step = None
        for _ in range(max_steps):
            step_result = orchestrator.step()
            last_event_type = step_result.get("event_type")

            if isinstance(step_result.get("total_tokens"), int):
                total_tokens += step_result["total_tokens"]

            if step_result.get("type") == "perturbation_triggered":
                probe_metrics = step_result.get("probe_metrics") or {}
                if isinstance(probe_metrics, dict) and isinstance(probe_metrics.get("probe_total_tokens"), int):
                    probe_total_tokens += probe_metrics["probe_total_tokens"]

            if bool(step_result.get("agent_done_claimed")):
                agent_done_claim_count += 1
                if first_agent_done_claim_step is None:
                    try:
                        first_agent_done_claim_step = int(step_result.get("step_index"))
                    except Exception:
                        first_agent_done_claim_step = orchestrator.step_count

            if step_result.get("task_complete") or last_event_type == "task_complete":
                break
            if last_event_type == "intervention" and enable_intervention:
                break

        validation = validate_scenario(scenario_id, orchestrator.sandbox_path)
        validation_passed = validation.passed if validation is not None else None
        validation_score = validation.score if validation is not None else None

        # Best-effort drift summary (written alongside the manifest for reproducibility).
        try:
            summary_path = os.path.join(orchestrator.run_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(orchestrator.compute_drift_summary(), f, indent=2, sort_keys=True)
        except Exception:
            pass

        return {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "model_name": model.name,
            "model": model.model,
            "scenario_id": scenario_id,
            "rep_index": int(rep_index),
            "max_steps": max_steps,
            "steps_executed": orchestrator.step_count,
            "last_event_type": last_event_type,
            "validation_passed": validation_passed,
            "validation_score": validation_score,
            "log_file": metrics_monitor.log_file,
            "run_dir": orchestrator.run_dir,
            "total_tokens": total_tokens,
            "probe_total_tokens": probe_total_tokens,
            "total_tokens_including_probes": total_tokens + probe_total_tokens,
            "agent_done_claim_count": agent_done_claim_count,
            "first_agent_done_claim_step": first_agent_done_claim_step,
            "probe_interval": probe_interval,
            "probe_branch_count": probe_branch_count,
            "enable_branching_probes": enable_branching_probes,
            "enable_intervention": enable_intervention,
            "stop_on_agent_done": bool(stop_on_agent_done),
            "metric_embedding_backend": metric_backend,
            "metric_embedding_model": metric_model,
            "metric_embedding_device": metric_device,
            "metric_local_files_only": metric_local_files_only,
            "metric_hash_dim": metric_hash_dim,
        }
    finally:
        # Always stop the sandbox connector, even on mid-run exceptions.
        try:
            if getattr(orchestrator, "connector", None):
                orchestrator.connector.stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible benchmark sweep across models and scenarios.")
    parser.add_argument("--models", default="benchmarks/models.example.json", help="Path to models JSON file.")
    parser.add_argument("--suite", default="benchmarks/suite_v1.json", help="Path to suite JSON file.")
    parser.add_argument(
        "--only-scenarios",
        default=None,
        help="Comma-separated scenario IDs to run (default: run all scenarios in the suite).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per (model, scenario).")
    parser.add_argument("--probe-mode", choices=["off", "shock", "periodic"], default="shock")
    parser.add_argument("--probe-interval", type=int, default=5, help="Used when --probe-mode=periodic.")
    parser.add_argument("--probe-branches", type=int, default=3, help="Branch count for SCR probes.")
    parser.add_argument("--max-steps", type=int, default=60, help="Fallback max steps if suite omits it.")
    parser.add_argument("--enable-intervention", action="store_true", help="Enable entropy-based intervention (if entropy is available).")
    parser.add_argument("--validation-interval", type=int, default=1, help="Validate every N steps.")
    parser.add_argument(
        "--stop-on-agent-done",
        action="store_true",
        help="Also stop early if the agent explicitly signals completion in an LLM reply.",
    )
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/results/benchmark_<ts>.csv).")
    args = parser.parse_args()

    dotenv = dotenv_values(".env")
    models_raw = _load_json(args.models)
    suite = _load_json(args.suite)

    model_specs = [_parse_model_spec(m, dotenv) for m in models_raw]
    scenarios = suite.get("scenarios") or []
    if not scenarios:
        raise SystemExit(f"Suite has no scenarios: {args.suite}")

    if args.only_scenarios:
        requested = {s.strip() for s in str(args.only_scenarios).split(",") if s.strip()}
        scenarios = [s for s in scenarios if str(s.get("scenario_id")) in requested]
        if not scenarios:
            raise SystemExit(f"No scenarios matched --only-scenarios={args.only_scenarios}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(RESULTS_DIR, f"benchmark_{suite.get('suite_id', 'suite')}_{ts}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    enable_branching_probes = args.probe_mode != "off"
    probe_interval = 0 if args.probe_mode != "periodic" else max(1, args.probe_interval)

    rows: List[Dict[str, Any]] = []
    for model in model_specs:
        api_key = _resolve_env(model.api_key_env, dotenv)
        if not api_key:
            print(f"SKIP model '{model.name}': missing env var {model.api_key_env}")
            continue

        for scenario in scenarios:
            scenario_id = str(scenario.get("scenario_id"))
            max_steps = int(scenario.get("max_steps") or args.max_steps)

            for rep in range(args.repeats):
                print(
                    f"Running model={model.name} scenario={scenario_id} rep={rep+1}/{args.repeats} "
                    f"probe={args.probe_mode} branches={args.probe_branches}..."
                )
                try:
                    row = _run_one(
                        model=model,
                        scenario_id=scenario_id,
                        max_steps=max_steps,
                        rep_index=rep,
                        dotenv=dotenv,
                        enable_branching_probes=enable_branching_probes,
                        probe_interval=probe_interval,
                        probe_branch_count=args.probe_branches,
                        enable_intervention=bool(args.enable_intervention),
                        validation_interval=max(1, int(args.validation_interval)),
                        stop_on_agent_done=bool(args.stop_on_agent_done),
                    )
                    rows.append(row)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    rows.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "run_id": None,
                            "model_name": model.name,
                            "model": model.model,
                            "scenario_id": scenario_id,
                            "error": str(e),
                            "error_class": _classify_error(e),
                        }
                    )

                pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Benchmark results saved to: {out_path}")


if __name__ == "__main__":
    main()
