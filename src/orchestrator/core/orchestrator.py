import os
import uuid
from typing import Any, Dict, List, Optional

from src.connectors.local_connect import LocalSandboxConnector
from src.interfaces import AgentProtocol, MetricServiceProtocol
from src.scenarios.definitions import SCENARIOS
from src.scenarios.perturbation_ops import apply_perturbation_if_needed
from src.scenarios.setup_ops import SCENARIO_SETUP_MAP
from src.scenarios.validation_ops import validate_scenario
from src.security.secrets import SecretScanner
from src.services.metrics import EmbeddingMetricService
from src.shared.constants import RUN_ARTIFACTS_DIR
from src.shared.run_manifest import build_manifest, write_json
from src.tools.registry import ToolRegistry

from .history import action_signature, format_tool_args_for_history
from .panic import update_entropy_panic, update_loop_panic
from .probes import run_branching_probe
from .step_state import (
    build_default_step_metrics,
    build_task_complete_payload,
    check_perturbation_trigger,
    compose_validation_details_for_log,
)
from .tools import execute_tool_and_measure
from .verifier import (
    agent_signaled_completion,
    normalize_ai_verifier_result,
    run_ai_verifier,
)


class Orchestrator:
    """
    Module 1: The Orchestrator (The Controller) - v2.2 Refactor

    Decoupled architecture using Protocols and Tool Registry.
    """

    def __init__(
        self,
        scenario_id: str,
        agent: AgentProtocol,
        metric_service: Optional[MetricServiceProtocol] = None,
        entropy_mean: float = None,
        entropy_std: float = None,
        connector: Any = None,
        run_id: str = None,
        metrics_monitor: Any = None,
        enable_intervention: bool = True,
        probe_interval: int = 0,
        enable_validation: bool = False,
        validation_interval: int = 1,
        stop_on_success: bool = False,
        stop_on_agent_done: bool = False,
        enable_ai_verifier: bool = False,
        ai_verifier_interval: int = 0,
        ai_verifier_confidence_threshold: float = 0.8,
        enable_branching_probes: bool = True,
        probe_branch_count: int = 5,
    ):
        self.scenario_id = scenario_id
        self.scenario = self._load_scenario(scenario_id)
        if not self.scenario:
            raise ValueError(f"Scenario with ID '{scenario_id}' not found.")

        self.run_id = run_id or str(uuid.uuid4())

        # --- ENVIRONMENT SETUP ---
        project_root = os.path.abspath(os.getcwd())
        backend = (os.getenv("SANDBOX_BACKEND") or "auto").strip().lower()
        sandbox_per_run = (os.getenv("SANDBOX_PER_RUN") or "0").strip().lower() in ("1", "true", "yes", "on")
        sandbox_dirname = f"sandbox_{scenario_id}"
        if sandbox_per_run and backend in ("local", "host"):
            sandbox_dirname = f"sandbox_{scenario_id}_{self.run_id}"
        sandbox_root = os.getenv("EXPERIMENT_SANDBOX_ROOT")
        if sandbox_root:
            sandbox_root = os.path.abspath(sandbox_root)
        else:
            sandbox_root = os.path.join(project_root, "data")
        self.sandbox_path = os.path.join(sandbox_root, sandbox_dirname)

        if scenario_id in SCENARIO_SETUP_MAP:
            print(f"Orchestrator: Running environment setup for {scenario_id}...")
            SCENARIO_SETUP_MAP[scenario_id](self.sandbox_path)
        else:
            os.makedirs(self.sandbox_path, exist_ok=True)

        self.agent = agent
        self.metrics_monitor = metrics_monitor
        self.enable_intervention = enable_intervention
        self.probe_interval = probe_interval
        self.enable_validation = enable_validation
        self.validation_interval = max(1, int(validation_interval))
        self.stop_on_success = stop_on_success
        self.stop_on_agent_done = bool(stop_on_agent_done)
        self.enable_ai_verifier = bool(
            enable_ai_verifier
            or (os.getenv("AI_VERIFIER") or "off").strip().lower() in ("1", "true", "yes", "on")
        )
        env_ai_interval = os.getenv("AI_VERIFIER_INTERVAL")
        if env_ai_interval is not None:
            try:
                ai_verifier_interval = int(env_ai_interval)
            except Exception:
                ai_verifier_interval = ai_verifier_interval
        self.ai_verifier_interval = max(0, int(ai_verifier_interval))
        env_ai_conf = os.getenv("AI_VERIFIER_CONFIDENCE")
        if env_ai_conf is not None:
            try:
                ai_verifier_confidence_threshold = float(env_ai_conf)
            except Exception:
                ai_verifier_confidence_threshold = ai_verifier_confidence_threshold
        self.ai_verifier_confidence_threshold = min(1.0, max(0.0, float(ai_verifier_confidence_threshold)))
        self.ai_verifier_feedback = (os.getenv("AI_VERIFIER_FEEDBACK") or "on").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.validation_feedback = (os.getenv("VALIDATION_FEEDBACK") or "off").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.enable_branching_probes = enable_branching_probes
        self.probe_branch_count = max(2, int(probe_branch_count))

        # Dependency Injection: Metric Service
        if metric_service:
            self.metric_service = metric_service
        else:
            print("Orchestrator: No metric_service provided, instantiating default EmbeddingMetricService.")
            self.metric_service = EmbeddingMetricService()

        # Dependency Injection: Tool Registry
        self.tool_registry = ToolRegistry()

        self.step_count = 0
        self.history: List[Dict] = []
        self.panic_counter = 0
        self.panic_threshold = 3
        self.entropy_threshold = 0.8

        self.entropy_mean = entropy_mean
        self.entropy_std = entropy_std
        self.z_score_threshold = 2.0

        self.last_tool_context: Optional[Dict] = None
        self.task_complete = False

        # Loop detection fallback (for providers without logprobs/entropy).
        self._prev_action_signature: Optional[str] = None

        # Initialize Sandbox Connector
        if connector:
            self.connector = connector
        else:
            if backend in ("local", "host"):
                print("Initializing Local Sandbox (no Docker)...")
                self.connector = LocalSandboxConnector(self.sandbox_path)
                self.connector.start()
            else:
                # Default/auto: prefer Docker, but fall back to local if Docker is unavailable.
                try:
                    from src.connectors.tb_connect import TerminalBenchConnector  # lazy import (docker optional)

                    print("Initializing TerminalBench Sandbox (Docker)...")
                    self.connector = TerminalBenchConnector(
                        scenario_id,
                        host_data_path=self.sandbox_path,
                        run_id=self.run_id,
                    )
                    self.connector.start()
                except Exception as e:
                    if backend in ("docker", "terminalbench"):
                        raise
                    print(f"WARNING: Docker sandbox unavailable ({e}). Falling back to LocalSandboxConnector.")
                    self.connector = LocalSandboxConnector(self.sandbox_path)
                    self.connector.start()

        # RDI Series Tracking
        self.rdi_series: List[Optional[float]] = []
        self.recovered_at_step: Optional[int] = None
        self.stability_counter = 0
        self.recovery_threshold = 2

        # Determine Ground Truth for RDI (Text only, embedding happens in service)
        self.ground_truth_text = self.scenario.get("ground_truth_goal") or self.scenario.get("initial_prompt", "")

        # Initialize History with Prompt
        if self.scenario.get("initial_prompt"):
            self.history.append({"role": "user", "content": self.scenario.get("initial_prompt")})

        # Run artifacts (manifest, summaries, plots) live under data/run_artifacts/<run_id>/.
        self.run_dir = os.path.join(RUN_ARTIFACTS_DIR, self.run_id)
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            log_file = getattr(self.metrics_monitor, "log_file", None) if self.metrics_monitor else None
            write_json(
                os.path.join(self.run_dir, "manifest.json"),
                build_manifest(
                    run_id=self.run_id,
                    scenario_id=self.scenario_id,
                    model_name=getattr(self.agent, "model_name", "unknown"),
                    sandbox_path=self.sandbox_path,
                    log_file=log_file,
                    config={
                        "enable_intervention": self.enable_intervention,
                        "probe_interval": self.probe_interval,
                        "enable_validation": self.enable_validation,
                        "validation_interval": self.validation_interval,
                        "stop_on_success": self.stop_on_success,
                        "stop_on_agent_done": self.stop_on_agent_done,
                        "enable_ai_verifier": self.enable_ai_verifier,
                        "ai_verifier_interval": self.ai_verifier_interval,
                        "ai_verifier_confidence_threshold": self.ai_verifier_confidence_threshold,
                        "enable_branching_probes": self.enable_branching_probes,
                        "probe_branch_count": self.probe_branch_count,
                        "agent_base_url": getattr(self.agent, "base_url", None),
                        "metric_embedding_backend": getattr(self.metric_service, "embedding_backend", None),
                        "metric_embedding_model": getattr(self.metric_service, "model_name", None),
                        "metric_embedding_device": getattr(self.metric_service, "device", None),
                        "metric_local_files_only": getattr(self.metric_service, "local_files_only", None),
                        "metric_hash_dim": getattr(self.metric_service, "hash_dim", None),
                    },
                ),
            )
        except Exception as e:
            print(f"Orchestrator Warning: Failed to write run manifest: {e}")

    def switch_agent(self, new_agent: AgentProtocol):
        """Swaps the current agent."""
        print(
            f"Orchestrator: Switching agent from {getattr(self.agent, 'model_name', 'Unknown')} "
            f"to {getattr(new_agent, 'model_name', 'Unknown')}."
        )
        self.agent = new_agent
        self.panic_counter = 0

    def _load_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s.model_dump()
        return None

    def compute_drift_summary(self) -> Dict[str, Any]:
        valid_rdi = [x for x in self.rdi_series if x is not None]
        max_drift = max(valid_rdi) if valid_rdi else 0.0
        drift_auc = sum(valid_rdi)

        post_recovery_mean = None
        if self.recovered_at_step is not None and self.recovered_at_step <= len(self.rdi_series):
            start_idx = self.recovered_at_step - 1
            post_series = [x for x in self.rdi_series[start_idx:] if x is not None]
            if post_series:
                post_recovery_mean = sum(post_series) / len(post_series)

        return {
            "max_drift": max_drift,
            "drift_auc": drift_auc,
            "recovered_at_step": self.recovered_at_step,
            "post_recovery_drift_mean": post_recovery_mean,
        }

    def _agent_signaled_completion(self, content: Any) -> bool:
        return agent_signaled_completion(content)

    def _agent_supports_ai_verifier(self) -> bool:
        return callable(getattr(self.agent, "assess_completion", None))

    def _normalize_ai_verifier_result(self, raw: Any) -> Dict[str, Any]:
        return normalize_ai_verifier_result(raw)

    def _run_ai_verifier(self, trigger: str) -> Dict[str, Any]:
        return run_ai_verifier(
            assess_func=getattr(self.agent, "assess_completion", None),
            history=self.history,
            scenario_id=self.scenario_id,
            ground_truth_goal=self.ground_truth_text,
            sandbox_path=self.sandbox_path,
            trigger=trigger,
        )

    def step(self) -> Dict:
        """
        Advances the simulation by one step.
        """
        if self.task_complete:
            return build_task_complete_payload(
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                step_index=self.step_count,
                model_name=getattr(self.agent, "model_name", "unknown"),
            )

        self.step_count += 1

        step_metrics = build_default_step_metrics(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            step_index=self.step_count,
            model_name=getattr(self.agent, "model_name", "unknown"),
            panic_counter=self.panic_counter,
        )

        # 1. Check for Periodic SCR Probe (Silent)
        if self.enable_branching_probes and self.probe_interval > 0 and self.step_count % self.probe_interval == 0:
            # Only probe if no perturbation is about to happen (priority to perturbation)
            if not self._check_perturbation_triggers():
                probe_results = run_branching_probe(
                    agent=self.agent,
                    metric_service=self.metric_service,
                    history=self.history,
                    probe_branch_count=self.probe_branch_count,
                    perturbation_instruction=None,
                )
                if self.metrics_monitor:
                    self.metrics_monitor.log_step(
                        run_id=self.run_id,
                        scenario_id=self.scenario_id,
                        model_name=step_metrics["model"],
                        step_index=self.step_count,
                        event_type="periodic_probe",
                        prompt="<silent_probe>",
                        scr=probe_results.get("scr"),
                        branches_count=len(probe_results.get("branches") or []),
                        panic_counter=self.panic_counter,
                    )

        # 2. Check for Perturbations (Scenario-driven)
        perturbation_instruction = self._check_perturbation_triggers()
        if perturbation_instruction:
            self.recovered_at_step = None
            self.stability_counter = 0

            step_metrics["event_type"] = "perturbation_triggered"

            # Apply any sandbox mutation that corresponds to this perturbation.
            apply_perturbation_if_needed(self.scenario_id, self.step_count, self.sandbox_path)

            if self.enable_branching_probes:
                probe_results = run_branching_probe(
                    agent=self.agent,
                    metric_service=self.metric_service,
                    history=self.history,
                    probe_branch_count=self.probe_branch_count,
                    perturbation_instruction=perturbation_instruction,
                )
                step_metrics["scr"] = probe_results["scr"]
            else:
                probe_results = {"scr": None, "branches": []}

            # Persist the requirement change so the agent sees it on subsequent steps.
            self.history.append({"role": "user", "content": perturbation_instruction})

            self.rdi_series.append(None)

            if self.metrics_monitor:
                self.metrics_monitor.log_step(
                    run_id=self.run_id,
                    scenario_id=self.scenario_id,
                    model_name=step_metrics["model"],
                    step_index=self.step_count,
                    event_type="perturbation_triggered",
                    prompt=perturbation_instruction,
                    scr=step_metrics["scr"],
                    branches_count=len(probe_results.get("branches") or []),
                    panic_counter=self.panic_counter,
                )

            return {
                **step_metrics,
                **{"type": "perturbation_triggered", "perturbation": perturbation_instruction, "probe_metrics": probe_results},
            }

        # Get Agent's next action
        agent_action_intent = self.agent.get_next_action(self.history)

        usage = agent_action_intent.get("usage")
        if isinstance(usage, dict):
            step_metrics["prompt_tokens"] = usage.get("prompt_tokens")
            step_metrics["completion_tokens"] = usage.get("completion_tokens")
            step_metrics["total_tokens"] = usage.get("total_tokens")

        # Calculate Entropy via Service
        current_entropy = self.metric_service.calculate_entropy(agent_action_intent.get("logprobs", []))
        step_metrics["current_entropy"] = current_entropy

        # IGE Calculation
        if self.last_tool_context:
            h_pre = self.last_tool_context["h_pre"]
            token_cost = self.last_tool_context["token_cost"]
            if h_pre is not None and current_entropy is not None:
                step_metrics["ige"] = self.metric_service.calculate_ige(h_pre, current_entropy, token_cost)
            self.last_tool_context = None

        # Calculate RDI via Service
        current_content = agent_action_intent.get("content", "")
        if not isinstance(current_content, str):
            current_content = str(current_content)

        step_metrics["rdi"] = self.metric_service.calculate_rdi(current_content, self.ground_truth_text)
        self.rdi_series.append(step_metrics["rdi"])

        # Loop detection (string-stable action signature).
        sig = action_signature(agent_action_intent)
        loop_repeat = bool(sig and sig == self._prev_action_signature)
        self._prev_action_signature = sig

        # Intervention Check
        panic_triggered = False
        panic_reason = None
        if current_entropy is not None:
            self.panic_counter, panic_triggered = update_entropy_panic(
                panic_counter=self.panic_counter,
                entropy=current_entropy,
                panic_threshold=self.panic_threshold,
                entropy_threshold=self.entropy_threshold,
                entropy_mean=self.entropy_mean,
                entropy_std=self.entropy_std,
                z_score_threshold=self.z_score_threshold,
            )
            panic_reason = "persistent_panic"
        else:
            self.panic_counter, panic_triggered = update_loop_panic(
                panic_counter=self.panic_counter,
                loop_repeat=loop_repeat,
                panic_threshold=self.panic_threshold,
            )
            if panic_triggered:
                panic_reason = "persistent_loop"

        if panic_triggered:
            step_metrics["event_type"] = "panic_detected"  # Log it even if we don't intervene
            step_metrics["panic_counter"] = self.panic_counter
            self.recovered_at_step = None
            self.stability_counter = 0

            if self.metrics_monitor:
                self.metrics_monitor.log_step(
                    run_id=self.run_id,
                    scenario_id=self.scenario_id,
                    model_name=step_metrics["model"],
                    step_index=self.step_count,
                    event_type="panic_detected",
                    current_entropy=current_entropy,
                    panic_counter=self.panic_counter,
                    rdi=step_metrics["rdi"],
                )

            if self.enable_intervention:
                step_metrics["event_type"] = "intervention"
                self.intervene()
                return {**step_metrics, **{"type": "intervention", "reason": panic_reason or "persistent_panic", "step": self.step_count}}

        step_metrics["panic_counter"] = self.panic_counter

        if self.panic_counter == 0:
            self.stability_counter += 1
        else:
            self.stability_counter = 0

        if self.stability_counter >= self.recovery_threshold and self.recovered_at_step is None:
            self.recovered_at_step = self.step_count

        # Handle Agent's Intent
        tool_result = None
        tool_name = None
        agent_done_claimed = False
        if agent_action_intent["type"] == "tool_use":
            tool_name = agent_action_intent["tool"]
            tool_args = agent_action_intent["content"]
            step_metrics["tool"] = tool_name

            token_count = None
            if isinstance(usage, dict):
                token_count = usage.get("completion_tokens") or usage.get("total_tokens")
            if not isinstance(token_count, int) or token_count <= 0:
                token_count = len(agent_action_intent.get("logprobs", []))
            if token_count == 0:
                token_count = len(str(tool_args)) // 4 + 1
            self.last_tool_context = {"h_pre": current_entropy, "token_cost": token_count}

            tool_result, cbf_value, _ = execute_tool_and_measure(
                tool_registry=self.tool_registry,
                connector=self.connector,
                action_intent=agent_action_intent,
            )
            step_metrics["event_type"] = "tool_execution"
            step_metrics["cbf"] = cbf_value

            safe_args = format_tool_args_for_history(tool_name, tool_args)
            # Record the tool call as an assistant action to keep dialogue roles coherent.
            self.history.append({"role": "assistant", "content": f"Used tool: {tool_name} with args: {safe_args}"})
            self.history.append({"role": "tool_output", "content": SecretScanner.redact(str(tool_result))})

        elif agent_action_intent["type"] == "llm_reply":
            step_metrics["event_type"] = "llm_reply"
            step_metrics["compression_ratio"] = self.metric_service.calculate_compression_ratio(agent_action_intent["content"])
            self.history.append({"role": "assistant", "content": agent_action_intent["content"]})
            if self._agent_signaled_completion(agent_action_intent["content"]):
                agent_done_claimed = True
                step_metrics["agent_done_claimed"] = True
        else:
            step_metrics["event_type"] = "unknown_action"

        # Optional scenario validation (benchmark-style ground truth).
        if self.enable_validation and (self.step_count % self.validation_interval == 0):
            validation = validate_scenario(self.scenario_id, self.sandbox_path)
            if validation is not None:
                step_metrics["validation_passed"] = bool(validation.passed)
                step_metrics["validation_score"] = validation.score
                step_metrics["validation_details"] = validation.details
                if self.validation_feedback and not validation.passed:
                    details = validation.details or "Validation failed."
                    if len(details) > 800:
                        details = details[:800] + "... <truncated>"
                    self.history.append({"role": "user", "content": f"VALIDATION FAILED: {details}"})
                if validation.passed and self.stop_on_success:
                    step_metrics["task_complete"] = True
                    self.task_complete = True

        ai_verifier_trigger = None
        if self.enable_ai_verifier:
            if agent_done_claimed:
                ai_verifier_trigger = "done_claim"
            elif self.ai_verifier_interval > 0 and self.step_count % self.ai_verifier_interval == 0:
                ai_verifier_trigger = "periodic"

        if ai_verifier_trigger and not self.task_complete:
            if self._agent_supports_ai_verifier():
                verdict = self._run_ai_verifier(trigger=ai_verifier_trigger)
                step_metrics["ai_verifier_verdict"] = verdict["verdict"]
                step_metrics["ai_verifier_confidence"] = verdict["confidence"]
                step_metrics["ai_verifier_reason"] = verdict["reason"]

                if verdict["verdict"] == "done" and verdict["confidence"] >= self.ai_verifier_confidence_threshold:
                    step_metrics["task_complete"] = True
                    self.task_complete = True
                elif (
                    agent_done_claimed
                    and self.ai_verifier_feedback
                    and verdict["verdict"] in ("not_done", "uncertain")
                ):
                    reason = verdict["reason"][:400]
                    self.history.append({"role": "user", "content": f"VERIFIER: task not complete yet. {reason}"})
            elif agent_done_claimed:
                # Preserve legacy stop behavior if verifier is enabled but unsupported by the active agent.
                step_metrics["task_complete"] = True
                self.task_complete = True

        if agent_done_claimed and self.stop_on_agent_done and not self.task_complete and not self.enable_ai_verifier:
            step_metrics["task_complete"] = True
            self.task_complete = True

        # Logging
        if self.metrics_monitor:
            self.metrics_monitor.log_step(
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                model_name=step_metrics["model"],
                step_index=self.step_count,
                event_type=step_metrics["event_type"],
                prompt=str(self.history[-1]["content"]) if self.history else "",
                current_entropy=step_metrics["current_entropy"],
                ige=step_metrics["ige"],
                scr=step_metrics["scr"],
                cbf=step_metrics["cbf"],
                rdi=step_metrics["rdi"],
                panic_counter=self.panic_counter,
                tool=step_metrics["tool"],
                compression_ratio=step_metrics["compression_ratio"],
                task_complete=step_metrics["task_complete"],
                agent_done_claimed=step_metrics["agent_done_claimed"],
                validation_passed=step_metrics["validation_passed"],
                validation_score=step_metrics["validation_score"],
                validation_details=self._compose_validation_details_for_log(step_metrics),
                ai_verifier_verdict=step_metrics["ai_verifier_verdict"],
                ai_verifier_confidence=step_metrics["ai_verifier_confidence"],
                prompt_tokens=step_metrics["prompt_tokens"],
                completion_tokens=step_metrics["completion_tokens"],
                total_tokens=step_metrics["total_tokens"],
                branching_func=None,
            )

        if tool_result:
            return {**step_metrics, **{"type": "tool_execution", "tool": tool_name, "result": tool_result}}
        if step_metrics["event_type"] == "llm_reply":
            return {**step_metrics, **{"type": "llm_reply", "content": agent_action_intent["content"]}}
        return {**step_metrics, **{"type": "unknown_action"}}

    def _compose_validation_details_for_log(self, step_metrics: Dict[str, Any]) -> Optional[str]:
        return compose_validation_details_for_log(step_metrics)

    def _check_perturbation_triggers(self) -> Optional[str]:
        return check_perturbation_trigger(self.scenario, self.step_count)

    def intervene(self):
        print(f"Intervention Triggered at step {self.step_count} due to persistent panic!")
        self.panic_counter = 0
        self.history.append(
            {
                "role": "system",
                "content": "Intervention: You seem stuck. Please reassess your goal and try a different approach.",
            }
        )
