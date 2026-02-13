import os
import json
import asyncio
import time
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI, APIConnectionError, RateLimitError, APIError, APITimeoutError
from src.agent.message_utils import (
    build_dynamic_execution_guidance,
    coerce_tool_args,
    content_to_text,
    extract_first_json_object,
    normalize_text_tool_call,
    normalize_tool_name,
    parse_tool_arguments,
)
from src.agent.protocol_config import (
    KNOWN_TOOLS,
    TOOL_NAME_ALIASES,
    build_completion_verifier_system_message,
    build_probe_system_message,
    build_system_message,
    build_text_tools_system_message,
    build_tools_schema,
)
from src.agent.wrapper import AgentWrapper

def retry_request(max_retries=3, backoff_factor=2):
    """
    Decorator for robust API calls.
    Retries on Network/Rate Limit errors.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (APIConnectionError, RateLimitError, APIError, APITimeoutError) as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Agent Error: Max retries exceeded for {func.__name__}. Error: {e}")
                        raise e
                    wait_time = backoff_factor ** retries
                    print(f"Agent Warning: API Error ({e}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                except Exception as e:
                    # Don't retry on logic errors (e.g., Invalid Request)
                    print(f"Agent Error: Unrecoverable error in {func.__name__}: {e}")
                    raise e
        return wrapper
    return decorator

class OpenAICompatibleAgent(AgentWrapper):
    """
    Implementation for OpenAI-compatible APIs (including vLLM).
    Connects to a remote server to generate text and logprobs.
    Includes Async Acceleration for Branching Probes.
    """

    _KNOWN_TOOLS = KNOWN_TOOLS
    _TOOL_NAME_ALIASES = TOOL_NAME_ALIASES
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.base_url = base_url
        try:
            self.request_timeout_seconds = max(10.0, float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS") or "180"))
        except Exception:
            self.request_timeout_seconds = 180.0
        
        # Sync client for standard steps (to keep Orchestrator simple)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.request_timeout_seconds,
        )
        
        # Async client for parallel probing
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.request_timeout_seconds,
        )

        # Some providers/models do not support token logprobs. Default to "auto":
        # try requesting logprobs, and if the provider rejects the parameter, disable for the rest of the run.
        self.logprobs_mode = (os.getenv("REQUEST_LOGPROBS") or "auto").strip().lower()
        self._logprobs_supported: Optional[bool] = None

        # Some providers/models do not support OpenAI-style tools/function calling. Default to "auto":
        # try sending tools/tool_choice, and if the provider rejects them, fall back to text-based tool calls.
        self.tools_mode = (os.getenv("REQUEST_TOOLS") or "auto").strip().lower()
        self._tools_supported: Optional[bool] = None

        # Token caps are important for local/open models that otherwise emit very long outputs.
        # Defaults are conservative for agentic code-writing; override per-run via env vars.
        try:
            self.max_completion_tokens = max(1, int(os.getenv("MAX_COMPLETION_TOKENS") or "1024"))
        except Exception:
            self.max_completion_tokens = 1024
        try:
            self.probe_max_tokens = max(1, int(os.getenv("PROBE_MAX_TOKENS") or "192"))
        except Exception:
            self.probe_max_tokens = 192
        
        self.system_message = build_system_message()
        self.text_tools_system_message = build_text_tools_system_message()
        self.probe_system_message = build_probe_system_message()
        self.completion_verifier_system_message = build_completion_verifier_system_message()
        self.tools_schema = build_tools_schema()

    def _should_request_logprobs(self) -> bool:
        mode = (self.logprobs_mode or "auto").strip().lower()
        if mode in ("0", "false", "no", "off"):
            return False
        if mode in ("1", "true", "yes", "on", "force"):
            return True
        # auto
        return self._logprobs_supported is not False

    def _looks_like_logprobs_unsupported(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "logprobs" in msg or "top_logprobs" in msg

    def _should_request_tools(self) -> bool:
        mode = (self.tools_mode or "auto").strip().lower()
        if mode in ("0", "false", "no", "off"):
            return False
        if mode in ("1", "true", "yes", "on", "force"):
            return True
        # auto
        return self._tools_supported is not False

    def _looks_like_tools_unsupported(self, error: Exception) -> bool:
        msg = str(error).lower()
        if "does not support tools" in msg:
            return True
        if "tool_choice" in msg or "tool choice" in msg:
            return True
        if "tools" in msg and ("unsupported" in msg or "unknown" in msg or "not allowed" in msg):
            return True
        return False

    def _normalize_tool_name(self, tool_name: Any) -> Optional[str]:
        return normalize_tool_name(
            tool_name,
            known_tools=self._KNOWN_TOOLS,
            aliases=self._TOOL_NAME_ALIASES,
        )

    def _coerce_tool_args(self, tool_name: str, args: Any) -> Optional[Dict[str, Any]]:
        return coerce_tool_args(tool_name, args)

    def _parse_tool_arguments(self, raw_arguments: Any, tool_name: str) -> Optional[Dict[str, Any]]:
        return parse_tool_arguments(raw_arguments, tool_name)

    def _normalize_text_tool_call(self, obj: Any) -> Optional[Dict[str, Any]]:
        return normalize_text_tool_call(
            obj,
            known_tools=self._KNOWN_TOOLS,
            aliases=self._TOOL_NAME_ALIASES,
        )

    def _content_to_text(self, content: Any) -> str:
        return content_to_text(content)

    def _extract_first_json_object(self, text: str) -> Optional[Any]:
        return extract_first_json_object(text)

    def _build_dynamic_execution_guidance(self, history: List[Dict]) -> Optional[Dict[str, str]]:
        return build_dynamic_execution_guidance(history)

    @retry_request(max_retries=3)
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Fetches the next action from the LLM using the Synchronous Client.
        """
        request_tools = self._should_request_tools()
        system_message = self.system_message if request_tools else self.text_tools_system_message
        messages = [system_message]
        dynamic_hint = self._build_dynamic_execution_guidance(history)
        if dynamic_hint is not None:
            messages.append(dynamic_hint)
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "tool_output":
                role = "user" # Map tool output to user for visibility
                content = f"Tool Output: {content}"
            
            if role in ["system", "user", "assistant"]:
                messages.append({"role": role, "content": content})

        if not messages:
            messages = [{"role": "user", "content": "Begin the task."}]

        try:
            request_logprobs = self._should_request_logprobs()
            request_tools = self._should_request_tools()
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_completion_tokens,
            }

            def apply_optional_params() -> None:
                if request_tools:
                    kwargs["tools"] = self.tools_schema
                    kwargs["tool_choice"] = "auto"
                else:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)

                if request_logprobs:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = 1
                else:
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)

            apply_optional_params()

            response = None
            for _ in range(3):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    if (
                        request_logprobs
                        and (self.logprobs_mode or "auto").strip().lower() == "auto"
                        and self._looks_like_logprobs_unsupported(e)
                    ):
                        self._logprobs_supported = False
                        request_logprobs = False
                        apply_optional_params()
                        continue

                    if (
                        request_tools
                        and (self.tools_mode or "auto").strip().lower() == "auto"
                        and self._looks_like_tools_unsupported(e)
                    ):
                        self._tools_supported = False
                        request_tools = False

                        # Rebuild messages with the text-tools protocol and retry without `tools`.
                        messages = [self.text_tools_system_message]
                        for msg in history:
                            role = msg["role"]
                            content = msg["content"]
                            if role == "tool_output":
                                role = "user"
                                content = f"Tool Output: {content}"
                            if role in ["system", "user", "assistant"]:
                                messages.append({"role": role, "content": content})
                        kwargs["messages"] = messages
                        apply_optional_params()
                        continue

                    raise

            if response is None:
                raise RuntimeError("Failed to obtain a response from the model after retries.")
            
            choice = response.choices[0]
            message = choice.message

            usage = {}
            try:
                if getattr(response, "usage", None):
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
            except Exception:
                usage = {}
            
            token_logprobs = []
            if choice.logprobs and choice.logprobs.content:
                token_logprobs = [token.logprob for token in choice.logprobs.content]

            if getattr(message, "tool_calls", None):
                tool_call = message.tool_calls[0]
                tool_name = self._normalize_tool_name(getattr(tool_call.function, "name", None))
                if not tool_name:
                    return {
                        "type": "llm_reply",
                        "content": "Error: Model returned a tool call without a valid tool name.",
                        "logprobs": [],
                    }
                tool_args = self._parse_tool_arguments(getattr(tool_call.function, "arguments", ""), tool_name)
                if tool_args is None:
                    raw_args = str(getattr(tool_call.function, "arguments", ""))[:180]
                    print(f"Tool argument parsing failed for tool={tool_name}. Raw={raw_args}")
                    return {
                        "type": "llm_reply",
                        "content": f"Error: Failed to parse tool arguments for tool '{tool_name}'.",
                        "logprobs": [],
                    }
                return {
                    "type": "tool_use",
                    "tool": tool_name,
                    "content": tool_args,
                    "logprobs": token_logprobs,
                    "usage": usage,
                }
            else:
                # Text-tools fallback (no native tool_calls).
                content_text = self._content_to_text(getattr(message, "content", ""))
                if content_text:
                    # Try full text first so patterns like
                    # "Used tool: run_shell with args: {...}" are preserved.
                    normalized = self._normalize_text_tool_call(content_text)
                    if normalized is None:
                        extracted = self._extract_first_json_object(content_text)
                        normalized = self._normalize_text_tool_call(extracted)
                    if normalized and normalized.get("type") == "tool_use":
                        normalized["logprobs"] = token_logprobs
                        normalized["usage"] = usage
                        return normalized
                    if normalized and normalized.get("type") == "llm_reply" and not request_tools:
                        normalized["logprobs"] = token_logprobs
                        normalized["usage"] = usage
                        return normalized

                return {
                    "type": "llm_reply",
                    "content": content_text,
                    "logprobs": token_logprobs,
                    "usage": usage,
                }

        except Exception as e:
            # Re-raise to let retry_request handle it, or catch if logic error
            if isinstance(e, (APIConnectionError, RateLimitError, APIError)):
                raise e
            print(f"Error calling LLM: {e}")
            # Logprobs are unavailable for this error path; treat entropy as missing (None).
            return {"type": "llm_reply", "content": f"Error: {str(e)}", "logprobs": []}

    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        """
        Generates N divergent responses for Branching Probe in PARALLEL.
        SAFE: Handles existing event loops to avoid 'RuntimeError: asyncio.run() cannot be called from a running event loop'.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return self._run_async_in_worker_thread(history, n)
        return asyncio.run(self._generate_multiple_async(history, n))

    def _run_async_in_worker_thread(self, history: List[Dict], n: int) -> List[Dict[str, Any]]:
        result: Dict[str, Any] = {"branches": [], "error": None}

        def _runner() -> None:
            try:
                result["branches"] = asyncio.run(self._generate_multiple_async(history, n))
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if result["error"] is not None:
            raise result["error"]
        return result["branches"]

    async def _generate_multiple_async(self, history: List[Dict], n: int) -> List[Dict[str, Any]]:

        """
        Internal Async implementation of Branching Probe.
        """
        messages = [self.probe_system_message]
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "tool_output":
                role = "user" 
                content = f"Tool Output: {content}"
            
            if role in ["system", "user", "assistant"]:
                messages.append({"role": role, "content": content})
        
        # Create N parallel tasks
        tasks = [self._generate_one_async(messages, i, n) for i in range(n)]
        
        # Wait for all to complete
        branches = await asyncio.gather(*tasks)
        
        # Filter out Nones (failed attempts)
        valid_branches = [b for b in branches if b is not None]
        
        # If all failed, return empty list
        return valid_branches

    async def _generate_one_async(self, messages: List[Dict], index: int, total: int) -> Optional[Dict[str, Any]]:
        """
        Helper for a single async generation.
        """
        try:
            request_logprobs = self._should_request_logprobs()
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.9,  # High temp for divergence
                "n": 1,
                "max_tokens": self.probe_max_tokens,
            }
            if request_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1

            response = None
            attempts = 0
            while attempts < 3:
                attempts += 1
                try:
                    response = await self.async_client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    if (
                        request_logprobs
                        and (self.logprobs_mode or "auto").strip().lower() == "auto"
                        and self._looks_like_logprobs_unsupported(e)
                    ):
                        self._logprobs_supported = False
                        request_logprobs = False
                        kwargs.pop("logprobs", None)
                        kwargs.pop("top_logprobs", None)
                        continue
                    if isinstance(e, (APIConnectionError, RateLimitError, APIError)) and attempts < 3:
                        await asyncio.sleep(2 ** attempts)
                        continue
                    raise

            if response is None:
                raise RuntimeError("Probe request failed after retries.")
            
            choice = response.choices[0]
            message = choice.message
            content = self._content_to_text(getattr(message, "content", ""))
            if not content.strip():
                # Some providers return probe outputs as tool-calls with empty text.
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    call_summaries: List[str] = []
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "") if fn is not None else ""
                        args = getattr(fn, "arguments", "") if fn is not None else ""
                        payload = {"tool": str(name)}
                        if args:
                            payload["args"] = str(args)
                        call_summaries.append(json.dumps(payload, ensure_ascii=True))
                    content = "\n".join(call_summaries).strip()

            if not content.strip():
                # Fallback for providers that expose reasoning separate from message.content.
                reasoning = getattr(message, "reasoning", None)
                if reasoning:
                    content = self._content_to_text(reasoning)
            token_logprobs = []
            if choice.logprobs and choice.logprobs.content:
                token_logprobs = [token.logprob for token in choice.logprobs.content]

            usage = {}
            try:
                if getattr(response, "usage", None):
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
            except Exception:
                usage = {}
            
            return {
                "type": "thought",
                "content": content,
                "logprobs": token_logprobs,
                "usage": usage,
            }
        except Exception as e:
            print(f"Error in async probe {index+1}/{total}: {e}")
            return None

    def assess_completion(
        self,
        history: List[Dict],
        scenario_id: str = "",
        ground_truth_goal: str = "",
        sandbox_path: str = "",
        trigger: str = "periodic",
    ) -> Dict[str, Any]:
        """
        Independent completion verifier call.
        Returns: {"verdict": "done|not_done|uncertain", "confidence": float, "reason": str}
        """
        compact_history = []
        for msg in history[-30:]:
            role = str(msg.get("role", "unknown"))
            content = self._content_to_text(msg.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "... <truncated>"
            compact_history.append({"role": role, "content": content})

        verifier_payload = {
            "scenario_id": scenario_id,
            "ground_truth_goal": ground_truth_goal,
            "sandbox_path": sandbox_path,
            "trigger": trigger,
            "recent_history": compact_history,
        }

        messages = [
            self.completion_verifier_system_message,
            {"role": "user", "content": json.dumps(verifier_payload, ensure_ascii=False)},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=220,
            )
            choice = response.choices[0]
            content_text = self._content_to_text(getattr(choice.message, "content", ""))
            parsed = self._extract_first_json_object(content_text)
            if not isinstance(parsed, dict):
                lowered = content_text.lower()
                if "not_done" in lowered:
                    parsed = {"verdict": "not_done", "confidence": 0.5, "reason": content_text[:220]}
                elif "\"done\"" in lowered or "verdict: done" in lowered:
                    parsed = {"verdict": "done", "confidence": 0.5, "reason": content_text[:220]}
                else:
                    parsed = {"verdict": "uncertain", "confidence": 0.0, "reason": content_text[:220]}

            verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
            if verdict not in ("done", "not_done", "uncertain"):
                verdict = "uncertain"
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = min(1.0, max(0.0, confidence))
            reason = str(parsed.get("reason", "")).strip() or "no reason provided"
            return {"verdict": verdict, "confidence": confidence, "reason": reason}
        except Exception as e:
            return {"verdict": "uncertain", "confidence": 0.0, "reason": f"verifier_error: {e}"}
