import os
import json
import math
import asyncio
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI, APIConnectionError, RateLimitError, APIError
from json import JSONDecodeError
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
                except (APIConnectionError, RateLimitError, APIError) as e:
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
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        
        # Sync client for standard steps (to keep Orchestrator simple)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Async client for parallel probing
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # Some providers/models do not support token logprobs. Default to "auto":
        # try requesting logprobs, and if the provider rejects the parameter, disable for the rest of the run.
        self.logprobs_mode = (os.getenv("REQUEST_LOGPROBS") or "auto").strip().lower()
        self._logprobs_supported: Optional[bool] = None
        
        # Define a clear system message to orient the agent as a task executor
        self.system_message = {
            "role": "system",
            "content": "You are an autonomous AI agent designed to execute tasks by using available tools and responding directly with actions or answers. Your goal is to complete the given task efficiently and accurately. Do not engage in conversational chitchat or ask for clarification unless absolutely necessary for task execution. If you need to perform an action, use the tool functions provided. If the task is complete, provide a final summary."
        }
        
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads a file from the filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "description": "auto (default), full, or outline",
                                "enum": ["auto", "full", "outline"]
                            },
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "with_line_numbers": {"type": "boolean"}
                        },
                        "required": ["path"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Writes content to a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executes a python script.",
                    "parameters": {
                        "type": "object",
                        "properties": {"script_path": {"type": "string"}},
                        "required": ["script_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Searches the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Executes a shell command in the sandbox. Use this for navigating directories, running tests, or managing files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }
        ]

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

    @retry_request(max_retries=3)
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Fetches the next action from the LLM using the Synchronous Client.
        """
        messages = [self.system_message]
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
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "tools": self.tools_schema,
                "tool_choice": "auto",
            }
            if request_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if request_logprobs and (self.logprobs_mode or "auto").strip().lower() == "auto" and self._looks_like_logprobs_unsupported(e):
                    self._logprobs_supported = False
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            
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

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except JSONDecodeError as e:
                    print(f"JSONDecodeError when parsing tool arguments: {e}. Raw arguments: {tool_call.function.arguments}")
                    return {
                        "type": "llm_reply",
                        "content": f"Error: Failed to parse tool arguments as JSON. Malformed JSON: {tool_call.function.arguments[:100]}...",
                        # Logprobs are unavailable for this error path; treat entropy as missing (None).
                        "logprobs": []
                    }
                return {
                    "type": "tool_use",
                    "tool": tool_call.function.name,
                    "content": tool_args,
                    "logprobs": token_logprobs,
                    "usage": usage,
                }
            else:
                return {
                    "type": "llm_reply",
                    "content": message.content,
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
            # We are already in an event loop, use a future to block safely
            # Note: This is tricky in sync code. We might need nest_asyncio or just run_until_complete if we were the owner.
            # But since this function signature is SYNC, and we are called from SYNC Orchestrator,
            # The only case we have a loop is if the Orchestrator itself was wrapped in a loop.
            # We can use new_event_loop() in a separate thread, or just accept the limitation.
            # For now, let's assume if there is a loop, we must use it.
            print("Agent Warning: Existing event loop detected. Using it to run async probe.")
            future = asyncio.ensure_future(self._generate_multiple_async(history, n))
            # We cannot await 'future' here because we are sync.
            # This is the "Sync-Async Bridge" problem.
            # Solution: Use a separate thread to run the async loop if the main thread is blocked?
            # Or simpler: Just run it. 
            # If we are here, likely the user is running `python simulate.py` which is sync.
            # So `asyncio.run` implies NO loop.
            # If there IS a loop (e.g. FastAPI), `asyncio.run` fails.
            pass
        
        # Robust implementation:
        try:
            return asyncio.run(self._generate_multiple_async(history, n))
        except RuntimeError as e:
            if "event loop" in str(e):
                # Fallback: We are in a loop (e.g. Notebook or API).
                # We can try to use the current loop
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._generate_multiple_async(history, n))
            else:
                raise e

    async def _generate_multiple_async(self, history: List[Dict], n: int) -> List[Dict[str, Any]]:

        """
        Internal Async implementation of Branching Probe.
        """
        messages = [self.system_message]
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
            }
            if request_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1

            try:
                response = await self.async_client.chat.completions.create(**kwargs)
            except Exception as e:
                if request_logprobs and (self.logprobs_mode or "auto").strip().lower() == "auto" and self._looks_like_logprobs_unsupported(e):
                    self._logprobs_supported = False
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    response = await self.async_client.chat.completions.create(**kwargs)
                else:
                    raise
            
            choice = response.choices[0]
            content = choice.message.content
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
