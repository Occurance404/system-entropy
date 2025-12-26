import os
import uvicorn
import json
import math
import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import dotenv_values

# Import our monitor
from src.monitor.terminal_bench_monitor import get_monitor

# --- Configuration & Classes ---

class DynamicConfig:
    """Manages runtime configuration for shock injection."""
    def __init__(self):
        self.shock_step = int(os.environ.get("SHOCK_TRIGGER_STEP", -1))
        self.shock_message = os.environ.get("SHOCK_MESSAGE", "")
    
    def update(self, step: int = None, message: str = None):
        if step is not None: self.shock_step = step
        if message is not None: self.shock_message = message
    
    def to_dict(self):
        return {"shock_step": self.shock_step, "shock_message": self.shock_message}

class SimpleRateLimiter:
    """Basic in-memory sliding window rate limiter."""
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def check_limit(self) -> bool:
        now = time.time()
        # Remove old requests
        self.requests = [req for req in self.requests if req > now - self.window_seconds]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True

# --- FastAPI App Setup ---
app = FastAPI(title="LLM Proxy with Metric Injection")
monitor = get_monitor()
config_store = DynamicConfig()
rate_limiter = SimpleRateLimiter(max_requests=100, window_seconds=60) # 100 RPM default
PROXY_RUN_ID = os.getenv("TB_RUN_ID") or str(uuid.uuid4())
CHEAP_MODE = (os.getenv("CHEAP_MODE") or "").strip().lower() in ("1", "true", "yes", "on")
PROXY_SCR_MODE = (os.getenv("PROXY_SCR_MODE") or ("off" if CHEAP_MODE else "shock")).strip().lower()
PROXY_REQUEST_LOGPROBS = (os.getenv("PROXY_REQUEST_LOGPROBS") or "auto").strip().lower()

# Load real API config from .env
config = dotenv_values(".env")
REAL_VLLM_API_KEY = config.get("VLLM_API_KEY")
REAL_VLLM_BASE_URL = config.get("VLLM_BASE_URL")
REAL_VLLM_MODEL_NAME = config.get("VLLM_MODEL_NAME", "deepseek-chat")
PROXY_AUTH_TOKEN = config.get("PROXY_AUTH_TOKEN", "dev-secret")

if not REAL_VLLM_API_KEY or not REAL_VLLM_BASE_URL:
    print("WARNING: Real LLM API credentials not fully set in .env. Proxy might fail.")

# Initialize OpenAI client
try:
    openai_client = OpenAI(
        api_key=REAL_VLLM_API_KEY,
        base_url=REAL_VLLM_BASE_URL,
    )
except Exception as e:
    print(f"ERROR: Could not initialize OpenAI client in proxy: {e}")
    openai_client = None

_STEP_LOCK = Lock()
_STEP_BY_RUN_ID: Dict[str, int] = defaultdict(int)
_LOGPROBS_LOCK = Lock()
_LOGPROBS_SUPPORTED: Optional[bool] = None


def _next_step_index(run_id: str) -> int:
    with _STEP_LOCK:
        _STEP_BY_RUN_ID[run_id] += 1
        return _STEP_BY_RUN_ID[run_id]

print(f"Proxy configured for real LLM: {REAL_VLLM_MODEL_NAME} at {REAL_VLLM_BASE_URL}")
print(f"Initial Shock Config: {config_store.to_dict()}")
print(f"Proxy Run ID: {PROXY_RUN_ID}")
print(f"Proxy SCR Mode: {PROXY_SCR_MODE}")
print(f"Proxy Logprobs Mode: {PROXY_REQUEST_LOGPROBS}")


def _should_request_logprobs() -> bool:
    mode = (PROXY_REQUEST_LOGPROBS or "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on", "force"):
        return True
    # auto
    with _LOGPROBS_LOCK:
        return _LOGPROBS_SUPPORTED is not False


def _looks_like_logprobs_unsupported(error: Exception) -> bool:
    msg = str(error).lower()
    return "logprobs" in msg or "top_logprobs" in msg

# --- Security Dependencies ---

async def verify_auth(x_proxy_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Verifies valid authentication token."""
    token = x_proxy_key
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
        else:
            token = authorization
            
    if token != PROXY_AUTH_TOKEN:
        # Allow internal dev mode if needed, but for now strictly enforce if token is set
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")

# --- Routes ---

@app.post("/admin/config")
async def update_config(request: Request, authorized: bool = Depends(verify_auth)):
    """Updates shock configuration dynamically."""
    body = await request.json()
    config_store.update(step=body.get("step"), message=body.get("message"))
    return {"status": "updated", "config": config_store.to_dict()}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorized: bool = Depends(verify_auth)):
    # Rate Limiting
    if not rate_limiter.check_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        req_body = await request.json()
        
        # Extract relevant fields
        model = req_body.get("model", REAL_VLLM_MODEL_NAME)
        messages = req_body.get("messages", [])

        scenario_id = request.headers.get("x-task-id") or os.getenv("TB_TASK_ID")
        metadata = req_body.get("metadata")
        if not scenario_id and isinstance(metadata, dict):
            scenario_id = metadata.get("task_id")
        scenario_id = str(scenario_id or "unknown")

        run_id = request.headers.get("x-run-id") or req_body.get("user") or PROXY_RUN_ID
        run_id = str(run_id)
        
        # --- SHOCK INJECTION LOGIC ---
        step_index = _next_step_index(run_id)
        print(f"Proxy: Processing Step {step_index} (run_id={run_id})...")

        shock_injected = False
        
        if config_store.shock_step > 0 and step_index == config_store.shock_step:
            if config_store.shock_message:
                print(f"!!! PROXY INJECTING SHOCK at Step {step_index} !!!")
                print(f"!!! Message: {config_store.shock_message}")
                
                # Inject the shock
                messages.append({
                    "role": "user", 
                    "content": f"[SYSTEM INTERRUPT] {config_store.shock_message}"
                })
                shock_injected = True
        # -----------------------------

        request_logprobs = _should_request_logprobs()
        if request_logprobs:
            req_body["logprobs"] = True
            req_body["top_logprobs"] = 5
        else:
            req_body.pop("logprobs", None)
            req_body.pop("top_logprobs", None)

        if openai_client is None:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

        # Filter parameters
        valid_params = {
            "messages", "model", "temperature", "top_p", "n", "stream", "stop", "max_tokens",
            "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format",
            "seed", "tools", "tool_choice", "logprobs", "top_logprobs"
        }
        
        client_kwargs = {k: v for k, v in req_body.items() if k in valid_params}
        client_kwargs["model"] = REAL_VLLM_MODEL_NAME

        # Call Real LLM
        response_obj = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_obj = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    **client_kwargs,
                )
                break
            except Exception as e:
                if request_logprobs and (PROXY_REQUEST_LOGPROBS or "auto").strip().lower() == "auto" and _looks_like_logprobs_unsupported(e):
                    with _LOGPROBS_LOCK:
                        global _LOGPROBS_SUPPORTED
                        _LOGPROBS_SUPPORTED = False
                    request_logprobs = False
                    client_kwargs.pop("logprobs", None)
                    client_kwargs.pop("top_logprobs", None)
                    continue

                if attempt == max_retries - 1:
                    print(f"Proxy Error: LLM Call failed: {e}")
                    raise e
                await asyncio.sleep(2**attempt)

        # Logging
        prompt_for_monitor = messages[-1].get("content", "") if messages else ""
        
        def proxy_branching_probe_func() -> List[str]:
            probe_branches = []
            try:
                for _ in range(5):
                    r = openai_client.chat.completions.create(
                        model=REAL_VLLM_MODEL_NAME,
                        messages=messages,
                        temperature=0.9,
                        n=1
                    )
                    probe_branches.append(r.choices[0].message.content)
            except Exception:
                pass
            return probe_branches

        def extract_chosen_logprobs(obj) -> List[float]:
            try:
                choice = obj.choices[0]
                if not choice.logprobs or not getattr(choice.logprobs, "content", None):
                    return []
                logprobs = []
                for token in choice.logprobs.content:
                    lp = getattr(token, "logprob", None)
                    if isinstance(lp, (int, float)):
                        logprobs.append(float(lp))
                return logprobs
            except Exception:
                return []

        def calculate_chosen_surprisal(token_logprobs: List[float]) -> Optional[float]:
            if not token_logprobs:
                return None

            clean: List[float] = []
            for lp in token_logprobs:
                if isinstance(lp, (int, float)) and math.isfinite(lp):
                    clean.append(float(lp))

            if not clean:
                return None

            # log(p) must be <= 0. Some providers return 0 placeholders when unsupported.
            if any(lp > 0.0 for lp in clean):
                return None
            if min(clean) > -1e-3:
                return None

            return sum(-lp for lp in clean) / len(clean)

        token_logprobs = extract_chosen_logprobs(response_obj)
        current_entropy = calculate_chosen_surprisal(token_logprobs)

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            usage = getattr(response_obj, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
        except Exception:
            pass

        # Decide whether to compute SCR (expensive: 5 extra LLM calls).
        probe_header = (request.headers.get("x-probe-scr") or "").strip()
        should_probe_scr = False
        if PROXY_SCR_MODE == "always":
            should_probe_scr = True
        elif PROXY_SCR_MODE == "shock" and shock_injected:
            should_probe_scr = True
        elif probe_header in ("1", "true", "yes", "on"):
            should_probe_scr = True

        event_type = "proxy_shock_injected" if shock_injected else "llm_call"
        if should_probe_scr and not shock_injected:
            event_type = "proxy_probe"

        branches: List[str] = []
        scr_value: Optional[float] = None
        if should_probe_scr:
            branches = await asyncio.to_thread(proxy_branching_probe_func)
            if monitor.metric_service:
                scr_value = monitor.metric_service.calculate_scr(branches)

        monitor.log_step(
            run_id=run_id,
            scenario_id=scenario_id,
            model_name=str(model),
            step_index=step_index,
            event_type=event_type,
            prompt=prompt_for_monitor,
            messages=messages,
            response_obj=response_obj.dict(),
            current_entropy=current_entropy,
            scr=scr_value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            branches_count=len(branches) if branches else None,
            branching_func=None,
        )
        
        return response_obj 

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Proxy: Server started and ready.")

if __name__ == "__main__":
    print("Initializing LLM Proxy...")
    get_monitor()
    uvicorn.run(app, host="0.0.0.0", port=8000)
