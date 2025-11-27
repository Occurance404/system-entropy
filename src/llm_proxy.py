import os
import uvicorn
import json
import asyncio
import time
from datetime import datetime
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

CURRENT_STEP = 0

print(f"Proxy configured for real LLM: {REAL_VLLM_MODEL_NAME} at {REAL_VLLM_BASE_URL}")
print(f"Initial Shock Config: {config_store.to_dict()}")

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
    global CURRENT_STEP
    
    # Rate Limiting
    if not rate_limiter.check_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        req_body = await request.json()
        
        # Extract relevant fields
        model = req_body.get("model", REAL_VLLM_MODEL_NAME)
        messages = req_body.get("messages", [])
        
        # --- SHOCK INJECTION LOGIC ---
        CURRENT_STEP += 1
        print(f"Proxy: Processing Step {CURRENT_STEP}...")
        
        if config_store.shock_step > 0 and CURRENT_STEP == config_store.shock_step:
            if config_store.shock_message:
                print(f"!!! PROXY INJECTING SHOCK at Step {CURRENT_STEP} !!!")
                print(f"!!! Message: {config_store.shock_message}")
                
                # Inject the shock
                messages.append({
                    "role": "user", 
                    "content": f"[SYSTEM INTERRUPT] {config_store.shock_message}"
                })
        # -----------------------------

        # Ensure logprobs are requested
        req_body["logprobs"] = True
        req_body["top_logprobs"] = 5

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
                    **client_kwargs
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Proxy Error: LLM Call failed: {e}")
                    raise e
                await asyncio.sleep(2 ** attempt)

        # Logging
        prompt_for_monitor = messages[-1].get("content", "") if messages else ""
        
        def proxy_branching_probe_func():
            probe_branches = []
            try:
                for _ in range(5):
                    r = openai_client.chat.completions.create(
                        model=REAL_VLLM_MODEL_NAME,
                        messages=messages,
                        temperature=0.9,
                        n=1,
                        logprobs=True,
                        top_logprobs=1
                    )
                    probe_branches.append(r.choices[0].message.content)
            except Exception:
                pass
            return probe_branches

        monitor.log_step(
            model_name=model,
            prompt=prompt_for_monitor,
            messages=messages,
            response_obj=response_obj.dict(),
            branching_func=proxy_branching_probe_func
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
