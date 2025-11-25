import os
import uvicorn
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
# from litellm import completion, CustomStreamWrapper, Message # Removed litellm from proxy logic
# from litellm.utils import token_counter # Removed
from openai import OpenAI # Use direct OpenAI client
from dotenv import dotenv_values

# Import our monitor
from src.monitor.terminal_bench_monitor import get_monitor

# --- FastAPI App Setup ---
app = FastAPI(title="LLM Proxy with Metric Injection")
monitor = get_monitor()

# Load real API config from .env
config = dotenv_values(".env")
REAL_VLLM_API_KEY = config.get("VLLM_API_KEY")
REAL_VLLM_BASE_URL = config.get("VLLM_BASE_URL")
REAL_VLLM_MODEL_NAME = config.get("VLLM_MODEL_NAME", "deepseek-chat")

if not REAL_VLLM_API_KEY or not REAL_VLLM_BASE_URL:
    print("WARNING: Real LLM API credentials not fully set in .env. Proxy might fail.")

# Initialize OpenAI client for the proxy to use directly
try:
    openai_client = OpenAI(
        api_key=REAL_VLLM_API_KEY,
        base_url=REAL_VLLM_BASE_URL,
    )
except Exception as e:
    print(f"ERROR: Could not initialize OpenAI client in proxy: {e}")
    openai_client = None

# --- Shock Injection Configuration ---
SHOCK_TRIGGER_STEP = int(os.environ.get("SHOCK_TRIGGER_STEP", -1)) # -1 means disabled
SHOCK_MESSAGE = os.environ.get("SHOCK_MESSAGE", "")
CURRENT_STEP = 0

print(f"Proxy configured for real LLM: {REAL_VLLM_MODEL_NAME} at {REAL_VLLM_BASE_URL}")
print(f"Shock Injection Config: Step={SHOCK_TRIGGER_STEP}, Message='{SHOCK_MESSAGE}'")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global CURRENT_STEP
    try:
        req_body = await request.json()
        
        # Extract relevant fields for OpenAI client completion
        model = req_body.get("model", REAL_VLLM_MODEL_NAME)
        messages = req_body.get("messages", [])
        temperature = req_body.get("temperature", 0.7)
        top_p = req_body.get("top_p", 1.0)
        n = req_body.get("n", 1) # Support n for branching if underlying API allows
        stream = req_body.get("stream", False)
        tools = req_body.get("tools")
        tool_choice = req_body.get("tool_choice")

        # --- SHOCK INJECTION LOGIC ---
        CURRENT_STEP += 1
        print(f"Proxy: Processing Step {CURRENT_STEP}...")
        
        if SHOCK_TRIGGER_STEP > 0 and CURRENT_STEP == SHOCK_TRIGGER_STEP:
            if SHOCK_MESSAGE:
                print(f"!!! PROXY INJECTING SHOCK at Step {CURRENT_STEP} !!!")
                print(f"!!! Message: {SHOCK_MESSAGE}")
                
                # Inject the shock as a user message at the end of the history
                # This simulates the user interrupting the agent
                messages.append({
                    "role": "user", 
                    "content": f"[SYSTEM INTERRUPT] {SHOCK_MESSAGE}"
                })
        # -----------------------------

        # Crucially, ensure logprobs are requested for our metrics
        # OpenAI client needs logprobs=True AND top_logprobs
        req_body["logprobs"] = True
        req_body["top_logprobs"] = 5 # Request Top-5 for Distribution Entropy (Fixing Scientific Validity)

        if openai_client is None:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized.")

        # Filter req_body to only include valid parameters for the OpenAI Python Client
        # Generic agents might send extra fields (like 'user', 'custom_id') that the client might reject
        # or we just want to be safe.
        valid_params = {
            "messages", "model", "temperature", "top_p", "n", "stream", "stop", "max_tokens",
            "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format",
            "seed", "tools", "tool_choice", "logprobs", "top_logprobs"
        }
        
        client_kwargs = {k: v for k, v in req_body.items() if k in valid_params}
        
        # Force our model name
        client_kwargs["model"] = REAL_VLLM_MODEL_NAME

        # Call the real LLM API directly with Retry Logic (Fixing Fragility)
        response_obj = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use asyncio.to_thread for blocking OpenAI client call
                response_obj = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    **client_kwargs # Unpack filtered arguments
                )
                break # Success
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Proxy Error: LLM Call failed after {max_retries} attempts: {e}")
                    raise e
                print(f"Proxy Warning: LLM Call failed (Attempt {attempt+1}/{max_retries}). Retrying...")
                await asyncio.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s

        # Log metrics (Branching Probe will be handled by the monitor if triggered)
        # The log_step needs the full litellm-like response dict for consistency,
        # so we convert the openai response to a dict similar to what litellm provides.
        
        # For monitor logging, prompt should be the last user message
        prompt_for_monitor = messages[-1].get("content", "") if messages else ""
        
        # Helper for branching probe (if SCR is needed by monitor)
        def proxy_branching_probe_func():
            # Temporarily disabled due to credit limits for branching probe
            return []
            # probe_branches = []
            # for _ in range(5): # Generate 5 branches for SCR
            #     r = openai_client.chat.completions.create(
            #         model=REAL_VLLM_MODEL_NAME, # FORCE the real model name from .env
            #         messages=messages,
            #         temperature=0.9, # High temp for divergence
            #         n=1,
            #         logprobs=True,
            #         top_logprobs=1
            #     )
            #     probe_branches.append(r.choices[0].message.content)
            # return probe_branches

        monitor.log_step(
            model_name=model,
            prompt=prompt_for_monitor,
            messages=messages, # Pass messages for context
            response_obj=response_obj.dict(), # Convert Pydantic model to dict
            branching_func=proxy_branching_probe_func # Pass func for SCR
        )
        
        # Return the response to TerminalBench
        # OpenAI client returns a Pydantic model, FastAPI converts it to JSON automatically
        return response_obj 

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    print("Proxy: Server started and ready to accept connections.")

if __name__ == "__main__":
    # Ensure monitor is initialized and prints confirmation
    print("Initializing LLM Proxy...")
    get_monitor()
    uvicorn.run(app, host="0.0.0.0", port=8000)
