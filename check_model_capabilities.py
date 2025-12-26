import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_env(key: str, dotenv: Dict[str, str]) -> Optional[str]:
    return os.getenv(key) or dotenv.get(key)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str


def _parse_model_spec(raw: Dict[str, Any], dotenv: Dict[str, str]) -> ModelSpec:
    name = str(raw["name"])
    model = str(raw["model"])
    base_url = str(raw.get("base_url") or _resolve_env("VLLM_BASE_URL", dotenv) or "")
    api_key_env = str(raw.get("api_key_env") or "VLLM_API_KEY")
    if not base_url:
        raise ValueError(f"ModelSpec '{name}' missing base_url (or VLLM_BASE_URL).")
    return ModelSpec(name=name, model=model, base_url=base_url, api_key_env=api_key_env)


def _supports_logprobs(client: OpenAI, model: str) -> Tuple[bool, str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ping"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=1,
        )
        choice = resp.choices[0]
        logprobs_obj = getattr(choice, "logprobs", None)
        content = getattr(logprobs_obj, "content", None) if logprobs_obj is not None else None
        if not content:
            return False, "No logprobs.content returned"
        token = content[0]
        lp = getattr(token, "logprob", None)
        if isinstance(lp, (int, float)):
            return True, ""
        return False, "logprob not numeric"
    except Exception as e:
        return False, str(e)


def _supports_tools(client: OpenAI, model: str) -> Tuple[bool, bool, str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Health-check tool. Returns pong.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call the ping tool now."}],
            max_tokens=10,
            temperature=0,
            tools=tools,
            tool_choice="auto",
        )
        choice = resp.choices[0]
        message = getattr(choice, "message", None)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        made_call = bool(tool_calls)
        return True, made_call, ""
    except Exception as e:
        return False, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe provider capabilities per model (cheap).")
    parser.add_argument("--models", default="benchmarks/models.example.json", help="Models JSON (same schema as run_benchmark).")
    parser.add_argument("--out", default=None, help="Output CSV (default: data/results/model_capabilities_<ts>.csv)")
    args = parser.parse_args()

    dotenv = dotenv_values(".env")
    raw_models = _load_json(args.models)
    model_specs = [_parse_model_spec(m, dotenv) for m in raw_models]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("data", "results", f"model_capabilities_{ts}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for spec in model_specs:
        api_key = _resolve_env(spec.api_key_env, dotenv)
        if not api_key:
            rows.append(
                {
                    "model_name": spec.name,
                    "model": spec.model,
                    "base_url": spec.base_url,
                    "api_key_env": spec.api_key_env,
                    "error": f"Missing API key env var: {spec.api_key_env}",
                }
            )
            continue

        client = OpenAI(api_key=api_key, base_url=spec.base_url)

        logprobs_ok, logprobs_note = _supports_logprobs(client, spec.model)
        tools_ok, tool_called, tools_note = _supports_tools(client, spec.model)

        rows.append(
            {
                "model_name": spec.name,
                "model": spec.model,
                "base_url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "supports_logprobs": logprobs_ok,
                "supports_tools_param": tools_ok,
                "made_tool_call": tool_called,
                "logprobs_note": logprobs_note,
                "tools_note": tools_note,
            }
        )

        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

