from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return int(e.returncode), (e.output or "").strip()
    except FileNotFoundError:
        return 127, ""


def _http_json(url: str, headers: dict[str, str] | None = None, timeout_s: int = 2) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            json.loads(data)
            return True, data
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        return False, str(e)


def main() -> int:
    print("== Setup Check ==")
    print(f"python: {sys.executable}")
    print(f"cwd: {os.getcwd()}")
    print("")

    print("## Python deps")
    required = ["openai", "numpy", "pandas", "scipy", "sentence_transformers"]
    for mod in required:
        try:
            __import__(mod)
            print(f"OK   import {mod}")
        except Exception as e:
            print(f"MISS import {mod}: {e}")
    print("")

    print("## Sandbox backend")
    sandbox_backend = (os.getenv("SANDBOX_BACKEND") or "auto").strip().lower()
    print(f"SANDBOX_BACKEND={sandbox_backend}")
    sandbox_per_run = (os.getenv("SANDBOX_PER_RUN") or "").strip()
    if sandbox_per_run:
        print(f"SANDBOX_PER_RUN={sandbox_per_run}")
    scenario_seed = (os.getenv("SCENARIO_SEED") or "").strip()
    if scenario_seed:
        print(f"SCENARIO_SEED={scenario_seed}")
    print("")

    print("## Docker (only needed for TerminalBench runs)")
    if shutil.which("docker") is None:
        print("docker: not found")
    else:
        code, out = _run(["docker", "ps"])
        if code == 0:
            print("docker: OK (daemon accessible)")
        else:
            print(f"docker: NOT usable here (exit={code})")
            if out:
                print(out.splitlines()[-1])
            print("Tip: use SANDBOX_BACKEND=local for local runs.")
    print("")

    print("## LLM endpoint")
    base_url = (os.getenv("VLLM_BASE_URL") or "").strip()
    model = (os.getenv("VLLM_MODEL_NAME") or "").strip()
    if not base_url:
        print("VLLM_BASE_URL: not set")
    else:
        print(f"VLLM_BASE_URL={base_url}")
    if not model:
        print("VLLM_MODEL_NAME: not set")
    else:
        print(f"VLLM_MODEL_NAME={model}")

    # Best-effort check for OpenAI-compatible /models endpoint.
    if base_url.startswith("http://") or base_url.startswith("https://"):
        url = base_url.rstrip("/") + "/models"
        ok, detail = _http_json(url, headers={"Authorization": "Bearer dummy"})
        if ok:
            print("models endpoint: OK")
        else:
            print(f"models endpoint: NOT reachable/JSON ({detail})")
    print("")

    print("## SCR embeddings")
    from src.services.metrics import EmbeddingMetricService  # local import for fast failure

    svc = EmbeddingMetricService()
    print(f"backend: {getattr(svc, 'embedding_backend', None)}")
    print(f"model:   {getattr(svc, 'model_name', None)}")
    print(f"device:  {getattr(svc, 'device', None)}")
    print(f"local_files_only: {getattr(svc, 'local_files_only', None)}")
    print(f"hash_dim:         {getattr(svc, 'hash_dim', None)}")
    print("")
    print("If backend is 'hash', SCR/RDI are lexical proxies (offline-safe).")
    print("To enable semantic embeddings, ensure the SentenceTransformer model is cached locally,")
    print("or set SCR_LOCAL_FILES_ONLY=0 and run once with network access to download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
