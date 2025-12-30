import hashlib
import json
from typing import Any, Dict, Optional

from src.security.secrets import SecretScanner


def format_tool_args_for_history(tool_name: str, tool_args: Any) -> str:
    """
    Prevents context poisoning + secret leakage by never embedding large blobs
    (e.g., write_file content) directly into the chat history.
    """
    if not isinstance(tool_args, dict):
        return SecretScanner.redact(str(tool_args))

    safe: Dict[str, Any] = {}
    for key, value in tool_args.items():
        if isinstance(value, str) and len(value) > 200:
            digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
            safe[key] = f"<{len(value)} chars sha256={digest}>"
        else:
            safe[key] = value

    # Extra hardening for write_file
    if tool_name == "write_file" and isinstance(tool_args.get("content"), str):
        content = tool_args["content"]
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        safe["content"] = f"<{len(content)} chars sha256={digest}>"

    return SecretScanner.redact(json.dumps(safe, ensure_ascii=False, sort_keys=True))


def action_signature(action_intent: Dict[str, Any]) -> Optional[str]:
    """
    Produces a stable signature for loop detection without storing large blobs.
    """
    try:
        intent_type = action_intent.get("type")
        if intent_type == "tool_use":
            tool_name = str(action_intent.get("tool") or "").strip()
            args_repr = format_tool_args_for_history(tool_name, action_intent.get("content"))
            return f"tool:{tool_name}:{args_repr}"
        if intent_type == "llm_reply":
            content = action_intent.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            normalized = " ".join(content.split())
            if len(normalized) > 600:
                normalized = normalized[:600]
            return f"reply:{normalized}"
    except Exception:
        return None
    return None

