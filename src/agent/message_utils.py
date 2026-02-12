from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Set


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return json.dumps(content, ensure_ascii=False)
    try:
        return str(content)
    except Exception:
        return ""


def extract_first_json_object(text: str) -> Optional[Any]:
    if not isinstance(text, str) or not text.strip():
        return None

    stripped = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            stripped = candidate

    try:
        return json.loads(stripped)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[i:])
            return obj
        except Exception:
            continue
    return None


def normalize_tool_name(tool_name: Any, *, known_tools: Set[str], aliases: Dict[str, str]) -> Optional[str]:
    if tool_name is None:
        return None
    normalized = str(tool_name).strip()
    if not normalized:
        return None
    lowered = normalized.lower().replace("-", "_")
    lowered = aliases.get(lowered, lowered)
    if lowered in known_tools:
        return lowered
    return normalized


def coerce_tool_args(tool_name: str, args: Any) -> Optional[Dict[str, Any]]:
    if isinstance(args, dict):
        coerced = dict(args)
        if tool_name == "run_shell" and "command" not in coerced:
            coerced["command"] = coerced.get("cmd") or coerced.get("shell_command")
        elif tool_name == "read_file" and "path" not in coerced:
            coerced["path"] = coerced.get("file") or coerced.get("filename")
        elif tool_name == "write_file":
            if "path" not in coerced:
                coerced["path"] = coerced.get("file") or coerced.get("filename")
            if "content" not in coerced:
                coerced["content"] = coerced.get("text") or coerced.get("body") or coerced.get("code")
        elif tool_name == "execute_python" and "script_path" not in coerced:
            coerced["script_path"] = coerced.get("path") or coerced.get("script")
        elif tool_name == "search_web" and "query" not in coerced:
            coerced["query"] = coerced.get("q") or coerced.get("search")
        return coerced

    if isinstance(args, str):
        raw = args.strip()
        if not raw:
            return {}
        parsed = extract_first_json_object(raw)
        if isinstance(parsed, dict):
            return coerce_tool_args(tool_name, parsed)

        if tool_name == "run_shell":
            return {"command": raw}
        if tool_name == "read_file":
            return {"path": raw}
        if tool_name == "execute_python":
            return {"script_path": raw}
        if tool_name == "search_web":
            return {"query": raw}
        if tool_name == "write_file":
            return {"path": "output.txt", "content": raw}
        return None

    return None


def parse_tool_arguments(raw_arguments: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    if isinstance(raw_arguments, dict):
        return coerce_tool_args(tool_name, raw_arguments)

    if not isinstance(raw_arguments, str):
        return coerce_tool_args(tool_name, raw_arguments)

    raw = raw_arguments.strip()
    if not raw:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            coerced = coerce_tool_args(tool_name, parsed)
            if coerced is not None:
                return coerced
        except Exception:
            continue

    extracted = extract_first_json_object(raw)
    if isinstance(extracted, dict):
        coerced = coerce_tool_args(tool_name, extracted)
        if coerced is not None:
            return coerced

    return coerce_tool_args(tool_name, raw)


def normalize_text_tool_call(
    obj: Any,
    *,
    known_tools: Set[str],
    aliases: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if isinstance(obj, list):
        obj = next((x for x in obj if isinstance(x, dict)), None)
    if not isinstance(obj, dict):
        return None

    raw_type = obj.get("type") or obj.get("action") or obj.get("kind")
    if isinstance(raw_type, str):
        raw_type = raw_type.strip().lower()

    tool_name = obj.get("tool") or obj.get("name") or obj.get("function")
    args = obj.get("content")
    if args is None:
        args = obj.get("args") or obj.get("arguments") or obj.get("parameters")

    if isinstance(tool_name, dict):
        args = tool_name.get("arguments", args)
        tool_name = tool_name.get("name")

    normalized_tool = normalize_tool_name(tool_name, known_tools=known_tools, aliases=aliases)
    if raw_type in ("tool_use", "tool", "function_call", "call_tool") or (
        normalized_tool and raw_type not in ("llm_reply", "final", "answer")
    ):
        if not normalized_tool:
            return None
        if args is None:
            args = {}
        coerced_args = coerce_tool_args(normalized_tool, args)
        if coerced_args is None:
            return None
        return {"type": "tool_use", "tool": normalized_tool, "content": coerced_args}

    if raw_type in ("llm_reply", "final", "answer"):
        content = obj.get("content") or obj.get("answer") or obj.get("final")
        if content is None:
            content = ""
        return {"type": "llm_reply", "content": str(content)}

    return None


def build_dynamic_execution_guidance(history: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not history:
        return None

    recent = history[-20:]

    tool_call_signatures: List[str] = []
    for msg in recent:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("Used tool: "):
            tool_call_signatures.append(content.strip())

    repeated_last_tool = False
    if len(tool_call_signatures) >= 2 and tool_call_signatures[-1] == tool_call_signatures[-2]:
        repeated_last_tool = True

    trailing_non_action_assistant = 0
    for msg in reversed(recent):
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool_output":
            break
        if role == "assistant":
            if isinstance(content, str) and content.startswith("Used tool: "):
                break
            trailing_non_action_assistant += 1

    hints: List[str] = [
        "Prioritize concrete progress over narration.",
        "Use exactly one best-next tool call when action is needed.",
    ]
    if trailing_non_action_assistant >= 2:
        hints.append("Do not continue with prose-only replies; take a concrete tool action now unless task is fully complete.")
    if repeated_last_tool:
        hints.append("Avoid repeating the exact same tool call/arguments; try a different diagnostic or fix strategy.")

    hints.append("After edits, verify quickly (e.g., read target files, run a focused command/test) before claiming completion.")

    return {
        "role": "system",
        "content": "Execution Guidance:\n- " + "\n- ".join(hints),
    }
