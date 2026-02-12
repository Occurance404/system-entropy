from __future__ import annotations

from typing import Any, Dict, List

KNOWN_TOOLS = {"read_file", "write_file", "execute_python", "run_shell", "search_web"}

TOOL_NAME_ALIASES = {
    "shell": "run_shell",
    "run_command": "run_shell",
    "execute_command": "run_shell",
    "exec_shell": "run_shell",
    "bash": "run_shell",
    "cat_file": "read_file",
    "open_file": "read_file",
    "read": "read_file",
    "save_file": "write_file",
    "write": "write_file",
    "edit_file": "write_file",
    "python": "execute_python",
    "run_python": "execute_python",
    "exec_python": "execute_python",
    "web_search": "search_web",
    "search": "search_web",
}


def build_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are an autonomous AI agent designed to execute tasks by using available tools and responding "
            "directly with actions or answers. Your goal is to complete the given task efficiently and accurately. "
            "Do not engage in conversational chitchat or ask for clarification unless absolutely necessary for "
            "task execution. If you need to perform an action, use the tool functions provided. "
            "If the task is complete, provide a final summary."
        ),
    }


def build_text_tools_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are an autonomous AI agent in an environment where tools exist, but native function calling may be unavailable.\n"
            "\n"
            "When you want to use a tool, respond with ONLY a single JSON object in one of these two forms:\n"
            '1) Tool call:\n{"type":"tool_use","tool":"<tool_name>","content":{...tool_args...}}\n'
            '2) Final answer:\n{"type":"llm_reply","content":"..."}\n'
            "\n"
            "Rules:\n"
            "- Output ONLY JSON (no markdown, no backticks, no extra keys).\n"
            "- Use exactly one tool call at a time.\n"
            "\n"
            "Available tools:\n"
            "- read_file: {path (string), mode (auto|full|outline, optional), start_line (int, optional), end_line (int, optional), with_line_numbers (bool, optional)}\n"
            "- write_file: {path (string), content (string)}\n"
            "- execute_python: {script_path (string)}\n"
            "- run_shell: {command (string)}\n"
            "- search_web: {query (string)}\n"
        ),
    }


def build_probe_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are running an internal probe to assess plan stability.\n"
            "Do NOT call tools. Output a short, high-level next-step plan (1-3 sentences)."
        ),
    }


def build_completion_verifier_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are an independent completion verifier.\n"
            "Decide whether the task is complete from provided evidence only.\n"
            "Return ONLY JSON with schema:\n"
            '{"verdict":"done|not_done|uncertain","confidence":0.0-1.0,"reason":"..."}\n'
            "Rules:\n"
            "- done: strong evidence that requirements are met.\n"
            "- not_done: clear missing work or failing evidence.\n"
            "- uncertain: insufficient evidence.\n"
            "- Never invent test results or files.\n"
        ),
    }


def build_tools_schema() -> List[Dict[str, Any]]:
    return [
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
                            "enum": ["auto", "full", "outline"],
                        },
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                        "with_line_numbers": {"type": "boolean"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
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
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Executes a python script.",
                "parameters": {
                    "type": "object",
                    "properties": {"script_path": {"type": "string"}},
                    "required": ["script_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Searches the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Executes a shell command in the sandbox. Use this for navigating directories, running tests, or managing files.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]
