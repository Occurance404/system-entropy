import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agent.real_agent import OpenAICompatibleAgent


@dataclass(frozen=True)
class ToolScenario:
    canonical_tool: str
    aliases: List[str]
    canonical_args: Dict[str, Any]
    shorthand_arg: str
    required_key: str


def _make_agent() -> OpenAICompatibleAgent:
    # No network calls happen on construction.
    return OpenAICompatibleAgent(
        model_name="tool-stress-dummy",
        base_url="http://example.com/v1",
        api_key="x",
        temperature=0.0,
    )


def _json_wrappers(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = json.dumps(payload, ensure_ascii=False)
    return [
        ("raw_json", raw),
        ("fenced_json", f"```json\n{raw}\n```"),
        ("prefixed_text", f"Action selected:\n{raw}\nProceed."),
        ("json_list_first", json.dumps([payload, {"type": "llm_reply", "content": "done"}], ensure_ascii=False)),
    ]


def _build_text_cases() -> List[Dict[str, Any]]:
    tools = [
        ToolScenario(
            canonical_tool="run_shell",
            aliases=["run_shell", "shell", "execute_command", "run-command"],
            canonical_args={"command": "ls -la"},
            shorthand_arg="ls -la",
            required_key="command",
        ),
        ToolScenario(
            canonical_tool="read_file",
            aliases=["read_file", "read", "open_file", "cat_file"],
            canonical_args={"path": "README.md"},
            shorthand_arg="README.md",
            required_key="path",
        ),
        ToolScenario(
            canonical_tool="write_file",
            aliases=["write_file", "write", "save_file", "edit_file"],
            canonical_args={"path": "out.txt", "content": "hello"},
            shorthand_arg="hello world",
            required_key="content",
        ),
        ToolScenario(
            canonical_tool="execute_python",
            aliases=["execute_python", "python", "run_python", "exec_python"],
            canonical_args={"script_path": "script.py"},
            shorthand_arg="script.py",
            required_key="script_path",
        ),
        ToolScenario(
            canonical_tool="search_web",
            aliases=["search_web", "search", "web_search"],
            canonical_args={"query": "best parser design"},
            shorthand_arg="best parser design",
            required_key="query",
        ),
    ]

    forms = [
        ("content_dict", lambda alias, s: {"type": "tool_use", "tool": alias, "content": dict(s.canonical_args)}),
        ("args_dict", lambda alias, s: {"type": "tool_use", "tool": alias, "args": dict(s.canonical_args)}),
        (
            "arguments_json_str",
            lambda alias, s: {
                "type": "tool_use",
                "tool": alias,
                "arguments": json.dumps(s.canonical_args, ensure_ascii=False),
            },
        ),
        (
            "arguments_py_dict_str",
            lambda alias, s: {
                "type": "tool_use",
                "tool": alias,
                "arguments": str(s.canonical_args),
            },
        ),
        ("content_shorthand", lambda alias, s: {"type": "tool_use", "tool": alias, "content": s.shorthand_arg}),
        (
            "function_obj",
            lambda alias, s: {
                "type": "function_call",
                "function": {"name": alias, "arguments": json.dumps(s.canonical_args, ensure_ascii=False)},
            },
        ),
    ]

    cases: List[Dict[str, Any]] = []
    case_id = 1
    for spec in tools:
        for alias in spec.aliases:
            for form_name, builder in forms:
                payload = builder(alias, spec)
                for wrap_name, text in _json_wrappers(payload):
                    cases.append(
                        {
                            "case_id": f"text_{case_id:04d}",
                            "category": "text_tool_call",
                            "tool_expected": spec.canonical_tool,
                            "required_key": spec.required_key,
                            "expected_type": "tool_use",
                            "input_text": text,
                            "form": form_name,
                            "wrapper": wrap_name,
                        }
                    )
                    case_id += 1

    # Additional negative/edge cases (expected to parse as llm_reply or None).
    edge_payloads = [
        {"type": "llm_reply", "content": "Task done."},
        {"type": "answer", "answer": "Complete."},
        {"foo": "bar"},
        {"type": "tool_use", "tool": "", "content": {}},
    ]
    for payload in edge_payloads:
        for wrap_name, text in _json_wrappers(payload):
            cases.append(
                {
                    "case_id": f"text_{case_id:04d}",
                    "category": "text_edge",
                    "tool_expected": None,
                    "required_key": None,
                    "expected_type": "llm_or_none",
                    "input_text": text,
                    "form": "edge",
                    "wrapper": wrap_name,
                }
            )
            case_id += 1
    return cases


def _build_argument_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case_id = 1
    arg_payloads = [
        ("run_shell", '{"cmd":"pytest -q"}', "command"),
        ("run_shell", "{'shell_command': 'pwd'}", "command"),
        ("read_file", '{"file":"README.md"}', "path"),
        ("write_file", '{"filename":"a.py","text":"print(1)"}', "content"),
        ("execute_python", '{"path":"script.py"}', "script_path"),
        ("search_web", '{"q":"entropy metric"}', "query"),
        ("run_shell", "ls -la", "command"),
        ("read_file", "README.md", "path"),
        ("execute_python", "script.py", "script_path"),
        ("search_web", "great schedulers", "query"),
    ]
    for _ in range(12):
        for tool_name, raw_arg, required_key in arg_payloads:
            rows.append(
                {
                    "case_id": f"arg_{case_id:04d}",
                    "category": "native_tool_args",
                    "tool_expected": tool_name,
                    "required_key": required_key,
                    "raw_args": raw_arg,
                }
            )
            case_id += 1
    return rows


def _build_content_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case_id = 1
    payloads: List[Any] = [
        "plain string",
        [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
        [{"type": "text", "text": "line1"}, {"type": "not_text", "value": "x"}],
        [{"foo": "bar"}],
        None,
    ]
    for _ in range(8):
        for payload in payloads:
            rows.append(
                {
                    "case_id": f"content_{case_id:04d}",
                    "category": "content_normalization",
                    "content_payload": payload,
                }
            )
            case_id += 1
    return rows


def _evaluate_text_case(agent: OpenAICompatibleAgent, row: Dict[str, Any]) -> Tuple[bool, str]:
    extracted = agent._extract_first_json_object(row["input_text"])
    normalized = agent._normalize_text_tool_call(extracted)

    expected_type = row["expected_type"]
    if expected_type == "tool_use":
        if not isinstance(normalized, dict):
            return False, "normalized=None"
        if normalized.get("type") != "tool_use":
            return False, f"type={normalized.get('type')}"
        if normalized.get("tool") != row["tool_expected"]:
            return False, f"tool={normalized.get('tool')}"
        content = normalized.get("content")
        if not isinstance(content, dict):
            return False, "content_not_dict"
        req = row["required_key"]
        if req not in content:
            return False, f"missing_key={req}"
        if content.get(req) in (None, ""):
            return False, f"empty_key={req}"
        return True, ""

    # llm_or_none edge class
    if normalized is None:
        return True, ""
    if normalized.get("type") == "llm_reply":
        return True, ""
    return False, f"unexpected={normalized}"


def _evaluate_argument_case(agent: OpenAICompatibleAgent, row: Dict[str, Any]) -> Tuple[bool, str]:
    parsed = agent._parse_tool_arguments(row["raw_args"], row["tool_expected"])
    if not isinstance(parsed, dict):
        return False, "parsed_not_dict"
    req = row["required_key"]
    if req not in parsed:
        return False, f"missing_key={req}"
    if parsed.get(req) in (None, ""):
        return False, f"empty_key={req}"
    return True, ""


def _evaluate_content_case(agent: OpenAICompatibleAgent, row: Dict[str, Any]) -> Tuple[bool, str]:
    text = agent._content_to_text(row["content_payload"])
    if not isinstance(text, str):
        return False, "output_not_str"
    if row["content_payload"] is None and text != "":
        return False, "none_not_empty"
    return True, ""


def run_suite() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    agent = _make_agent()
    results: List[Dict[str, Any]] = []

    for row in _build_text_cases():
        ok, reason = _evaluate_text_case(agent, row)
        results.append({**row, "passed": ok, "reason": reason})

    for row in _build_argument_cases():
        ok, reason = _evaluate_argument_case(agent, row)
        results.append({**row, "passed": ok, "reason": reason})

    for row in _build_content_cases():
        ok, reason = _evaluate_content_case(agent, row)
        results.append({**row, "passed": ok, "reason": reason})

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    by_category: Dict[str, Dict[str, Any]] = {}
    for row in results:
        cat = row["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if row["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    for cat, bucket in by_category.items():
        bucket["pass_rate"] = round((bucket["passed"] / bucket["total"]) * 100.0, 2) if bucket["total"] else 0.0

    fail_reasons: Dict[str, int] = {}
    for row in results:
        if row["passed"]:
            continue
        fail_reasons[row["reason"]] = fail_reasons.get(row["reason"], 0) + 1

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        "by_category": by_category,
        "top_fail_reasons": sorted(fail_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }
    return results, summary


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test tool-call parsing and normalization.")
    parser.add_argument("--out-prefix", default=None, help="Output prefix (without extension).")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or os.path.join("data", "results", f"tool_stress_{ts}")

    results, summary = run_suite()
    csv_path = f"{prefix}.csv"
    json_path = f"{prefix}.summary.json"
    _write_csv(csv_path, results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved case table: {csv_path}")
    print(f"Saved summary:    {json_path}")
    print("")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
