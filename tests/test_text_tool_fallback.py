import pytest
from types import SimpleNamespace

from src.agent.real_agent import OpenAICompatibleAgent


@pytest.fixture
def agent():
    # Construction should be side-effect free (no network calls).
    return OpenAICompatibleAgent(model_name="dummy", base_url="http://example.com/v1", api_key="x", temperature=0.0)


def test_extracts_json_from_plain_string(agent):
    text = '{"type":"tool_use","tool":"read_file","content":{"path":"README.md"}}'
    obj = agent._extract_first_json_object(text)
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {"type": "tool_use", "tool": "read_file", "content": {"path": "README.md"}}


def test_extracts_json_from_wrapped_text(agent):
    text = 'Here you go:\n```json\n{"type":"llm_reply","content":"ok"}\n```\n'
    obj = agent._extract_first_json_object(text)
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {"type": "llm_reply", "content": "ok"}


def test_normalizes_shorthand_string_args(agent):
    obj = {"tool": "run_shell", "args": "ls -la"}
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {"type": "tool_use", "tool": "run_shell", "content": {"command": "ls -la"}}


def test_normalizes_tool_alias_and_args(agent):
    obj = {"type": "tool_use", "tool": "execute_command", "arguments": {"cmd": "pwd"}}
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {"type": "tool_use", "tool": "run_shell", "content": {"cmd": "pwd", "command": "pwd"}}


def test_extracts_first_dict_from_json_list(agent):
    text = '[{"type":"tool_use","tool":"read","content":"README.md"},{"type":"llm_reply","content":"done"}]'
    obj = agent._extract_first_json_object(text)
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {"type": "tool_use", "tool": "read_file", "content": {"path": "README.md"}}


def test_parses_native_tool_args_with_single_quotes(agent):
    parsed = agent._parse_tool_arguments("{'cmd': 'pytest -q'}", "run_shell")
    assert parsed == {"cmd": "pytest -q", "command": "pytest -q"}


def test_content_to_text_handles_part_list(agent):
    content = [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
    assert agent._content_to_text(content) == "line1\nline2"


def test_parses_used_tool_signature_with_json_args(agent):
    text = 'Used tool: run_shell with args: {"command": "python process_timeline.py"}'
    normalized = agent._normalize_text_tool_call(text)
    assert normalized == {
        "type": "tool_use",
        "tool": "run_shell",
        "content": {"command": "python process_timeline.py"},
    }


def test_parses_used_tool_signature_with_alias_and_literal_dict(agent):
    text = "Used tool: execute_command with args: {'cmd': 'pwd'}"
    normalized = agent._normalize_text_tool_call(text)
    assert normalized == {
        "type": "tool_use",
        "tool": "run_shell",
        "content": {"cmd": "pwd", "command": "pwd"},
    }


def test_parses_used_tool_signature_when_prefixed_by_prose(agent):
    text = (
        "I will now create the file.\n\n"
        'Used tool: write_file with args: {"path":"timeline.csv","content":"a,b\\n1,2"}'
    )
    normalized = agent._normalize_text_tool_call(text)
    assert normalized == {
        "type": "tool_use",
        "tool": "write_file",
        "content": {"path": "timeline.csv", "content": "a,b\n1,2"},
    }


def test_parses_tool_when_wrapped_as_llm_reply_content(agent):
    obj = {
        "type": "llm_reply",
        "content": 'Used tool: execute_python with args: {"script_path": "process_timeline.py"}',
    }
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {
        "type": "tool_use",
        "tool": "execute_python",
        "content": {"script_path": "process_timeline.py"},
    }


def test_parses_nested_tool_json_inside_llm_reply_content(agent):
    obj = {
        "type": "llm_reply",
        "content": '{"type":"tool_use","tool":"run_shell","content":{"command":"pwd"}}',
    }
    normalized = agent._normalize_text_tool_call(obj)
    assert normalized == {
        "type": "tool_use",
        "tool": "run_shell",
        "content": {"command": "pwd"},
    }


def test_get_next_action_prefers_used_tool_text_before_inner_json(agent, monkeypatch):
    # Reproduces provider output where text includes "Used tool: ... with args: {...}".
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='Used tool: read_file with args: {"path":"dependencies.csv"}'
                ),
                logprobs=None,
            )
        ],
        usage=None,
    )

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        lambda **kwargs: response,
    )

    action = agent.get_next_action([{"role": "user", "content": "continue"}])
    assert action["type"] == "tool_use"
    assert action["tool"] == "read_file"
    assert action["content"] == {"path": "dependencies.csv"}
