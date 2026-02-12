import pytest

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
