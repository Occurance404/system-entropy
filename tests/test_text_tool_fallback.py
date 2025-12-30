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

