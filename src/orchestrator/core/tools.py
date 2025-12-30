from typing import Any, Dict, Optional, Tuple


def execute_tool_and_measure(*, tool_registry: Any, connector: Any, action_intent: Dict[str, Any]) -> Tuple[str, Optional[int], None]:
    """
    Executes a tool via ToolRegistry.
    """
    tool_name = action_intent["tool"]
    tool_args = action_intent["content"]

    tool = tool_registry.get_tool(tool_name)
    if not tool:
        return f"Unknown tool: {tool_name}", None, None

    result, cbf = tool.execute(tool_args, connector)
    return result, cbf, None

