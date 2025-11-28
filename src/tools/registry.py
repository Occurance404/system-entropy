from typing import Dict, Optional
from src.tools.base import ToolProtocol
from src.tools.definitions import ReadFileTool, WriteFileTool, ExecutePythonTool, RunShellTool, SearchWebTool

class ToolRegistry:
    """
    Central registry for available tools.
    """
    def __init__(self):
        self._tools: Dict[str, ToolProtocol] = {}
        self._register_defaults()
        
    def _register_defaults(self):
        self.register(ReadFileTool())
        self.register(WriteFileTool())
        self.register(ExecutePythonTool())
        self.register(RunShellTool())
        self.register(SearchWebTool())
        
    def register(self, tool: ToolProtocol):
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[ToolProtocol]:
        return self._tools.get(name)
