import ast
from typing import Dict, Any, Tuple, Optional
from src.tools.base import ToolProtocol
from src.interfaces import SandboxConnectorProtocol

class ReadFileTool(ToolProtocol):
    name = "read_file"
    description = "Reads a file from the filesystem."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        path = args.get('path', '')
        return connector.read_file(path), None

class WriteFileTool(ToolProtocol):
    name = "write_file"
    description = "Writes content to a file."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        path = args.get('path', '')
        content = args.get('content', '')
        success = connector.write_file(path, content)
        result = "File written successfully." if success else "Failed to write file."
        
        # Calculate CBF if python file
        cbf = None
        if path.endswith('.py'):
            cbf = self._measure_cbf(content)
            
        return result, cbf

    def _measure_cbf(self, code: str) -> int:
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity += 1
            return complexity
        except:
            return -1

class ExecutePythonTool(ToolProtocol):
    name = "execute_python"
    description = "Executes a python script."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        script_path = args.get('script_path', '')
        exit_code, output = connector.execute_command(f"python3 {script_path}")
        return f"Exit Code: {exit_code}\nOutput:\n{output}", None

class RunShellTool(ToolProtocol):
    name = "run_shell"
    description = "Executes a shell command."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        command = args.get('command', '')
        exit_code, output = connector.execute_command(command)
        return f"Exit Code: {exit_code}\nOutput:\n{output}", None

class SearchWebTool(ToolProtocol):
    name = "search_web"
    description = "Searches the web."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        return "Search is currently disabled in the sandbox environment.", None
