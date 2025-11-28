from typing import Protocol, Dict, Any, Tuple, Optional
from src.interfaces import SandboxConnectorProtocol

class ToolProtocol(Protocol):
    """Interface for a Tool."""
    name: str
    description: str
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        """
        Executes the tool.
        Returns: (Output String, Optional CBF/Complexity Score)
        """
        ...
