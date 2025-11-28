from typing import Protocol, List, Dict, Any, Optional, Tuple

class AgentProtocol(Protocol):
    """Interface for an autonomous agent."""
    model_name: str
    
    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        """Determines the next action based on conversation history."""
        ...
        
    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        """Generates multiple divergent responses for probing."""
        ...

class SandboxConnectorProtocol(Protocol):
    """Interface for the environment/sandbox connector."""
    def start(self) -> None:
        """Starts the sandbox environment."""
        ...
        
    def stop(self) -> None:
        """Stops and cleans up the sandbox environment."""
        ...
        
    def execute_command(self, command: str, timeout: int = 30) -> Tuple[int, str]:
        """Executes a shell command."""
        ...
        
    def read_file(self, path: str) -> str:
        """Reads a file's content."""
        ...
        
    def write_file(self, path: str, content: str) -> bool:
        """Writes content to a file."""
        ...

class MetricServiceProtocol(Protocol):
    """Interface for metric calculation and state monitoring."""
    
    def calculate_scr(self, branches: List[str]) -> float:
        """Calculates Semantic Collapse Ratio from text branches."""
        ...
        
    def calculate_rdi(self, current_content: str, ground_truth_text: str) -> Optional[float]:
        """Calculates Regressive Debt Index (Goal Deviance)."""
        ...
        
    def calculate_entropy(self, logprobs: List[Any]) -> float:
        """Calculates entropy from logprobs."""
        ...
        
    def calculate_ige(self, h_pre: float, h_post: float, token_cost: int) -> float:
        """Calculates Information Gain Efficiency."""
        ...
        
    def measure_cbf(self, code_snippet: str) -> int:
        """Measures Code Block Factor (Cyclomatic Complexity)."""
        ...
        
    def calculate_compression_ratio(self, text: str) -> float:
        """Calculates Compression Ratio."""
        ...
