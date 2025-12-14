from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Perturbation(BaseModel):
    step: int
    type: str
    instruction: str

class AgentAction(BaseModel):
    type: str
    tool: Optional[str] = None
    content: Any = None  # Arguments for the tool or content for llm_reply
    logprobs: Optional[List[float]] = None

class GoldenStep(BaseModel):
    step_description: str
    agent_action: AgentAction
    tool_output: Optional[str] = None
    expected_file_changes: Optional[Dict[str, str]] = Field(default_factory=dict)

class Scenario(BaseModel):
    id: str
    name: str
    initial_prompt: str
    description: str
    ground_truth_goal: Optional[str] = None
    golden_path: List[GoldenStep] = Field(default_factory=list)
    perturbations: List[Perturbation] = Field(default_factory=list)
    image_name: Optional[str] = None
