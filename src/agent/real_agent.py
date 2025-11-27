import os
import json
import math
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.agent.wrapper import AgentWrapper

class OpenAICompatibleAgent(AgentWrapper):
    """
    Implementation for OpenAI-compatible APIs (including vLLM).
    Connects to a remote server to generate text and logprobs.
    """
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        )
        # Define a clear system message to orient the agent as a task executor
        self.system_message = {
            "role": "system",
            "content": "You are an autonomous AI agent designed to execute tasks by using available tools and responding directly with actions or answers. Your goal is to complete the given task efficiently and accurately. Do not engage in conversational chitchat or ask for clarification unless absolutely necessary for task execution. If you need to perform an action, use the tool functions provided. If the task is complete, provide a final summary."
        }
        
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads a file from the filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Writes content to a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executes a python script.",
                    "parameters": {
                        "type": "object",
                        "properties": {"script_path": {"type": "string"}},
                        "required": ["script_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Searches the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Executes a shell command in the sandbox. Use this for navigating directories, running tests, or managing files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }
        ]

    def get_next_action(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Fetches the next action from the LLM.
        """
        # Prepend the system message to the history
        messages = [self.system_message] + [{"role": msg["role"], "content": msg["content"]} for msg in history if msg["role"] in ["system", "user", "assistant"]]
        if not messages:
            messages = [{"role": "user", "content": "Begin the task."}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                tools=self.tools_schema,
                tool_choice="auto",
                logprobs=True, 
                top_logprobs=1
            )
            
            choice = response.choices[0]
            message = choice.message
            
            token_logprobs = []
            if choice.logprobs and choice.logprobs.content:
                token_logprobs = [token.logprob for token in choice.logprobs.content]

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                return {
                    "type": "tool_use",
                    "tool": tool_call.function.name,
                    "content": json.loads(tool_call.function.arguments),
                    "logprobs": token_logprobs
                }
            else:
                return {
                    "type": "llm_reply",
                    "content": message.content,
                    "logprobs": token_logprobs
                }

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"type": "llm_reply", "content": f"Error: {str(e)}", "logprobs": [0.0]}

    def generate_multiple(self, history: List[Dict], n: int = 5) -> List[Dict[str, Any]]:
        """
        Generates N divergent responses for Branching Probe.
        Uses Sequential Branching (looping n times) to ensure compatibility 
        with APIs that don't support n > 1 (like DeepSeek).
        """
        # Prepend the system message to the history
        messages = [self.system_message] + [{"role": msg["role"], "content": msg["content"]} for msg in history if msg["role"] in ["system", "user", "assistant"]]
        
        branches = []
        for i in range(n):
            try:
                # Call API n times sequentially
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.9, # High temp for divergence
                    n=1, # Always 1 for compatibility
                    logprobs=True,
                    top_logprobs=1
                )
                
                choice = response.choices[0]
                content = choice.message.content
                token_logprobs = []
                if choice.logprobs and choice.logprobs.content:
                    token_logprobs = [token.logprob for token in choice.logprobs.content]
                
                branches.append({
                    "type": "thought",
                    "content": content,
                    "logprobs": token_logprobs
                })
            except Exception as e:
                print(f"Error in branching probe iteration {i+1}/{n}: {e}")
                # Continue to try getting other branches even if one fails
                continue
                
        return branches
