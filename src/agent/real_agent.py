import os
import json
import math
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
from src.agent.wrapper import AgentWrapper

class OpenAICompatibleAgent(AgentWrapper):
    """
    Implementation for OpenAI-compatible APIs (including vLLM).
    Connects to a remote server to generate text and logprobs.
    Includes Async Acceleration for Branching Probes.
    """
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        
        # Sync client for standard steps (to keep Orchestrator simple)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Async client for parallel probing
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
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
        Fetches the next action from the LLM using the Synchronous Client.
        """
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
        Generates N divergent responses for Branching Probe in PARALLEL.
        Wraps async calls in a synchronous runner to maintain Protocol compatibility.
        """
        return asyncio.run(self._generate_multiple_async(history, n))

    async def _generate_multiple_async(self, history: List[Dict], n: int) -> List[Dict[str, Any]]:
        """
        Internal Async implementation of Branching Probe.
        """
        messages = [self.system_message] + [{"role": msg["role"], "content": msg["content"]} for msg in history if msg["role"] in ["system", "user", "assistant"]]
        
        # Create N parallel tasks
        tasks = [self._generate_one_async(messages, i, n) for i in range(n)]
        
        # Wait for all to complete
        branches = await asyncio.gather(*tasks)
        
        # Filter out Nones (failed attempts)
        valid_branches = [b for b in branches if b is not None]
        
        # If all failed, return empty list
        return valid_branches

    async def _generate_one_async(self, messages: List[Dict], index: int, total: int) -> Optional[Dict[str, Any]]:
        """
        Helper for a single async generation.
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.9, # High temp for divergence
                n=1, 
                logprobs=True,
                top_logprobs=1
            )
            
            choice = response.choices[0]
            content = choice.message.content
            token_logprobs = []
            if choice.logprobs and choice.logprobs.content:
                token_logprobs = [token.logprob for token in choice.logprobs.content]
            
            return {
                "type": "thought",
                "content": content,
                "logprobs": token_logprobs
            }
        except Exception as e:
            print(f"Error in async probe {index+1}/{total}: {e}")
            return None