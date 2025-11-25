import sys
print(f"sys.path: {sys.path}")

from terminal_bench.llms.lite_llm import LiteLLM
print(f"LiteLLM file: {LiteLLM.__module__}")
import inspect
print(f"LiteLLM location: {inspect.getfile(LiteLLM)}")

from terminal_bench.agents.naive_agent import NaiveAgent
try:
    agent = NaiveAgent(model_name="test")
    print("Success! Created NaiveAgent with model_name only.")
except Exception as e:
    print(f"Failed: {e}")

try:
    llm = LiteLLM(model_name="test")
    agent = NaiveAgent(llm=llm)
    print("Success! Created NaiveAgent with llm.")
except Exception as e:
    print(f"Failed with llm: {e}")
