import os
import sys
import openai
from src.connectors.tb_connect import TerminalBenchConnector

# This is a "Universal Agent" - just a standard Python script.
# It doesn't know it's being tested. It just sees:
# 1. An OpenAI Endpoint (which is actually our Proxy)
# 2. A Shell (provided by our Connector)

def run_agent():
    print("--- Universal Agent Starting ---")
    
    # 1. Connect to the Environment
    task_id = os.getenv("TB_TASK_ID", "unknown_task")
    connector = TerminalBenchConnector(task_id)
    connector.start()
    
    # 2. Connect to the "LLM" (Proxy)
    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"Agent: Connected to LLM at {client.base_url}")

    # 3. The Agent Loop (Iterative)
    # Goal: Create a file called 'solution.txt' with the answer to life.
    
    messages = [
        {"role": "system", "content": "You are a helpful agent. You can execute bash commands by wrapping them in ```bash ... ``` blocks."},
        {"role": "user", "content": "Organize the files in 'data/sandbox_task_1'. Move all .jpg files to the 'images' folder."}
    ]
    
    for turn in range(6): # Run for 6 turns to allow for shock reaction
        print(f"\n--- Turn {turn + 1} ---")
        
        # Ask LLM
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1
            )
        except Exception as e:
            print(f"Agent Error: {e}")
            break
            
        llm_msg = response.choices[0].message
        content = llm_msg.content
        print(f"Agent: LLM says -> {content[:100]}...")
        
        # Add to history
        messages.append(llm_msg)
        
        # Parse for commands
        if "```bash" in content:
            cmd = content.split("```bash")[1].split("```")[0].strip()
            print(f"Agent: Executing -> {cmd}")
            
            # Execute
            exit_code, output = connector.execute_command(cmd)
            print(f"Agent: Result -> {output[:100]}...")
            
            # Feed result back to LLM
            messages.append({
                "role": "user",
                "content": f"Command executed. Exit code: {exit_code}. Output:\n{output}"
            })
        else:
            print("Agent: No command found. Waiting for next turn...")
            # If no command, maybe it's just talking. Let's prompt it to continue or stop.
            if "done" in content.lower():
                print("Agent: Job appears done.")
                break
            
            # If it didn't run a command, we just continue to see if the Shock hit
            # or if it needs to think more.
            pass
    
    connector.stop()
    print("--- Universal Agent Finished ---")

if __name__ == "__main__":
    run_agent()
