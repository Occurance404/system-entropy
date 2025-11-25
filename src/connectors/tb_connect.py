import docker
import time
import os

class TerminalBenchConnector:
    """
    Provides a simple interface for External Agents to interact with the TerminalBench environment.
    Manages the Docker container lifecycle for a specific task.
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.client = docker.from_env()
        self.container_name = f"tb_universal_{task_id}"
        self.container = None

    def start(self):
        """Starts the Task Environment (Docker Container)."""
        print(f"Connector: Starting environment for {self.task_id}...")
        
        # In a real impl, we would pull the correct image for the task.
        # Here we use a generic python/bash image as a placeholder for the "Sandbox".
        # We mount the task data or use the pre-built TB images.
        try:
            self.container = self.client.containers.run(
                "python:3.11-slim", # Replace with actual TerminalBench image
                command="tail -f /dev/null", # Keep alive
                name=self.container_name,
                detach=True,
                auto_remove=True,
                working_dir="/workspace"
            )
            print(f"Connector: Environment ready. ID: {self.container.short_id}")
        except docker.errors.APIError as e:
            # If container exists, just grab it
            if "Conflict" in str(e):
                self.container = self.client.containers.get(self.container_name)
                print(f"Connector: Reconnected to existing environment.")
            else:
                raise e

    def execute_command(self, command: str):
        """Executes a shell command in the sandbox and returns output."""
        if not self.container:
            raise RuntimeError("Environment not started. Call start() first.")
            
        print(f"Connector: Executing '{command}'...")
        # Explicitly run command via bash to ensure shell features like redirection work
        exec_result = self.container.exec_run(f"/bin/bash -c \"{command}\"")
        output = exec_result.output.decode("utf-8")
        exit_code = exec_result.exit_code
        return exit_code, output

    def stop(self):
        """Cleans up the environment."""
        if self.container:
            print("Connector: Stopping environment...")
            self.container.stop()

if __name__ == "__main__":
    # Test
    conn = TerminalBenchConnector("test_task")
    conn.start()
    code, out = conn.execute_command("echo 'Hello from Sandbox!'")
    print(f"Output: {out.strip()}")
    conn.stop()
