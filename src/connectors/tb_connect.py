import docker
import time
import os
import io
import tarfile

class TerminalBenchConnector:
    """
    Provides a simple interface for External Agents to interact with the TerminalBench environment.
    Manages the Docker container lifecycle for a specific task.
    """
    def __init__(self, task_id: str, image_name: str = "python:3.11-slim"):
        self.task_id = task_id
        self.image_name = image_name
        self.client = docker.from_env()
        self.container_name = f"tb_universal_{task_id}"
        self.container = None
        self.cwd = "/workspace"

    def start(self):
        """Starts the Task Environment (Docker Container)."""
        print(f"Connector: Starting environment for {self.task_id} using image {self.image_name}...")
        
        # Reset CWD
        self.cwd = "/workspace"
        
        # Resolve absolute path for the sandbox data
        # Use dynamic path based on task_id (scenario_id)
        project_root = os.path.abspath(os.getcwd())
        host_data_path = os.path.join(project_root, "data", f"sandbox_{self.task_id}")
        
        # Ensure host path exists
        if not os.path.exists(host_data_path):
             os.makedirs(host_data_path, exist_ok=True)

        # In a real impl, we would pull the correct image for the task.
        # Here we use a generic python/bash image as a placeholder for the "Sandbox".
        # We mount the task data or use the pre-built TB images.
        try:
            self.container = self.client.containers.run(
                self.image_name, 
                command="tail -f /dev/null", # Keep alive
                name=self.container_name,
                detach=True,
                auto_remove=True,
                working_dir="/workspace",
                volumes={
                    host_data_path: {'bind': '/workspace', 'mode': 'rw'} # Mount directly to workspace for simplicity
                }
            )
            print(f"Connector: Environment ready. ID: {self.container.short_id} mapped to {host_data_path}")
        except docker.errors.APIError as e:
            # If container exists, just grab it
            if "Conflict" in str(e):
                self.container = self.client.containers.get(self.container_name)
                print(f"Connector: Reconnected to existing environment.")
            else:
                raise e

    def _is_safe_path(self, path: str) -> bool:
        """
        Validates that the path is within the allowed directories (/workspace or /tmp).
        Prevents directory traversal attacks.
        """
        if not path:
            return False
            
        # Resolve relative paths against CWD
        if not path.startswith("/"):
            path = os.path.join(self.cwd, path)
            
        # Normalize path to resolve .. components
        normalized_path = os.path.normpath(path)
        
        # Check against allowed prefixes
        allowed_prefixes = ["/workspace", "/tmp"]
        is_safe = any(normalized_path.startswith(prefix) for prefix in allowed_prefixes)
        
        if not is_safe:
            print(f"Connector: Security Warning - Blocked access to unsafe path: {normalized_path}")
            
        return is_safe

    def execute_command(self, command: str, timeout: int = 30):
        """
        Executes a shell command in the sandbox and returns output, maintaining CWD.
        Includes a timeout mechanism to prevent hanging processes.
        """
        if not self.container:
            raise RuntimeError("Environment not started. Call start() first.")
            
        print(f"Connector: Executing '{command}' in {self.cwd}...")
        
        # Handle 'cd' command specifically to update internal state
        if command.strip().startswith("cd "):
            target_dir = command.strip()[3:].strip()
            # Verify directory exists and get absolute path
            # We chain: go to current cwd -> try cd to target -> print pwd
            check_cmd = f"cd {self.cwd} && cd {target_dir} && pwd"
            
            # Use timeout command for safety
            safe_cmd = f"timeout {timeout}s /bin/bash -c \"{check_cmd}\""
            exec_result = self.container.exec_run(safe_cmd)
            
            output = exec_result.output.decode("utf-8").strip()
            exit_code = exec_result.exit_code
            
            if exit_code == 124:
                return 124, "Command timed out."
            
            if exit_code == 0:
                self.cwd = output
                return 0, f"Changed directory to {self.cwd}"
            else:
                return exit_code, f"cd: {target_dir}: No such file or directory"

        # For all other commands, execute in current CWD
        full_cmd = f"cd {self.cwd} && {command}"
        
        # Wrap in timeout command
        safe_cmd = f"timeout {timeout}s /bin/bash -c \"{full_cmd}\""
        
        exec_result = self.container.exec_run(safe_cmd)
        output = exec_result.output.decode("utf-8")
        exit_code = exec_result.exit_code
        
        if exit_code == 124:
             return 124, f"Command timed out after {timeout} seconds."
             
        return exit_code, output

    def write_file(self, path: str, content: str) -> bool:
        """Writes content to a file inside the container."""
        if not self.container:
            raise RuntimeError("Environment not started. Call start() first.")
        
        if not self._is_safe_path(path):
            print(f"Connector: Blocked write to unsafe path '{path}'")
            return False

        print(f"Connector: Writing to file '{path}'...")
        try:
            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                # Create a TarInfo object for the file
                tar_info = tarfile.TarInfo(name=os.path.basename(path))
                encoded_content = content.encode('utf-8')
                tar_info.size = len(encoded_content)
                tar.addfile(tar_info, io.BytesIO(encoded_content))
            
            tar_stream.seek(0)
            
            # Determine the directory to put the file in
            dir_path = os.path.dirname(path)
            if not dir_path:
                dir_path = self.cwd
            
            # Ensure the directory exists
            self.execute_command(f"mkdir -p {dir_path}")
            
            self.container.put_archive(dir_path, tar_stream)
            return True
        except Exception as e:
            print(f"Connector: Error writing file: {e}")
            return False

    def read_file(self, path: str) -> str:
        """Reads a file from the container."""
        if not self.container:
            raise RuntimeError("Environment not started. Call start() first.")
            
        if not self._is_safe_path(path):
            return "Error: Access denied (unsafe path)."
            
        print(f"Connector: Reading file '{path}'...")
        # Using cat is simpler for text files than get_archive processing
        exit_code, output = self.execute_command(f"cat {path}", timeout=10)
        if exit_code != 0:
            return f"Error reading file: {output}"
        return output

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
