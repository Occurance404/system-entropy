from __future__ import annotations

import os
import subprocess
from typing import Tuple


class LocalSandboxConnector:
    """
    A lightweight, non-Docker sandbox connector.

    This connector executes commands directly on the host machine inside a
    scenario sandbox directory (e.g. data/sandbox_<scenario_id>).

    It exists to remove Docker as a hard dependency for running experiments.
    """

    def __init__(self, sandbox_root: str):
        self.sandbox_root = os.path.abspath(sandbox_root)
        self.cwd = self.sandbox_root

        os.makedirs(self.sandbox_root, exist_ok=True)

    def start(self) -> None:
        # No-op for local execution.
        os.makedirs(self.sandbox_root, exist_ok=True)

    def stop(self) -> None:
        # No-op for local execution.
        return

    def _is_safe_path(self, path: str) -> bool:
        if not path:
            return False

        if path.startswith("/tmp/") or path == "/tmp":
            return True

        # Resolve relative paths against current cwd.
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)

        normalized = os.path.normpath(path)
        return normalized.startswith(os.path.join(self.sandbox_root, ""))

    def _resolve_path(self, path: str) -> str:
        if not path:
            return path

        if os.path.isabs(path):
            return os.path.normpath(path)

        return os.path.normpath(os.path.join(self.cwd, path))

    def execute_command(self, command: str, timeout: int = 300) -> Tuple[int, str]:
        """
        Executes a shell command inside the sandbox directory.

        Mimics TerminalBenchConnector semantics:
        - Maintains a persistent `cwd`.
        - Updates `cwd` only for a simple `cd <dir>` command (no chaining).
        """
        if not command:
            return 0, ""

        # Run full command under bash for parity with Docker connector.
        full_cmd = f"cd {self.cwd} && {command}"
        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", full_cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout} seconds."

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        output = stdout + (("\n" + stderr) if stderr else "")

        # Opportunistic CWD update for simple `cd`.
        stripped = command.strip()
        if (
            completed.returncode == 0
            and stripped.startswith("cd ")
            and "&&" not in stripped
            and ";" not in stripped
        ):
            raw_target = stripped[3:].strip()
            new_path = os.path.normpath(os.path.join(self.cwd, raw_target))
            if os.path.isdir(new_path):
                self.cwd = new_path

        return int(completed.returncode), output

    def read_file(self, path: str) -> str:
        if not self._is_safe_path(path):
            return "Error: Access denied (unsafe path)."

        resolved = self._resolve_path(path)
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error reading file: [Errno 2] No such file or directory: '{path}'"
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> bool:
        if not self._is_safe_path(path):
            return False

        resolved = self._resolve_path(path)
        try:
            os.makedirs(os.path.dirname(resolved) or self.cwd, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content or "")
            return True
        except Exception:
            return False

