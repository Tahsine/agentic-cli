import subprocess
import shlex
import threading
from typing import Tuple, Optional

class SafeSubprocess:
    def __init__(self, default_timeout: int = 60):
        self.default_timeout = default_timeout

    def run(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """
        Executes a shell command safely with a timeout.
        Returns: (return_code, stdout, stderr)
        """
        # Security check: basic strict blocking of obviously dangerous commands not caught by LLM
        # NOTE: The primary safety check should be in the SafetyGuard node.
        # This is a last-resort fail-safe.
        # Windows/Linux agnostic basic checks.
        forbidden_patterns = ["rm -rf /", "format c:", "rd /s /q c:\\"] 
        for pattern in forbidden_patterns:
            if pattern in command.lower():
                return -1, "", f"CRITICAL SECURITY: Command blocked by SafeSubprocess: {command}"

        timeout_val = timeout if timeout is not None else self.default_timeout

        try:
            # shell=True is needed for complex commands (pipes, redirects) but adds risk.
            # We mitigate this by validating inputs in the Agent graph upstream.
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace' # Prevent decoding errors from crashing
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_val)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                process.kill()
                # Try to get partial output
                outs, errs = process.communicate()
                return 124, outs or "", f"Command timed out after {timeout_val}s. {errs or ''}"

        except Exception as e:
            return -1, "", f"Execution failed: {str(e)}"
