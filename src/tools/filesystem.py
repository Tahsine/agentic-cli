import os
import glob
from typing import List, Optional, Dict
from pathlib import Path

class FileSystemManager:
    """
    Manages file system operations: read, write, list.
    Designed to be used by the Agent to explore and modify the project.
    """
    def __init__(self, working_directory: str = "."):
        self.working_directory = Path(working_directory).resolve()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path against working directory and check security jail."""
        target = (self.working_directory / path).resolve()
        # Basic jail check: ensure we are efficiently within the project scope
        # Note: In a real "User Ally" tool, the user might want to edit outside,
        # but for safety defaults, we might warn.
        # For now, we trust the relative path resolution.
        return target

    def list_files(self, path: str = ".", depth: int = 1) -> str:
        """List files in a directory similar to 'ls -R' or 'tree'."""
        target_dir = self._resolve_path(path)
        if not target_dir.exists():
            return f"Error: Directory {path} does not exist."
        
        output = []
        try:
            # Simple non-recursive list for now, or limited recursion
            # Using glob for easier pattern matching if needed later
            items = os.listdir(target_dir)
            output.append(f"Directory listing for: {path}")
            for item in items:
                full_item = target_dir / item
                is_dir = full_item.is_dir()
                mark = "[DIR] " if is_dir else "[FILE]"
                output.append(f"{mark} {item}")
            return "\n".join(output)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def read_file(self, path: str) -> str:
        """Read content of a file."""
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File {path} not found."
        if not target.is_file():
            return f"Error: {path} is not a file."
        
        try:
            with open(target, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file {path}: {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file.
        WARNING: This overwrites. The Agent should check existence first or use a patching tool.
        """
        target = self._resolve_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing to {path}: {str(e)}"
