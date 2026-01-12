import sqlite3
from typing import Any, Optional
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path

class SessionManager:
    """
    Manages persistence for LangGraph using SQLite.
    Enables 'Time Travel' by saving state at every step.
    """
    def __init__(self, db_path: str = ".2giants/sessions.db"):
        self.db_path = Path(db_path).resolve()
        
    def setup(self):
        """Ensure the DB directory and connection exist."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        # The connection will be managed by SqliteSaver
        # We just ensure the path is ready.
        return self.db_path

    def get_checkpointer(self) -> SqliteSaver:
        """Returns a configured SqliteSaver for the graph."""
        self.setup()
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return SqliteSaver(conn)

    def list_sessions(self):
        """Not yet implemented: list available session threads."""
        pass
