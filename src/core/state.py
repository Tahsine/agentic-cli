from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class PlanStep(TypedDict):
    id: str
    description: str
    command: Optional[str]
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    status: str      # "pending", "in_progress", "done", "failed"
    output: Optional[str]

class GlobalState(TypedDict):
    # Chat history with the user (Human + AI messages)
    messages: Annotated[List[dict], add_messages]
    
    # The action plan
    plan: List[PlanStep]
    
    # Current execution pointer
    current_step_index: int
    
    # Research findings
    research_outputs: List[str]
    
    # Raw execution logs (command + stdout/stderr)
    execution_history: List[Dict[str, Any]]
    
    # Safety flag: has the user approved the CURRENT plan?
    user_validated: bool
    
    # Context cache for read files to save tokens
    file_context: Dict[str, str]
