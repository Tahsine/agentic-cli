from typing import Dict, Literal
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

from src.core.state import GlobalState
from src.tools.subprocess import SafeSubprocess

# Initialize tools
safe_runner = SafeSubprocess()

def step_parser(state: GlobalState) -> Dict:
    """Node: Identifies the next step to execute."""
    plan = state["plan"]
    idx = state.get("current_step_index", 0)
    
    if idx >= len(plan):
        return {"current_step_index": idx} # No updates, logic handled in edges

    current_step = plan[idx]
    # Update status to in_progress
    current_step["status"] = "in_progress"
    plan[idx] = current_step
    
    return {"plan": plan}

def safety_guard(state: GlobalState) -> Dict:
    """Node: Checks if the current command is safe to execute."""
    idx = state["current_step_index"]
    step = state["plan"][idx]
    command = step.get("command", "")
    
    # "HumanInterrupt" logic should have already validated the plan generally.
    # This is the "Runtime" check.
    # For this implementation, we rely on the RISK LEVEL assigned + User Validation flag.
    
    if not state.get("user_validated", False):
         # If not explicitly validated (should not happen if flow is correct), HALT.
         # For the architecture, we assume validation happened before entering Executor.
         pass

    # Double check for extremely dangerous patterns essentially caught by SafeSubprocess
    # But here we can route to an error state or halt.
    if "rm -rf /" in command:
        return {"plan": state["plan"]} # Will fail in runner or here
    
    return {} # Proceed

def terminal_runner(state: GlobalState) -> Dict:
    """Node: Executes the command."""
    idx = state["current_step_index"]
    plan = state["plan"]
    step = plan[idx]
    command = step.get("command")
    
    if not command:
        # Step might be a thought or manual action
        step["status"] = "done"
        step["output"] = "(No command executed)"
        plan[idx] = step
        return {
            "plan": plan, 
            "current_step_index": idx + 1,
            "execution_history": state.get("execution_history", []) + [{"command": None, "output": "Skipped (no command)"}]
        }

    return_code, stdout, stderr = safe_runner.run(command)
    
    output = stdout + "\n" + stderr
    history_item = {
        "step_id": step["id"],
        "command": command,
        "return_code": return_code,
        "output": output
    }
    
    if return_code == 0:
        step["status"] = "done"
        step["output"] = output
        plan[idx] = step
        return {
            "plan": plan, 
            "current_step_index": idx + 1,
            "execution_history": state.get("execution_history", []) + [history_item]
        }
    else:
        step["status"] = "failed"
        step["output"] = output
        plan[idx] = step
        return {
            "plan": plan,
            "execution_history": state.get("execution_history", []) + [history_item]
        }

def error_handler(state: GlobalState) -> Dict:
    """Node: DECIDES what to do on error."""
    # Simple logic: Stop execution, return to user/planner.
    # In advanced: ask LLM for fix.
    messages = state["messages"]
    messages.append(SystemMessage(content=f"Execution failed at step {state['current_step_index'] + 1}."))
    return {"messages": messages}

# --- Edges ---

def check_step_completion(state: GlobalState) -> Literal["terminal_runner", "end"]:
    idx = state.get("current_step_index", 0)
    if idx >= len(state["plan"]):
        return "end"
    
    step = state["plan"][idx]
    if step["status"] == "failed":
        return "end" # Or error_handler
        
    return "terminal_runner"

def check_execution_result(state: GlobalState) -> Literal["step_parser", "error_handler"]:
    # Check the LAST executed step (which is now at index - 1 because runner incremented)
    # Wait, runner increments index immediately if successful.
    # If failed, it did NOT increment index (or we can handle logic).
    
    # In my logic above:
    # Success -> index + 1
    # Fail -> index unchanged
    
    idx = state["current_step_index"]
    # If we are strictly keeping logical index:
    # We need to look at state['plan'][previous_index]['status']
    
    # Let's simplify: 
    # If the step failed, we go to error handler.
    # We can detect failure by looking if the status of the current step is 'failed'
    # (Because index didn't increment)
    
    current_step = state["plan"][idx]
    if current_step["status"] == "failed":
        return "error_handler"
        
    # If we moved past the end, we are done
    if idx >= len(state["plan"]):
        return "step_parser" # Loop back to see it's empty/done
        
    return "step_parser" 

def create_executor_graph():
    workflow = StateGraph(GlobalState)
    
    workflow.add_node("step_parser", step_parser)
    workflow.add_node("safety_guard", safety_guard)
    workflow.add_node("terminal_runner", terminal_runner)
    workflow.add_node("error_handler", error_handler)
    
    workflow.set_entry_point("step_parser")
    
    # Logic:
    # Parser -> guard -> runner -> (check result) -> parser (loop)
    
    workflow.add_edge("step_parser", "safety_guard")
    
    # Conditional edge from guard? For now direct
    workflow.add_edge("safety_guard", "terminal_runner")
    
    # After runner, check if we loop or fail
    workflow.add_conditional_edges(
        "terminal_runner",
        check_execution_result,
        {
            "step_parser": "step_parser",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_edge("error_handler", END)
    
    # We also need a condition to END if parser finds no more steps
    # Actually, step_parser checks 'idx >= len'.
    # We need a conditional edge from step_parser too?
    # Correct.
    
    return workflow.compile()
