from typing import Dict, List, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.core.state import GlobalState, PlanStep
import json

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

PLANNER_PROMPT = """
You are an expert technical planner for a CLI Agent.
Your goal is to break down a high-level user objective into a series of safe, executable atomic steps.

Analyze the user's request and the current state.
Generate a JSON plan where each item is a step.

Each step MUST have:
- id: unique string (e.g., "step_1")
- description: clear explanation of what we are doing
- command: the exact shell command to run (or empty if it's a manual/thought step)
- risk_level: "LOW" (read-only), "MEDIUM" (creates files), "HIGH" (modifies/deletes), "CRITICAL" (system changes)

OUTPUT FORMAT:
Return a JSON array of objects. Do not wrap in markdown code blocks.
Example:
[
  {"id": "1", "description": "List files", "command": "ls -la", "risk_level": "LOW", "status": "pending"},
  {"id": "2", "description": "Read file", "command": "cat README.md", "risk_level": "LOW", "status": "pending"}
]
"""

def draft_plan(state: GlobalState) -> Dict:
    """Node: Generates the initial plan based on user messages."""
    messages = state["messages"]
    # Extract the user's objective from the messages
    # In a real scenario, we might summarization history
    
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        *messages,
        HumanMessage(content="Generate a plan for the above request.")
    ])
    
    try:
        # Clean up potential markdown wrapper from LLM
        content = response.content.replace("```json", "").replace("```", "").strip()
        plan_data = json.loads(content)
        # Ensure default status
        for step in plan_data:
            if "status" not in step:
                step["status"] = "pending"
        
        return {"plan": plan_data, "current_step_index": 0, "user_validated": False}
    except Exception as e:
        # Fallback or error handling
        return {
            "plan": [], 
            "messages": [SystemMessage(content=f"Error generating plan: {str(e)}")]
        }

def plan_refiner(state: GlobalState) -> Dict:
    """Node: Updates the plan based on user feedback."""
    # This node is entered if the user rejects the plan in the 'HumanInterrupt' logic
    # The last message from the user contains the feedback
    messages = state["messages"]
    
    feedback_prompt = f"""
    The user rejected the previous plan. 
    Current Plan: {json.dumps(state['plan'])}
    User Feedback: {messages[-1].content}
    
    Update the plan accordingly. Return the full updated JSON array.
    """
    
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=feedback_prompt)
    ])
    
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        plan_data = json.loads(content)
        for step in plan_data:
             if "status" not in step: step["status"] = "pending"
             
        return {"plan": plan_data, "user_validated": False}
    except Exception as e:
        return {"messages": [SystemMessage(content=f"Error refining plan: {str(e)}")]}

def create_planner_graph():
    """Builds the Planner Sub-graph."""
    workflow = StateGraph(GlobalState)
    
    workflow.add_node("draft_plan", draft_plan)
    workflow.add_node("plan_refiner", plan_refiner)
    
    # We don't add 'human_interrupt' as a node per se to the graph structure 
    # if we use LangGraph's .compile(interrupt_before=[...]) feature.
    # However, to be explicit in the logic logic flows:
    
    workflow.set_entry_point("draft_plan")
    
    # The 'human_interrupt' logic effectively happens at the edge or via the main graph's control flow.
    # For a sub-graph, we simply return the plan. The interaction happens in the Main Graph.
    # But effectively, after drafting, we consider the planning 'done' for this sub-graph,
    # and the Main Graph will decide to interrupt before Execution.
    
    workflow.add_edge("draft_plan", END)
    workflow.add_edge("plan_refiner", END)
    
    return workflow.compile()
