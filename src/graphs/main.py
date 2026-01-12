from typing import Dict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.core.state import GlobalState
from src.graphs.planner import create_planner_graph
from src.graphs.executor import create_executor_graph
from src.graphs.researcher import create_researcher_graph

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

def router(state: GlobalState) -> Dict:
    """Node: Decides where to send the request."""
    messages = state["messages"]
    last_msg = messages[-1].content
    
    prompt = f"""
    Classify the following user request into one of these categories:
    - EXECUTE: The user wants to perform an action (create files, run commands, edit code).
    - RESEARCH: The user is asking a question that requires web search or external knowledge.
    - CHAT: The user is saying hello, asking a clarification, or chatting casually.
    
    Request: {last_msg}
    
    Return ONLY the category name.
    """
    
    response = llm.invoke(prompt)
    category = response.content.strip().upper()
    
    # Simple mapping
    if "EXECUTE" in category:
        target = "planner"
    elif "RESEARCH" in category:
        target = "researcher"
    else:
        target = "chat"
        
    return {"_route_target": target}

def chat_node(state: GlobalState) -> Dict:
    """Node: Handles simple conversation."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- Edges ---

def route_request(state: GlobalState) -> Literal["planner", "researcher", "chat_node"]:
    return state.get("_route_target", "chat_node")

def create_main_graph():
    workflow = StateGraph(GlobalState)
    
    # Add Nodes
    workflow.add_node("router", router)
    workflow.add_node("chat_node", chat_node)
    
    # Sub-graphs
    workflow.add_node("planner", create_planner_graph())
    workflow.add_node("executor", create_executor_graph())
    workflow.add_node("researcher", create_researcher_graph())
    
    # Entry Point
    workflow.set_entry_point("router")
    
    # Routing Logic
    workflow.add_conditional_edges(
        "router",
        route_request,
        {
            "planner": "planner",
            "researcher": "researcher",
            "chat_node": "chat_node"
        }
    )
    
    # Executor Flow: Planner -> (Interrupt) -> Executor
    # In this unified graph, we need to handle the flow between Planner and Executor.
    # The Planner sub-graph returns a drafted plan.
    # We exit Planner, then logic check: if plan exists, go to human check then executor.
    # For now, let's keep it simple: Planner -> Executor.
    # The 'HumanInterrupt' is ideally handled by LangGraph's .compile(interrupt_before=["executor"])
    
    workflow.add_edge("planner", "executor")
    
    # Both Executor, Researcher, and Chat end the turn
    workflow.add_edge("executor", END)
    workflow.add_edge("researcher", END)
    workflow.add_edge("chat_node", END)
    
    # Compile with Memory and Interrupts
    # Note: Memory checkpointer will be injected from the outside (CLI app) 
    return workflow
