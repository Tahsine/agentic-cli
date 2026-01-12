from typing import Dict, Literal, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.core.state import GlobalState
from src.tools.search import TavilySearch

# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
search_tool = TavilySearch()

def search_engine(state: GlobalState) -> Dict:
    """Node: Performs web search based on the last message."""
    messages = state["messages"]
    last_user_msg = messages[-1].content
    
    # Simple query extraction (can be improved with LLM)
    query = last_user_msg 
    if len(query) > 100:
        # Ask LLM to extract query
        query_resp = llm.invoke(f"Extract a concise search query from this: {last_user_msg}")
        query = query_resp.content
        
    results = search_tool.search(query)
    
    current_results = state.get("research_outputs", [])
    current_results.append(f"Query: {query}\nResult: {results}")
    
    return {"research_outputs": current_results}

def grader(state: GlobalState) -> Dict:
    """Node: Checks if the research results answer the user's request."""
    messages = state["messages"]
    research = state["research_outputs"][-1] # Check latest
    original_req = messages[0].content # Simplified: Assuming first msg is request within this sub-graph context
    
    prompt = f"""
    User Request: {original_req}
    Research Result: {research}
    
    Does the research result contain enough information to answer the request?
    Return 'YES' or 'NO'.
    """
    
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    
    # We store the decision in a temporary key or infer it in the edge
    return {"_grader_decision": decision}

def drafter(state: GlobalState) -> Dict:
    """Node: Synthesizes the answer."""
    messages = state["messages"]
    research = "\n\n".join(state["research_outputs"])
    
    prompt = f"""
    You are a researcher. Answer the user's request based strictly on the following research.
    
    Research:
    {research}
    
    User Request: {messages[-1].content}
    """
    
    response = llm.invoke(prompt)
    
    # Start a conversation response
    return {"messages": [response]}

# --- Edges ---

def check_grade(state: GlobalState) -> Literal["drafter", "search_engine", "end"]:
    decision = state.get("_grader_decision", "NO")
    attempts = len(state.get("research_outputs", []))
    
    if "YES" in decision:
        return "drafter"
    
    if attempts >= 2:
        # Give up after 2 tries and just draft with what we have
        return "drafter"
        
    return "search_engine" # Loop back (ideally with query refinement, skipped for brevity)

def create_researcher_graph():
    workflow = StateGraph(GlobalState)
    
    workflow.add_node("search_engine", search_engine)
    workflow.add_node("grader", grader)
    workflow.add_node("drafter", drafter)
    
    workflow.set_entry_point("search_engine")
    
    workflow.add_edge("search_engine", "grader")
    
    workflow.add_conditional_edges(
        "grader",
        check_grade,
        {
            "drafter": "drafter",
            "search_engine": "search_engine"
        }
    )
    
    workflow.add_edge("drafter", END)
    
    return workflow.compile()
