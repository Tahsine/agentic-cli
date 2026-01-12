import typer
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load ENV before imports that might check it
load_dotenv()

from src.core.session import SessionManager
from src.graphs.main import create_main_graph

app = typer.Typer(name="2g", help="Agentic CLI")
console = Console()

def run_graph_sync(graph, input_message: str, config: dict):
    """Refactored async wrapper for sync typer."""
    
    events = graph.stream(
        {"messages": [("user", input_message)]},
        config,
        stream_mode="values"
    )
    
    final_state = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description="Thinking...", total=None)
        
        for event in events:
            final_state = event
            
            if "plan" in event and event.get("current_step_index", 0) > 0:
                progress.update(task_id, description="Executing plan...")
            elif "research_outputs" in event:
                progress.update(task_id, description="Researching...")
            elif "_route_target" in event:
                 progress.update(task_id, description=f"Routing to {event['_route_target']}...")
    
    return final_state

def display_output(state: dict):
    """Render the final state to the user."""
    if not state:
        return

    messages = state.get("messages", [])
    if messages:
        # Only show the last message which is the AI response
        last_msg = messages[-1]
        if hasattr(last_msg, 'content') and last_msg.type == 'ai':
             text_content = last_msg.content
             # Gemini 3 / LangChain compatibility
             if hasattr(last_msg, "text") and last_msg.text:
                  text_content = last_msg.text
                  
             console.print(Panel(Markdown(str(text_content)), title="Agent", border_style="green"))

    execution_history = state.get("execution_history", [])
    if execution_history:
        console.print("\n[bold]Execution Log:[/bold]")
        for item in execution_history:
            status_color = "green" if item.get("return_code") == 0 else "red"
            console.print(f"[{status_color}]➜ {item.get('command')}[/{status_color}]")
            if item.get("output"):
                 console.print(f"[dim]{item.get('output').strip()}[/dim]")

def process_turn(graph, message: str, config: dict, dry_run: bool):
    """Process a single turn of conversation/execution."""
    try:
        final_state = run_graph_sync(graph, message, config)
        display_output(final_state)
        
        # Check if paused (interrupt)
        snapshot = graph.get_state(config)
        if snapshot.next:
            console.print("\n[yellow]⚠ Plan requires approval:[/yellow]")
            
            current_state = snapshot.values
            plan = current_state.get("plan", [])
            for step in plan:
                console.print(f"[bold]{step['id']}. {step['description']}[/bold] ({step['risk_level']})")
                console.print(f"   [dim]{step['command']}[/dim]")
            
            if dry_run:
                console.print("[blue]Dry run complete.[/blue]")
                return

            confirm = typer.confirm("Do you want to proceed with execution?")
            
            if confirm:
                # Update state to validated
                graph.update_state(config, {"user_validated": True})
                
                console.print("[green]Resuming execution...[/green]")
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    final_state = event
                
                display_output(final_state)
            else:
                console.print("[red]Execution cancelled. Plan discarded.[/red]")
                # Ideally we might want to clean up the plan in state?
                
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

@app.command()
def version():
    """Print the version."""
    console.print("Agentic CLI v0.1.0")

@app.command()
def start(
    session_id: str = typer.Argument("default", help="Name of the session"),
    dry_run: bool = typer.Option(False, help="Show plan without executing"),
    # message: str = typer.Option(None, "--message", "-m", help="Example: 'List files'") # Commented out one-shot
):
    """
    Start the Agentic CLI in interactive mode.
    """
    session_manager = SessionManager()
    checkpointer = session_manager.get_checkpointer()
    
    workflow = create_main_graph()
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["executor"]
    ) 
    
    config = {"configurable": {"thread_id": session_id}}
    
    console.print(f"[bold blue]Welcome to Agentic CLI (2g)[/bold blue]")
    console.print(f"Session: [cyan]{session_id}[/cyan]")
    
    # Interactive mode
    console.print("[dim]Type 'exit' or 'quit' to leave.[/dim]")
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            if user_input.lower() in ["exit", "quit"]:
                console.print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            process_turn(graph, user_input, config, dry_run)
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break

if __name__ == "__main__":
    app()
