"""
main.py — CLI entrypoint. Run with:
    python main.py "What are the latest breakthroughs in fusion energy?"
"""
from __future__ import annotations
import sys
import logging
import uuid
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import get_settings
from graph import get_graph
from models import ResearchState
from memory.store import VectorMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def run_research(query: str) -> ResearchState:
    """Execute the full multi-agent pipeline and return the final state."""
    settings = get_settings()
    graph    = get_graph()

    initial_state = ResearchState(original_query=query)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Running agents...", total=None)
        final_state = graph.invoke(initial_state)
        progress.update(task, description="Done!")

    return ResearchState(**final_state)


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/] python main.py \"Your research question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    console.rule(f"[bold]Multi-Agent Research[/bold]")
    console.print(f"[dim]Query:[/dim] {query}\n")

    try:
        state = run_research(query)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    # Display results
    console.rule("[bold green]Answer[/bold green]")
    console.print(Markdown(state.final_answer or state.draft))

    console.rule("[bold]Metadata[/bold]")
    console.print(f"  Confidence score : [bold]{state.confidence_score}/10[/bold]")
    console.print(f"  Sources cited    : {len(state.sources)}")
    console.print(f"  Search iterations: {state.search_iterations}")
    console.print(f"  Critic retries   : {state.critic_retries}")

    if state.sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, url in enumerate(state.sources, 1):
            console.print(f"  [{i}] {url}")

    # Persist to long-term memory
    try:
        vm = VectorMemory(persist_dir=get_settings().chroma_persist_dir)
        vm.save_session(
            query=query,
            answer=state.final_answer or state.draft,
            session_id=str(uuid.uuid4()),
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
