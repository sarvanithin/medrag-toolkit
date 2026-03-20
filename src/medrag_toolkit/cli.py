"""
medrag CLI — Click-based command line interface.

Commands:
  query   — run a medical query and print results
  index   — build FAISS index from knowledge sources
  serve   — start FastAPI server
  health  — check system health
"""
from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
log = structlog.get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="medrag")
def main():
    """medrag-toolkit — Production Medical RAG Framework."""


@main.command()
@click.argument("question")
@click.option("--top-k", default=10, show_default=True, help="Number of documents to retrieve")
@click.option("--stream", is_flag=True, help="Stream the answer token by token")
def query(question: str, top_k: int, stream: bool):
    """Query the medical RAG system with a QUESTION."""
    asyncio.run(_query_async(question, top_k, stream))


async def _query_async(question: str, top_k: int, stream: bool) -> None:
    from medrag_toolkit.config import Settings
    from medrag_toolkit.core import MedRAG

    config = Settings.from_file()
    config.faiss.top_k = top_k

    async with MedRAG(config) as medrag:
        if stream:
            console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
            console.print("[bold green]Answer:[/bold green] ", end="")
            async for token in medrag.stream_query(question):
                console.print(token, end="", highlight=False)
            console.print()
        else:
            with console.status("[bold green]Querying..."):
                resp = await medrag.query(question)

            console.print(Panel(resp.answer, title="Answer", border_style="green"))

            # Citations table
            if resp.citations:
                table = Table(title="Citations", show_header=True)
                table.add_column("Type", style="cyan")
                table.add_column("ID")
                table.add_column("Grounded", style="green")

                grounded_ids = {c.id for c in resp.citation_report.grounded_citations}
                for cite in resp.citations:
                    table.add_row(
                        cite.type.upper(),
                        cite.id,
                        "✓" if cite.id in grounded_ids else "✗",
                    )
                console.print(table)

            # Metrics
            console.print(
                f"\n[bold]Confidence:[/bold] {resp.confidence:.1%} | "
                f"[bold]Hallucination score:[/bold] {resp.hallucination_score:.2f} | "
                f"[bold]Docs retrieved:[/bold] {len(resp.retrieved_docs)} | "
                f"[bold]Time:[/bold] {resp.processing_time_ms:.0f}ms"
            )

            if resp.hallucination_flags:
                console.print(f"\n[bold red]Hallucination flags ({len(resp.hallucination_flags)}):[/bold red]")
                for flag in resp.hallucination_flags[:3]:
                    console.print(f"  • [{flag.type.value}] {flag.explanation[:100]}")


@main.group()
def index():
    """Build or manage FAISS knowledge base indices."""


@index.command("build")
@click.option(
    "--source",
    type=click.Choice(["pubmed", "drug_kb", "all"]),
    default="pubmed",
    show_default=True,
)
@click.option(
    "--topics",
    required=True,
    help="Comma-separated list of medical topics to index",
)
def build_index(source: str, topics: str):
    """Build FAISS index from the specified knowledge source."""
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    asyncio.run(_build_index_async(source, topic_list))


async def _build_index_async(source: str, topics: list[str]) -> None:
    from medrag_toolkit.config import Settings
    from medrag_toolkit.core import MedRAG

    config = Settings.from_file()
    async with MedRAG(config) as medrag:
        with console.status(f"[bold green]Building {source} index for {len(topics)} topics..."):
            if source in ("pubmed", "all"):
                await medrag._pubmed_kb.build_index(topics)
            if source in ("drug_kb", "all"):
                await medrag._drug_kb.build_index(topics)

    console.print(f"[bold green]✓[/bold green] Index built: {source} | {len(topics)} topics")


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True)
def serve(host: str, port: int, reload: bool):
    """Start the MedRAG FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        sys.exit(1)

    console.print(f"[bold green]Starting MedRAG API on http://{host}:{port}[/bold green]")
    uvicorn.run(
        "medrag_toolkit.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
def health():
    """Check Ollama connectivity and index availability."""
    asyncio.run(_health_async())


async def _health_async() -> None:
    import httpx

    from medrag_toolkit.config import Settings
    from medrag_toolkit.core import MedRAG

    config = Settings.from_file()
    async with MedRAG(config) as medrag:
        # Ollama check
        ollama_ok = False
        try:
            r = await medrag._http.get(
                f"{config.ollama.base_url}/api/tags", timeout=3.0
            )
            r.raise_for_status()
            ollama_ok = True
        except Exception as exc:
            console.print(f"[red]✗ Ollama:[/red] {exc}")

        if ollama_ok:
            console.print(f"[green]✓ Ollama:[/green] {config.ollama.base_url} ({config.ollama.model})")

        pubmed_ready = medrag._pubmed_kb._indexer.is_ready
        drug_ready = medrag._drug_kb._indexer.is_ready

        icon = lambda ok: "[green]✓[/green]" if ok else "[yellow]○[/yellow]"
        console.print(f"{icon(pubmed_ready)} PubMed index: {'ready' if pubmed_ready else 'not built — run: medrag index build'}")
        console.print(f"{icon(drug_ready)} Drug KB index: {'ready' if drug_ready else 'not built'}")
