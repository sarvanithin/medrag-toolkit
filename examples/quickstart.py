"""
medrag-toolkit quickstart example.

Prerequisites:
  1. Install: pip install -e .
  2. Start Ollama: ollama serve && ollama pull llama3.2
  3. Build index: medrag index build --source pubmed --topics "myocardial infarction"
  4. Run this script: python examples/quickstart.py
"""
from __future__ import annotations

import asyncio

from medrag_toolkit import MedRAG, Settings


async def main():
    config = Settings.from_file()

    async with MedRAG(config) as medrag:
        print("Building index for demo...")
        await medrag._pubmed_kb.build_index(["myocardial infarction", "aspirin cardiology"])
        print("Index built!\n")

        question = "What is the recommended aspirin dose for acute MI?"
        print(f"Question: {question}\n")

        response = await medrag.query(question)

        print(f"Answer:\n{response.answer}\n")
        print(f"Citations: {len(response.citations)}")
        for cite in response.citations:
            grounded = cite in response.citation_report.grounded_citations
            print(f"  [{cite.type.upper()}:{cite.id}] {'✓ grounded' if grounded else '✗ not grounded'}")

        print(f"\nConfidence: {response.confidence:.1%}")
        print(f"Hallucination score: {response.hallucination_score:.2f}")
        print(f"Processing time: {response.processing_time_ms:.0f}ms")

        if response.hallucination_flags:
            print("\nHallucination flags:")
            for flag in response.hallucination_flags:
                print(f"  [{flag.type.value}] {flag.explanation[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
