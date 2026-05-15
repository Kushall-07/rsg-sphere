"""Answer generation via local Ollama model."""
from __future__ import annotations
from typing import Dict, List
import ollama

def build_prompt(query: str, chunks: List[Dict], intent: str) -> str:
    """Create grounded prompt with strict context-only response policy."""
    context = "\n\n".join([f"[{i}] Source: {c.get('filename')} | Page {c.get('page_number')}\n{c.get('text')}" for i, c in enumerate(chunks, start=1)])
    return (
        "You are RSGSphere, an academic assistant. Answer ONLY from the provided context. "
        "If answer not in context, say 'I could not find this in your uploaded documents.' Always cite the source.\n\n"
        f"Intent: {intent}\n\nContext:\n{context}\n\nQuestion: {query}"
    )

def generate_answer_stream(prompt: str, model: str = "llama3.2"):
    """Stream answer token chunks from Ollama chat endpoint."""
    stream = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in stream:
        yield chunk.get("message", {}).get("content", "")
