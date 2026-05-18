"""Shared ChromaDB indexing and tutor RAG context helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from rag.embedder import EmbeddingService
from rag.loader import chunk_text, extract_text_from_pdf
from rag.retriever import ChromaRetriever


def get_tutor_rag_context(
    query: str,
    embedder: Optional[EmbeddingService] = None,
    retriever: Optional[ChromaRetriever] = None,
    k: int = 5,
    filenames: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Semantic search over indexed chunks; returns (context text, source line)."""
    embedder = embedder or EmbeddingService()
    retriever = retriever or ChromaRetriever()
    q_emb = embedder.embed_query(query)
    hits = retriever.semantic_search(q_emb, k=k, filenames=filenames)
    if not hits:
        return "", ""

    parts: List[str] = []
    sources: Set[str] = set()
    for chunk in hits:
        excerpt = (chunk.get("text") or "")[:900]
        fname = chunk.get("filename", "document.pdf")
        page = chunk.get("page_number", 0)
        parts.append(f"File `{fname}`, page {page}:\n{excerpt}")
        sources.add(f"{fname} p.{page}")

    context = "\n\n".join(parts)
    source_info = "📄 Sources: " + ", ".join(sorted(sources))
    return context, source_info


def index_pdfs_into_chroma(
    pdf_files,
    *,
    embedder: Optional[EmbeddingService] = None,
    retriever: Optional[ChromaRetriever] = None,
    skip_filenames: Optional[Set[str]] = None,
) -> Tuple[bool, int, List[str], List[Dict]]:
    """
    Index uploaded PDFs into the shared ChromaDB collection.

    Returns (success, chunk_count, newly_indexed_filenames, new_chunks).
    """
    if not pdf_files:
        return False, 0, [], []

    embedder = embedder or EmbeddingService()
    retriever = retriever or ChromaRetriever()
    skip = skip_filenames or set()

    new_chunks: List[Dict] = []
    indexed_names: List[str] = []

    for pdf_file in pdf_files:
        filename = getattr(pdf_file, "name", "document.pdf")
        if filename in skip:
            continue

        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)

        pages = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(pages, filename)
        if not chunks:
            continue

        embeddings = embedder.embed_texts([c["text"] for c in chunks])
        retriever.index_documents(chunks, embeddings)
        new_chunks.extend(chunks)
        indexed_names.append(filename)

    return bool(indexed_names), len(new_chunks), indexed_names, new_chunks
