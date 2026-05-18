"""PDF loading and chunking utilities for RAG."""
from __future__ import annotations

import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """
    Extract text from every page of a PDF (no page limit).
    Returns page dicts: page_number, text.
    """
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)

    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    pages: List[Dict] = []

    for page_num in range(total_pages):
        try:
            text = (reader.pages[page_num].extract_text() or "").strip()
            if not text:
                continue
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            pages.append({"page_number": page_num + 1, "text": text})
        except Exception as exc:
            print(f"Error on page {page_num + 1}: {exc}")
            continue

    fname = getattr(pdf_file, "name", "document.pdf")
    char_count = sum(len(p["text"]) for p in pages)
    print(
        f"Extracted {total_pages} pages, {len(pages)} with text, "
        f"{char_count} characters from {fname}"
    )
    return pages


def chunk_text(
    pages: List[Dict],
    filename: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split full document text into overlapping chunks (entire PDF covered).
    """
    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    segments: List[tuple[int, str]] = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if text:
            segments.append((int(page["page_number"]), text))

    if not segments:
        return []

    full_text = "\n\n".join(text for _, text in segments)
    page_boundaries: List[tuple[int, int, int]] = []
    offset = 0
    for page_num, text in segments:
        page_boundaries.append((offset, offset + len(text), page_num))
        offset += len(text) + 2

    def page_for_offset(char_idx: int) -> int:
        for start, end, pnum in page_boundaries:
            if start <= char_idx <= end:
                return pnum
        return page_boundaries[-1][2]

    chunks: List[Dict] = []
    search_from = 0
    for chunk_id, part in enumerate(splitter.split_text(full_text)):
        clean = part.strip()
        if not clean:
            continue

        idx = full_text.find(clean, search_from)
        if idx < 0:
            probe = clean[: min(80, len(clean))]
            idx = full_text.find(probe) if probe else -1
        if idx < 0:
            idx = search_from

        chunks.append(
            {
                "text": clean,
                "filename": filename,
                "page_number": page_for_offset(idx),
                "chunk_id": f"{filename}-{chunk_id}",
            }
        )
        search_from = max(0, idx)

    print(f"Created {len(chunks)} chunks from {filename}")
    return chunks
