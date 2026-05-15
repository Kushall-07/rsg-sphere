"""PDF loading and chunking utilities for RAG."""
from __future__ import annotations
from typing import Dict, List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """Extract text page by page from a PDF-like object."""
    reader = PdfReader(pdf_file)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": idx, "text": text})
    return pages

def chunk_text(pages: List[Dict], filename: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split PDF page text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    chunk_id = 0
    for page in pages:
        for part in splitter.split_text(page["text"]):
            clean = part.strip()
            if not clean:
                continue
            chunks.append({"text": clean, "filename": filename, "page_number": page["page_number"], "chunk_id": f"{filename}-{chunk_id}"})
            chunk_id += 1
    return chunks
