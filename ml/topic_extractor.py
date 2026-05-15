"""Topic extraction from note PDFs for exam prediction."""
from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rag.loader import extract_text_from_pdf
TECH_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b")

def extract_topics_from_pdfs(note_files) -> Tuple[List[str], Dict[str, List[int]], str]:
    """Extract candidate study topics and page references from note PDFs."""
    page_map = {}
    corpus_parts = []
    for pdf in note_files:
        pages = extract_text_from_pdf(pdf)
        for p in pages:
            text = p["text"]
            corpus_parts.append(text)
            for phrase in TECH_PATTERN.findall(text):
                phrase_clean = phrase.strip()
                if len(phrase_clean) >= 3:
                    page_map.setdefault(phrase_clean.lower(), set()).add(p["page_number"])
    corpus = "\n".join(corpus_parts)
    if not corpus.strip():
        return [], {}, corpus
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300)
    matrix = vectorizer.fit_transform([corpus])
    scores = np.asarray(matrix.sum(axis=0)).ravel(); terms = np.array(vectorizer.get_feature_names_out())
    ranked_terms = terms[np.argsort(scores)[::-1]]
    frequent_caps = Counter([k for k in page_map.keys() if len(k.split()) <= 4])
    topics = [term for term in ranked_terms[:60] if len(term) >= 4]
    topics.extend([term for term in frequent_caps if term not in topics])
    unique = []
    seen = set()
    for t in topics:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k); unique.append(t.title())
    return unique[:40], {k.title(): sorted(v) for k, v in page_map.items()}, corpus
