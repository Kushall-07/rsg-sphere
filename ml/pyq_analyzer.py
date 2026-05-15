"""Previous-year paper pattern analysis for topic frequency."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
from fuzzywuzzy import fuzz
from rag.loader import extract_text_from_pdf

def analyze_pyq_patterns(pyq_files, topics: List[str]) -> Dict:
    """Analyze year-wise and overall topic occurrence in PYQ papers."""
    topic_year_hits = defaultdict(list)
    year_tables = []
    for idx, pdf in enumerate(pyq_files, start=1):
        pages = extract_text_from_pdf(pdf)
        paper_text = " ".join([p["text"] for p in pages]).lower()
        row = {"year": f"Paper-{idx}"}
        for topic in topics:
            t = topic.lower()
            hit = 1 if ((t in paper_text) or (fuzz.partial_ratio(t, paper_text[:12000]) > 85 if paper_text else False)) else 0
            row[topic] = hit
            if hit:
                topic_year_hits[topic].append(idx)
        year_tables.append(row)
    total = max(1, len(pyq_files))
    frequency = {topic: {"appeared_count": len(topic_year_hits.get(topic, [])), "total_papers": total, "ratio": len(topic_year_hits.get(topic, [])) / total, "years": topic_year_hits.get(topic, [])} for topic in topics}
    return {"frequency": frequency, "year_tables": year_tables}
