"""
PYQ analyzer: PDF text extraction, question segmentation, NLP topic signals,
Random Forest regression for ranked topic predictions.
"""
from __future__ import annotations

import io
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from rag.loader import extract_text_from_pdf
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

MARKS_PATTERN = re.compile(
    r"(?:\[|\()?\s*(\d+)\s*(?:marks?|m)\s*(?:\]|\))?",
    re.IGNORECASE,
)
QNUMBER_PATTERN = re.compile(
    r"^\s*(?:q(?:uestion)?\s*)?[.\-]?\s*(\d+)\s*[.):\-]\s*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class QuestionRecord:
    text: str
    year_idx: int
    marks_hint: float
    position: float  # 0 = start of doc, 1 = end


def _read_pdf(upload: Any) -> List[Dict]:
    if hasattr(upload, "getbuffer"):
        bio = io.BytesIO(bytes(upload.getbuffer()))
        bio.seek(0)
        return extract_text_from_pdf(bio)
    if hasattr(upload, "read"):
        data = upload.read()
        bio = io.BytesIO(data if isinstance(data, (bytes, bytearray)) else bytes(str(data).encode()))
        bio.seek(0)
        if hasattr(upload, "seek"):
            upload.seek(0)
        return extract_text_from_pdf(bio)
    return extract_text_from_pdf(upload)


def _full_document_text(pages: List[Dict]) -> str:
    chunks = []
    for p in pages:
        chunks.append((p["text"] or "").strip())
    return "\n\n".join(c for c in chunks if c)


def _guess_marks_near(text_segment: str) -> float:
    best = 0.0
    for m in MARKS_PATTERN.finditer(text_segment):
        try:
            v = float(m.group(1))
            best = max(best, min(v / 30.0, 1.0))
        except (TypeError, ValueError):
            continue
    return best


def extract_questions_from_text(full_text: str, year_idx: int) -> List[QuestionRecord]:
    lines = full_text.replace("\x00", "").split("\n")
    n_chars = len(full_text)
    cursor = 0
    idx_to_pos = []

    rough_blocks: List[str] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            block = "\n".join(buf).strip()
            if len(block) > 5:
                rough_blocks.append(block)
            buf = []

    for ln in lines:
        stripped = ln.strip()
        cursor = full_text.find(ln + "\n", cursor)
        pos = cursor / max(1, n_chars)
        idx_to_pos.append(pos)
        is_q_line = stripped.endswith("?") or (
            stripped and stripped[0].isdigit() and QNUMBER_PATTERN.search(stripped[:12])
        )
        if stripped and (
            stripped.endswith("?") or bool(QNUMBER_PATTERN.match(stripped)) or is_q_line
        ):
            flush_buf()
            buf.append(ln.strip())
        elif buf:
            if stripped.startswith("Ans") or stripped.upper().startswith("OR"):
                flush_buf()
            else:
                buf.append(ln.strip())
        cursor += len(ln) + 1

    flush_buf()

    sentence_qs = []
    blob = full_text.replace("\n", " ")
    for chunk in re.split(r"(?<=[.?])\s+", blob):
        c = chunk.strip()
        if c.endswith("?") and len(c) > 20:
            sentence_qs.append(c)

    all_blocks = list(dict.fromkeys(rough_blocks + sentence_qs))
    records: List[QuestionRecord] = []
    doc_len = max(1, len(full_text))
    seen = set()
    for qb in all_blocks[:400]:
        if qb in seen:
            continue
        seen.add(qb)
        sub = qb[:280]
        marks = _guess_marks_near(qb[-120:])
        pos = full_text.find(qb[: min(48, len(qb))])
        frac = float(pos) / doc_len if pos >= 0 else 0.5
        complexity = min(1.0, len(qb.split()) / 120.0)
        if qb.endswith("?") or QNUMBER_PATTERN.search(qb[:64]) or len(qb) > 35:
            records.append(QuestionRecord(qb[:2000], year_idx, max(marks, complexity * 0.2), frac))
    return records


def noun_phrases_simple(texts: List[str]) -> List[str]:
    phrases: List[str] = []
    pat = re.compile(r"\b([A-Z][a-z]+(?:\s+[a-z]+){1,6})\b")
    for t in texts:
        for m in pat.finditer(t):
            w = m.group(1).strip()
            if len(w) >= 8 and len(w) < 72:
                phrases.append(w)
    return phrases


def analyze_pyq_ml_pipeline(pyq_uploads: List[Any]) -> Dict[str, Any]:
    """Run PYQ PDF analysis and return ranked priorities + telemetry."""
    if not pyq_uploads:
        return {"error": "No PDFs uploaded"}

    all_questions: List[QuestionRecord] = []
    texts_for_phrases: List[str] = []

    for idx, up in enumerate(pyq_uploads):
        pages = _read_pdf(up)
        full_txt = _full_document_text(pages)
        qs = extract_questions_from_text(full_txt, year_idx=idx)
        all_questions.extend(qs)
        texts_for_phrases.extend(full_txt[s : s + 400] for s in range(0, len(full_txt), 500))

    n_years = len(pyq_uploads)

    corpus = [q.text for q in all_questions if len(q.text) > 8]
    extra_terms = noun_phrases_simple(texts_for_phrases[:200])

    if len(corpus) < 3 and extra_terms:
        corpus.extend(extra_terms)

    unique_topics_fallback = []
    for ph in noun_phrases_simple([q.text for q in all_questions]):
        unique_topics_fallback.append(ph)

    vec = TfidfVectorizer(
        max_features=180,
        ngram_range=(1, 3),
        min_df=max(1, min(3, len(corpus) // 3)),
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )

    topic_labels: List[str] = []

    try:
        X_tfidf = vec.fit_transform(corpus)
        terms_arr = vec.get_feature_names_out()
        col_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
        top_idx = np.argsort(col_sums)[::-1][:80]
        topic_labels.extend(str(terms_arr[i]) for i in top_idx if col_sums[i] > 0)

        noun_set = sorted(set(extra_terms))[:120]
        lowered = {t.lower() for t in topic_labels}
        for n in noun_set:
            if n.lower() not in lowered:
                topic_labels.append(n)
                lowered.add(n.lower())
    except Exception:
        vec = None
        topic_labels = sorted(set(unique_topics_fallback))

    topic_labels = [t.strip() for t in topic_labels if t and len(t) > 2][:150]
    topic_labels = list(dict.fromkeys(topic_labels))

    if not topic_labels and corpus:
        words = sorted({w.strip(".,):;\"'") for c in corpus for w in re.split(r"\s+", c) if len(w) > 8})
        topic_labels = words[:60]

    if not topic_labels:
        return {"error": "Could not extract topics — try clearer PDF scans."}

    year_to_topic_hits = defaultdict(lambda: defaultdict(int))
    topic_marks = defaultdict(list)
    topic_pos = defaultdict(list)
    topic_complexity = defaultdict(list)

    for q in all_questions:
        low = q.text.lower()
        for t in topic_labels:
            tl = t.lower()
            parts = tl.split()
            hits = (
                tl in low
                or all(p in low for p in parts if len(p) > 3)
                or (sum(1 for p in parts if len(p) > 3 and p in low) >= max(1, len(parts) // 2))
            )
            if hits:
                year_to_topic_hits[q.year_idx][t] += 1
                topic_marks[t].append(q.marks_hint)
                topic_pos[t].append(q.position)
                topic_complexity[t].append(min(1.0, len(q.text.split()) / 150.0))

    rows = []
    for t in topic_labels:
        year_hits = [1 if year_to_topic_hits[y][t] > 0 else 0 for y in range(n_years)]
        years_ct = sum(year_hits)
        freq_cross_years = years_ct / max(1, n_years)
        occ = sum(year_to_topic_hits[y][t] for y in range(n_years))
        marks_mu = float(np.mean(topic_marks[t])) if topic_marks[t] else 0.35
        pos_mu = float(np.mean(topic_pos[t])) if topic_pos[t] else 0.5
        comp_mu = float(np.mean(topic_complexity[t])) if topic_complexity[t] else 0.4

        base_score = (
            45.0 * freq_cross_years
            + 30.0 * min(occ / max(1, n_years * 2), 1.5)
            + 25.0 * marks_mu
            + (1.0 - pos_mu) * 5
        )

        rows.append(
            {
                "topic": t,
                "freq_cross_years": freq_cross_years,
                "occurrences": occ,
                "marks_proxy": marks_mu,
                "position_mu": pos_mu,
                "complexity_mu": comp_mu,
                "years_count": years_ct,
                "_base_score": base_score,
            }
        )

    if not rows:
        return {"error": "Insufficient topic-topic overlap"}

    feats = []
    ys = []
    for r in rows:
        feats.append(
            [
                r["freq_cross_years"],
                min(r["occurrences"] / max(10, len(all_questions) / 50), 2.5),
                r["marks_proxy"],
                r["complexity_mu"],
                r["position_mu"],
            ]
        )
        ys.append(r["_base_score"])

    X_arr = np.array(feats, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)

    if len(rows) >= 8:
        model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=12)
        model.fit(X_arr, y_arr)
        predicted = model.predict(X_arr).astype(float)
        importances = model.feature_importances_.tolist()
    else:
        model = None
        predicted = X_arr[:, 0] * 55 + X_arr[:, 1] * 15 + X_arr[:, 2] * 30
        importances = [0.35, 0.22, 0.18, 0.13, 0.12]

    for i, r in enumerate(rows):
        r["score"] = float(predicted[i])
        r["priority"] = "LOW"
        yrs = r["years_count"]
        r["years_label"] = f"{yrs}/{max(1, n_years)} years"

    order = sorted(range(len(rows)), key=lambda i: rows[i]["score"], reverse=True)

    thresh_hi = sorted(rows[k]["score"] for k in order)[max(0, min(len(rows) // 4, len(rows) - 1))]
    thresh_med = sorted(rows[k]["score"] for k in order)[max(0, min(len(rows) // 2, len(rows) - 1))]

    for r in rows:
        if r["score"] >= thresh_hi:
            r["priority"] = "HIGH"
        elif r["score"] >= thresh_med:
            r["priority"] = "MEDIUM"
        else:
            r["priority"] = "LOW"

    ranked = sorted(rows, key=lambda rr: rr["score"], reverse=True)

    topic_count = sum(1 for r in ranked if r["occurrences"] > 0)
    topic_count = max(topic_count, len([r for r in ranked[:50] if r["score"] > 0]))

    return {
        "ranked": ranked[:40],
        "n_questions_detected": len(all_questions),
        "n_years": n_years,
        "n_unique_topics_surfaces": len(topic_labels),
        "feature_names": ["freq_cross_years", "occ_density", "marks_proxy", "complexity", "position"],
        "feature_importances": importances,
        "model": model,
        "years_analyzed_note": n_years,
        "topics_found_note": topic_count,
    }


def format_priority_blocks(result: Dict[str, Any]) -> Dict[str, List[Dict]]:
    buckets = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in result.get("ranked", []):
        buckets.setdefault(r["priority"], []).append(r)
    return buckets
