"""Local Ollama chat for the AI Tutor (llama3.2 by default)."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Literal, Optional

import ollama

SYSTEM_PROMPT = """You are RSGSphere AI Tutor, an expert academic assistant for engineering students.
You help with any subject doubt including Mathematics, Physics, Computer Science,
Electronics, and all engineering subjects.
Always structure your answers as:
1. Simple explanation first
2. Technical details second
3. Real-world example
4. Step-by-step solution if applicable
5. Key points to remember
Be encouraging, patient, and thorough.
Generate practice problems when asked.
When using math, prefer LaTeX-like expressions using $...$ for inline and $$...$$ for display when helpful.
When writing code, use fenced code blocks with a language tag (e.g. ```python)."""


def build_ollama_messages(
    history: List[Dict],
    user_message: str,
    subject_label: str,
    notes_context: str,
) -> List[Dict[str, str]]:
    """Build message list for Ollama chat API."""
    sys = SYSTEM_PROMPT
    if subject_label and subject_label != "General":
        sys += f"\n\nThe student selected subject focus: {subject_label}."
    if notes_context.strip():
        sys += (
            "\n\nThe student has uploaded study materials indexed in ChromaDB. "
            "When relevant, ground explanations in the provided context, explain step by step, "
            "and reference the document when applicable.\n\n---\nDOCUMENT CONTEXT:\n"
            f"{notes_context.strip()[:12000]}\n---"
        )
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    for m in history:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_tutor_reply(messages: List[Dict[str, str]], model: str = "llama3.2") -> Iterable[str]:
    """Yield assistant content tokens from Ollama."""
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


def extract_topic_candidates(text: str, max_items: int = 12) -> List[str]:
    """Lightweight topic phrase extraction for the session panel."""
    found: List[str] = []
    for pat in (
        r"\*\*([^*]{4,64})\*\*",
        r"`([^`]{4,48})`",
        r"(?:Topic|Subject|Chapter)\s*[:\-]\s*([^\n.]{4,80})",
    ):
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = m.group(1).strip()
            if s and s not in found:
                found.append(s)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[a-z]+){0,4})\b", text):
        s = m.group(1).strip()
        if len(s) >= 6 and s not in found and s.isascii():
            found.append(s)
        if len(found) >= max_items * 2:
            break
    return found[:max_items]


def suggest_followups(last_assistant: str, subject: str) -> List[str]:
    """Rule-based follow-up chips from the last reply."""
    tail = (last_assistant or "")[-800:]
    keywords = extract_topic_candidates(tail, 5)
    focus = keywords[0] if keywords else (subject if subject != "General" else "this topic")
    return [
        f"Give a short intuition for {focus} with a diagram in words.",
        f"Show a worked example (step-by-step) for {focus}.",
        f"What are common exam mistakes on {focus}?",
    ]


def exam_hint_prompt(has_pyq_ml: bool, subject: str) -> str:
    if has_pyq_ml:
        return (
            "Based on the ML topic predictions from the uploaded previous-year papers (shown in the right panel), "
            "summarize which themes are most exam-relevant, why, and how to revise them in 1–2 days. "
            "Reference the HIGH / MEDIUM / LOW priorities conceptually (no need to repeat exact scores)."
        )
    sub = subject if subject and subject != "General" else "the selected subject"
    return (
        f"Without any uploaded papers, give general, honest advice on what kinds of topics often show up in university "
        f"exams for {sub}, how to guess emphasis from syllabus wording, and a practical revision checklist. "
        "Avoid claiming specific questions will appear."
    )
