"""Session helpers for stats, topics, and clipboard."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from doubt_solver.ollama_tutor import extract_topic_candidates


def init_tutor_session_state() -> None:
    """Ensure keys used by the AI Tutor tab exist."""
    defaults: Dict[str, Any] = {
        "tutor_messages": [],
        "tutor_subject_id": "general",
        "tutor_session_start": None,
        "tutor_pyq_bytes": [],
        "tutor_pyq_result": None,
        "tutor_topic_status": {},
        "tutor_input_nonce": 0,
        "tutor_auto_send": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def touch_session_start() -> None:
    if st.session_state["tutor_session_start"] is None:
        st.session_state["tutor_session_start"] = datetime.now()


def count_user_questions() -> int:
    return sum(1 for m in st.session_state["tutor_messages"] if m.get("role") == "user")


def session_duration_mins() -> float:
    start = st.session_state.get("tutor_session_start")
    if not start:
        return 0.0
    return max(0.0, (datetime.now() - start).total_seconds() / 60.0)


def merge_topics_from_message(
    role: str,
    content: str,
    topic_status: Dict[str, str],
) -> None:
    if role != "assistant":
        return
    for t in extract_topic_candidates(content, 15):
        key = t.lower()
        if key not in topic_status:
            topic_status[key] = t


def topic_list_for_panel(topic_status: Dict[str, str]) -> List[str]:
    return list(topic_status.values())[:20]


def try_copy(text: str) -> bool:
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False


def sanitize_copy_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s)[:48]
