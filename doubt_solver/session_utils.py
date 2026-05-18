"""Session helpers for stats, topics, chat history, and clipboard."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
        "pending_tutor_message": None,
        "doc_intel_result": None,
        "tutor_doc_intel_result": None,
        "chat_sessions": [],
        "current_session_id": None,
        "student_name": "Student",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _session_created_date(sess: Dict[str, Any]):
    created = sess.get("created_at")
    if isinstance(created, datetime):
        return created.date()
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created).date()
        except ValueError:
            pass
    return datetime.now().date()


def get_current_session() -> Optional[Dict[str, Any]]:
    cid = st.session_state.get("current_session_id")
    for sess in st.session_state.get("chat_sessions", []):
        if sess.get("id") == cid:
            return sess
    return None


def sync_messages_from_current_session() -> None:
    sess = get_current_session()
    if not sess:
        return
    st.session_state["tutor_messages"] = list(sess.get("messages", []))
    subj = sess.get("subject")
    if subj:
        st.session_state["tutor_subject_id"] = subj


def persist_current_session() -> None:
    sess = get_current_session()
    if not sess:
        return
    msgs = list(st.session_state.get("tutor_messages", []))
    sess["messages"] = msgs
    sess["subject"] = st.session_state.get("tutor_subject_id", "general")
    title = "New chat"
    for m in msgs:
        if m.get("role") == "user":
            raw = (m.get("content") or "").strip().replace("\n", " ")
            if raw:
                title = raw[:30] + ("…" if len(raw) > 30 else "")
            break
    sess["title"] = title
    created = sess.get("created_at")
    if not isinstance(created, datetime):
        sess["created_at"] = datetime.now()


def create_new_chat_session() -> str:
    """Start a fresh chat session and make it active."""
    persist_current_session()
    sid = str(uuid.uuid4())
    entry: Dict[str, Any] = {
        "id": sid,
        "title": "New chat",
        "messages": [],
        "created_at": datetime.now(),
        "subject": st.session_state.get("tutor_subject_id", "general"),
    }
    sessions: List[Dict[str, Any]] = st.session_state.setdefault("chat_sessions", [])
    sessions.insert(0, entry)
    st.session_state["current_session_id"] = sid
    st.session_state["tutor_messages"] = []
    st.session_state["tutor_topic_status"] = {}
    st.session_state["tutor_session_start"] = None
    st.session_state["tutor_input_nonce"] = int(st.session_state.get("tutor_input_nonce", 0)) + 1
    return sid


def select_chat_session(session_id: str) -> None:
    """Switch active chat and load its messages."""
    persist_current_session()
    st.session_state["current_session_id"] = session_id
    sync_messages_from_current_session()


def group_sessions_by_date(
    sessions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket chat sessions for sidebar history groups."""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    week_floor = today - timedelta(days=7)
    groups: Dict[str, List[Dict[str, Any]]] = {
        "Today": [],
        "Yesterday": [],
        "7 Days Ago": [],
        "Older": [],
    }
    for sess in sessions:
        d = _session_created_date(sess)
        if d == today:
            groups["Today"].append(sess)
        elif d == yesterday:
            groups["Yesterday"].append(sess)
        elif d > week_floor:
            groups["7 Days Ago"].append(sess)
        else:
            groups["Older"].append(sess)
    return groups


def init_chat_sessions() -> None:
    """Initialize multi-chat history and migrate legacy single-thread state."""
    if "chat_sessions" not in st.session_state:
        st.session_state["chat_sessions"] = []
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = None

    sessions: List[Dict[str, Any]] = st.session_state["chat_sessions"]
    legacy_msgs = st.session_state.get("tutor_messages") or []

    if not sessions and legacy_msgs:
        sid = str(uuid.uuid4())
        title = "New chat"
        for m in legacy_msgs:
            if m.get("role") == "user":
                raw = (m.get("content") or "").strip().replace("\n", " ")
                if raw:
                    title = raw[:30] + ("…" if len(raw) > 30 else "")
                break
        sessions.append(
            {
                "id": sid,
                "title": title,
                "messages": list(legacy_msgs),
                "created_at": datetime.now(),
                "subject": st.session_state.get("tutor_subject_id", "general"),
            }
        )
        st.session_state["current_session_id"] = sid
    elif not sessions:
        create_new_chat_session()
    elif not st.session_state.get("current_session_id"):
        st.session_state["current_session_id"] = sessions[0]["id"]

    sync_messages_from_current_session()


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
