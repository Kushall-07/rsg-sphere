"""Streamlit Tab 2: AI Tutor with fixed sidebar + scrollable chat pane."""
from __future__ import annotations

import html as html_module
import io
import re
from datetime import datetime
from typing import Callable, List, Optional

import streamlit as st

from doubt_solver.ml.pyq_analyzer import analyze_pyq_ml_pipeline
from doubt_solver.ollama_tutor import (
    build_ollama_messages,
    exam_hint_prompt,
    stream_tutor_reply,
)
from doubt_solver.session_utils import (
    create_new_chat_session,
    group_sessions_by_date,
    init_chat_sessions,
    init_tutor_session_state,
    merge_topics_from_message,
    persist_current_session,
    sanitize_copy_id,
    select_chat_session,
    touch_session_start,
)
from utils.voice import get_voice_input

def get_quick_topics_from_notes(notes_context):
    """Extract key topics from uploaded notes"""
    if not notes_context or len(notes_context) < 100:
        return []
    
    import re
    # Simple topic extraction:
    # Find capitalized technical words/phrases
    words = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', 
                       notes_context[:3000])
    # Count frequency
    from collections import Counter
    freq = Counter(words)
    # Filter out common words
    stopwords = {'The', 'This', 'That', 'These', 
                 'When', 'Where', 'What', 'How',
                 'With', 'From', 'Into', 'They',
                 'There', 'Then', 'Here'}
    topics = [w for w, c in freq.most_common(10) 
              if w not in stopwords and len(w) > 3]
    return topics[:4]  # Return max 4 topics

def extract_topics_from_pyqs(pdf_files):
    """
    Extract important topics from PYQ PDFs
    using TF-IDF and frequency analysis
    """
    import PyPDF2
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import Counter
    
    all_text = ""
    for pdf_file in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                all_text += page.extract_text() + " "
        except:
            continue
    
    if not all_text.strip():
        return []
    
    # Extract noun phrases (capitalized terms)
    words = re.findall(
        r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',
        all_text)
    
    # Common words to ignore
    ignore = {'What', 'Explain', 'Define', 
              'Write', 'Describe', 'Compare',
              'List', 'Discuss', 'With', 'The',
              'This', 'That', 'How', 'When',
              'Short', 'Long', 'Answer', 'Note',
              'Question', 'Part', 'Section',
              'Module', 'Unit', 'Mark', 'Marks'}
    
    # Count and filter
    freq = Counter(words)
    topics = [w for w, c in freq.most_common(20)
              if w not in ignore 
              and len(w) > 3
              and c >= 1]
    
    return topics[:8]

def predict_exam_questions(pdf_files):
    """
    Extract actual questions from PYQ PDFs
    by finding question patterns
    """
    import PyPDF2
    import re
    
    all_questions = []
    
    for pdf_file in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if not text:
                    continue
                    
                # Find questions by patterns:
                # Lines starting with number/letter
                # Lines containing question words
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Pattern 1: numbered questions
                    if re.match(
                        r'^[0-9]+[\.\)]\s+[A-Z]',
                        line) and len(line) > 20:
                        # Clean the line
                        q = re.sub(
                            r'^[0-9]+[\.\)]\s+',
                            '', line)
                        if len(q) > 15:
                            all_questions.append(q)
                    # Pattern 2: starts with 
                    # question words
                    elif re.match(
                        r'^(What|Explain|Define|'
                        r'Describe|Compare|List|'
                        r'Discuss|Write|How|Why)',
                        line) and len(line) > 20:
                        all_questions.append(line)
        except:
            continue
    
    # Remove duplicates, keep best ones
    seen = set()
    unique_questions = []
    for q in all_questions:
        q_clean = q[:50].lower()
        if q_clean not in seen:
            seen.add(q_clean)
            unique_questions.append(q)
    
    return unique_questions[:6]


def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 1rem;
            max-width: 100%;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #1E293B;
            border-color: #334155 !important;
        }
        .tutor-history-label {
            color: #64748B;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 12px 0 6px 2px;
        }
        [data-testid="stTextInput"] input {
            background: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 24px !important;
            color: #F1F5F9 !important;
            padding: 12px 16px !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
        }
        .tutor-footnote {
            text-align: center;
            font-size: 12px;
            color: #64748B;
            margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _time_greeting() -> str:
    h = datetime.now().hour
    if h < 12:
        return "Good Morning"
    if h < 17:
        return "Good Afternoon"
    return "Good Evening"


def _format_message_html(content: str) -> str:
    safe = html_module.escape(content or "")
    safe = safe.replace("\n", "<br>")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(
        r"`([^`]+)`",
        r"<code style='background:#334155;padding:2px 6px;border-radius:4px;'>\1</code>",
        safe,
    )
    return safe


def _render_welcome(student: str) -> None:
    greet = _time_greeting()
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            f"""
            <p style="text-align:center;font-size:48px;margin:24px 0 16px;">📚</p>
            <h2 style="text-align:center;color:#F1F5F9;margin:0 0 8px;font-size:1.5rem;font-weight:600;">
                ✨ {greet}, {html_module.escape(student)}!
            </h2>
            <p style="text-align:center;color:#94A3B8;margin:0;font-size:1rem;">
                Hello! Ask me anything about your subjects.
            </p>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_messages(messages: List[dict]) -> None:
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        stamp = msg.get("time", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
                if stamp:
                    st.caption(stamp)
        else:
            with st.chat_message("assistant", avatar="📚"):
                st.markdown(content)
                if stamp:
                    st.caption(stamp)


def _render_chat_container(messages: List[dict], student: str) -> None:
    with st.container(height=520, border=True):
        if not messages:
            _render_welcome(student)
        else:
            _render_chat_messages(messages)


def _render_sidebar() -> None:
    with st.container(border=True):
        st.markdown("### 📚 RSGSphere")
        if st.button("+ New Chat", key="tutor_new_chat_btn", use_container_width=True, type="primary"):
            create_new_chat_session()
            st.rerun()

        for i, session in enumerate(st.session_state.tutor_sessions):
            col1, col2 = st.columns([5, 1])
            with col1:
                title_str = session.get("title") or "New chat"
                if st.button(
                    title_str[:25] + ("..." if len(title_str) > 25 else ""),
                    key=f"hist_{i}",
                    use_container_width=True
                ):
                    # Load this chat
                    st.session_state.current_session_id = session["id"]
                    st.session_state.tutor_messages = session.get("messages", [])
                    st.rerun()
            with col2:
                if st.button(
                    "🗑️",
                    key=f"del_{i}",
                    help="Delete chat"
                ):
                    st.session_state.tutor_sessions.pop(i)
                    if st.session_state.current_session_id == session["id"]:
                        st.session_state.tutor_messages = []
                        st.session_state.current_session_id = None
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🎯 Exam Predictor")

        pyq_files = st.file_uploader(
            "Upload PYQ Papers",
            type=["pdf"],
            accept_multiple_files=True,
            key="pyq_uploader_sidebar",
            label_visibility="collapsed"
        )

        # BUTTON 1: Analyze Topics
        if st.button("📚 Analyze Topics",
                     key="analyze_topics_btn",
                     use_container_width=True):
            if pyq_files:
                with st.spinner("Extracting topics..."):
                    topics = extract_topics_from_pyqs(pyq_files)
                    st.session_state.pyq_topics = topics
                    st.session_state.pyq_questions = []
                st.rerun()
            else:
                st.warning("Upload PYQ PDFs first")

        # Show topics if available
        if st.session_state.get("pyq_topics"):
            st.markdown("**📌 Important Topics:**")
            for i, topic in enumerate(st.session_state.pyq_topics[:8]):
                if st.button(
                    f"• {topic}",
                    key=f"topic_{i}",
                    use_container_width=True
                ):
                    # Send to Smart Chat (Tab 1) 
                    # by storing in shared session state
                    st.session_state.smart_chat_input = \
                        f"Explain {topic} in detail with examples and key points"
                    st.session_state.switch_to_tab = 0
                    st.rerun()

        st.markdown("")

        # BUTTON 2: Predict Questions
        if st.button("🎯 Predict Questions",
                     key="predict_questions_btn",
                     use_container_width=True):
            if pyq_files:
                with st.spinner("Predicting questions..."):
                    questions = predict_exam_questions(pyq_files)
                    st.session_state.pyq_questions = questions
                    st.session_state.pyq_topics = st.session_state.get("pyq_topics", [])
                st.rerun()
            else:
                st.warning("Upload PYQ PDFs first")

        # Show predicted questions if available
        if st.session_state.get("pyq_questions"):
            st.markdown("**❓ Predicted Questions:**")
            
            # Show in expandable container
            with st.expander("View Questions", expanded=True):
                for i, question in enumerate(st.session_state.pyq_questions[:6]):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**Q{i+1}.** {question}")
                    with col2:
                        if st.button("→", key=f"q_send_{i}", help="Send to chat"):
                            # Send question to AI Tutor chat
                            st.session_state.tutor_input_temp = \
                                f"Answer this exam question in detail: {question}"
                            st.rerun()


def _process_outgoing(
    outgoing: str,
    notes_ctx_fn: Callable[[str], str],
    status_slot,
) -> None:
    touch_session_start()
    stamp = datetime.now().strftime("%H:%M:%S")
    hist = list(st.session_state["tutor_messages"])
    ut = outgoing.strip()
    st.session_state["tutor_messages"].append({"role": "user", "content": ut, "time": stamp})
    msgs = build_ollama_messages(hist, ut, "General", notes_ctx_fn(ut))
    accumulated = ""
    try:
        for token in stream_tutor_reply(msgs):
            accumulated += token
            status_slot.markdown(f"**RSGSphere is typing…**\n\n{accumulated}▎")
        status_slot.empty()
    except Exception as exc:
        accumulated = (
            "**Could not reach Ollama.** Start `ollama serve` and pull the model "
            f"(`ollama pull llama3.2`).\n\n`{exc}`"
        )
        status_slot.empty()

    astamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["tutor_messages"].append(
        {"role": "assistant", "content": accumulated, "time": astamp, "feedback": None}
    )
    merge_topics_from_message("assistant", accumulated, st.session_state["tutor_topic_status"])
    persist_current_session()


def render_ai_tutor_tab(get_notes_context: Optional[Callable[[str], str]] = None):
    """Fixed sidebar + scrollable chat pane + pinned input bar."""
    init_tutor_session_state()
    init_chat_sessions()
    _inject_layout_css()

    if "tutor_sessions" not in st.session_state:
        st.session_state["tutor_sessions"] = []
    if "tutor_messages" not in st.session_state:
        st.session_state["tutor_messages"] = []
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = None
    if "last_input" not in st.session_state:
        st.session_state.last_input = ""
    if "tutor_input_temp" not in st.session_state:
        st.session_state.tutor_input_temp = ""
    if "pyq_topics" not in st.session_state:
        st.session_state.pyq_topics = []
    if "pyq_questions" not in st.session_state:
        st.session_state.pyq_questions = []
    if "smart_chat_input" not in st.session_state:
        st.session_state.smart_chat_input = ""
    st.session_state["tutor_sessions"] = st.session_state.get("chat_sessions", [])

    def default_notes(_q: str) -> str:
        return ""

    notes_ctx_fn = get_notes_context or default_notes
    pending_auto = str(st.session_state.pop("tutor_auto_send", "") or "").strip()
    student = str(st.session_state.get("student_name", "Student"))
    messages = list(st.session_state.get("tutor_messages", []))
    send = False

    side_col, main_col = st.columns([1, 3], gap="medium")

    with side_col:
        _render_sidebar()

    with main_col:
        _render_chat_container(messages, student)
        stream_status = st.empty()

        if not messages:
            notes_context = notes_ctx_fn("summary of topics") if notes_ctx_fn else ""
            quick_topics = get_quick_topics_from_notes(notes_context)

            if quick_topics:
                st.markdown("**📚 From your uploaded notes:**")
                cols = st.columns(len(quick_topics))
                for i, topic in enumerate(quick_topics):
                    with cols[i]:
                        if st.button(topic, key=f"quick_topic_{i}", use_container_width=True):
                            st.session_state["tutor_pending_send"] = f"Explain {topic} in detail with examples"
                            st.rerun()

        in_left, in_mid, in_right = st.columns([1, 10, 1])
        with in_left:
            if st.button("🎤", key="tutor_voice_btn", help="Voice input"):
                text, err = get_voice_input()
                if text:
                    st.session_state["tutor_pending_send"] = text
                    st.rerun()
                elif err:
                    st.toast(err)

        with in_mid:
            user_input = st.text_input(
                "Ask anything",
                placeholder="Ask anything...",
                key="tutor_input_temp",
                label_visibility="collapsed",
            )

        with in_right:
            send = st.button("➤", key="tutor_send_btn", help="Send", type="primary")

        st.markdown(
            '<p class="tutor-footnote">RSGSphere uses llama3.2 locally · Inter</p>',
            unsafe_allow_html=True,
        )

    dispatch = str(st.session_state.pop("tutor_pending_send", "") or "").strip()
    manual = pending_auto if pending_auto else user_input.strip()
    
    outgoing = ""
    if dispatch:
        outgoing = dispatch
    elif send or (user_input and user_input.strip() != "" and st.session_state.get("last_input") != user_input) or pending_auto:
        outgoing = manual

    if outgoing:
        with st.spinner("Thinking…"):
            _process_outgoing(outgoing, notes_ctx_fn, stream_status)
        st.session_state.last_input = user_input
        st.session_state["tutor_sessions"] = st.session_state.get("chat_sessions", [])
        st.rerun()
