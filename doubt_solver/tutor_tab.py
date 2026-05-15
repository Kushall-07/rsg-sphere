"""Streamlit Tab 2: AI Tutor (Ollama) + Smart study panel + optional PYQ ML."""
from __future__ import annotations

import hashlib
import html as html_module
import io
from datetime import datetime
from typing import Callable, List, Optional

import streamlit as st

from doubt_solver.ml.pyq_analyzer import analyze_pyq_ml_pipeline
from doubt_solver.ollama_tutor import (
    build_ollama_messages,
    exam_hint_prompt,
    stream_tutor_reply,
    suggest_followups,
)
from doubt_solver.session_utils import (
    count_user_questions,
    init_tutor_session_state,
    merge_topics_from_message,
    session_duration_mins,
    sanitize_copy_id,
    topic_list_for_panel,
    touch_session_start,
    try_copy,
)

SUBJECT_OPTIONS: List[tuple[str, str, str]] = [
    ("General", "general", "🎓 Any subject"),
    ("Mathematics", "maths", "📐 Maths"),
    ("Physics", "physics", "⚛️ Physics"),
    ("Computer Science", "cs", "💻 CS"),
    ("Electronics", "electronics", "⚡ Electronics"),
    ("Chemistry", "chem", "🔬 Chemistry"),
    ("Machine Learning / AI", "ml", "📊 ML/AI"),
    ("DBMS", "dbms", "🗄️ DBMS"),
    ("Networking", "net", "🌐 Networks"),
]


def _chip_from_state() -> tuple[str, str]:
    sid = str(st.session_state.get("tutor_subject_id", "general"))
    for canon, cid, lbl in SUBJECT_OPTIONS:
        if cid == sid:
            return canon, lbl
    return "General", "🎓 Any subject"


def _subject_label_plain() -> str:
    canon, _ = _chip_from_state()
    return canon


def render_ai_tutor_tab(get_notes_context: Optional[Callable[[str], str]] = None):
    """
    Replace legacy Exam Predictor tab with conversational AI tutor + smart panel.

    get_notes_context: optional callable(query)->str injecting Tab 1 indexed notes excerpts.
    """
    init_tutor_session_state()

    def default_notes(_q: str) -> str:
        return ""

    notes_ctx_fn = get_notes_context or default_notes

    pending_auto = str(st.session_state.pop("tutor_auto_send", "") or "").strip()

    st.subheader("🤖 AI Tutor")

    canon_focus, chip_lbl_top = _chip_from_state()
    plain_subject = _subject_label_plain()

    left, right = st.columns([7, 3], gap="medium")

    with left:
        st.markdown(
            """
            <div style="padding:12px;background:rgba(255,255,255,0.06);border-radius:12px;margin-bottom:8px;">
            Full-height chat powered by local <b>Ollama (llama3.2)</b>. No PDF required here — optional PYQs power the predictor in the panel.
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_box = st.container(height=460, border=True)
        with chat_box:
            for i, msg in enumerate(st.session_state["tutor_messages"]):
                if msg["role"] == "user":
                    with st.chat_message("user", avatar=None):
                        st.markdown(html_module.escape(msg["content"]))
                        if msg.get("time"):
                            st.caption(msg["time"])
                else:
                    content = msg.get("content", "") or ""
                    with st.chat_message("assistant", avatar="🪐"):
                        st.markdown("**RSGSphere**")
                        st.markdown(content)
                        st.caption(msg.get("time", ""))
                        b1, b2, b3 = st.columns([1.2, 1, 6])
                        with b1:
                            if st.button("📋", key=f"cp_{i}", help="Copy response"):
                                if try_copy(content):
                                    st.toast("Copied to clipboard.")
                                else:
                                    st.session_state[f"_copy_fallback_{i}"] = True
                        if st.session_state.get(f"_copy_fallback_{i}"):
                            st.text_area(
                                "Copy manually",
                                content,
                                height=min(280, max(100, len(content) // 3)),
                                key=f"cf_{i}",
                            )
                        with b2:
                            if st.button("👍", key=f"up_{i}"):
                                msg["feedback"] = "up"
                        with b3:
                            if st.button("👎", key=f"dn_{i}"):
                                msg["feedback"] = "down"

            if not st.session_state["tutor_messages"]:
                st.info("Ask any subject doubt — structuring, derivations, code, intuition, or exams.")

        nonce = int(st.session_state["tutor_input_nonce"])
        ta_key = f"tutor_txt_{nonce}"
        if ta_key not in st.session_state:
            st.session_state[ta_key] = st.session_state.pop("tutor_prefill_buffer", "")

        user_in = st.text_area(
            "Message",
            height=140,
            key=ta_key,
            label_visibility="collapsed",
            placeholder="Type your question (multiline). Replies support Markdown, code fences, and $math$.",
        )

        send = st.button("Send", type="primary", key="send_btn_tutor")

        st.caption("Subject focus")
        nrow = min(6, len(SUBJECT_OPTIONS))
        rcols = st.columns(nrow)
        first = SUBJECT_OPTIONS[:nrow]
        rest = SUBJECT_OPTIONS[nrow:]
        for j, (_, cid, lbl) in enumerate(first):
            with rcols[j]:
                if st.button(lbl, key=f"sj_{cid}"):
                    st.session_state["tutor_subject_id"] = cid
                    st.rerun()
        if rest:
            r2 = st.columns(len(rest))
            for j, (_, cid, lbl) in enumerate(rest):
                with r2[j]:
                    if st.button(lbl, key=f"sj2_{cid}"):
                        st.session_state["tutor_subject_id"] = cid
                        st.rerun()

        if canon_focus != "General":
            st.caption(f"Focus: **{chip_lbl_top}** — {canon_focus}")

        st.markdown("**Quick actions**")
        aq1, aq2 = st.columns(2)
        with aq1:
            if st.button("💡 Explain this topic", key="qa_expl"):
                st.session_state["tutor_quick_dispatch"] = (
                    "Explain the topic we are discussing in simple terms first, then go deeper technically, "
                    "with a real-world example and step-by-step notes."
                )
        with aq2:
            if st.button("📝 Give practice problems", key="qa_prac"):
                st.session_state["tutor_quick_dispatch"] = (
                    "Give me scaffolded practice problems (easy → hard) with final answers in separate short hints."
                )
        aq3, aq4 = st.columns(2)
        with aq3:
            if st.button("🔄 Explain differently", key="qa_diff"):
                st.session_state["tutor_quick_dispatch"] = (
                    "Explain the same idea again using a different analogy and a shorter roadmap."
                )
        with aq4:
            if st.button("📋 Summarize key points", key="qa_sum"):
                st.session_state["tutor_quick_dispatch"] = "Summarize the key points from your last answer as a tight checklist."
        if st.button("🎯 What might come in exam?", key="qa_exam"):
            pyq = st.session_state.get("tutor_pyq_result") or {}
            has_ml = bool(pyq and not pyq.get("error"))
            st.session_state["tutor_quick_dispatch"] = exam_hint_prompt(has_ml, plain_subject)

        dispatch = st.session_state.pop("tutor_quick_dispatch", None)

        manual_text_raw = pending_auto.strip() if pending_auto else str(user_in or "").strip()

        outgoing = dispatch.strip() if isinstance(dispatch, str) and dispatch.strip() else manual_text_raw

        if send and manual_text_raw and not outgoing:
            outgoing = manual_text_raw

        if outgoing:

            def _flush_turn(txt: str):
                touch_session_start()
                stamp = datetime.now().strftime("%H:%M:%S")
                hist = list(st.session_state["tutor_messages"])
                ut = txt.strip()
                st.session_state["tutor_messages"].append({"role": "user", "content": ut, "time": stamp})
                msgs = build_ollama_messages(hist, ut, plain_subject, notes_ctx_fn(ut))
                stream_holder = chat_box.empty()
                accumulated = ""
                try:
                    for token in stream_tutor_reply(msgs):
                        accumulated += token
                        stream_holder.markdown(accumulated + "▎")
                    stream_holder.markdown(accumulated)
                except Exception as exc:
                    accumulated = (
                        "**Could not reach Ollama.** Start `ollama serve` and ensure the model is pulled "
                        f"(`ollama pull llama3.2`).\n\n`{exc}`"
                    )
                    stream_holder.markdown(accumulated)

                astamp = datetime.now().strftime("%H:%M:%S")
                st.session_state["tutor_messages"].append(
                    {"role": "assistant", "content": accumulated, "time": astamp, "feedback": None}
                )
                merge_topics_from_message("assistant", accumulated, st.session_state["tutor_topic_status"])

            nonce_out = nonce
            st.session_state.pop("tutor_prefill_buffer", None)
            st.session_state["tutor_input_nonce"] = nonce_out + 1
            with st.spinner("Thinking…"):
                _flush_turn(outgoing)
            st.rerun()

    with right:
        st.markdown("### 🎯 Exam Topic Predictor")

        uploads = st.file_uploader(
            "Upload PYQ Papers (optional)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pyq_tutor_ml",
            help="Adds ML-powered rankings in this panel.",
        )

        predictor_subject = st.text_input(
            "Subject (optional, for your notes)",
            value="",
            key="predictor_subject_ml",
            placeholder="e.g. Data Structures",
        )

        if uploads:
            st.session_state["tutor_pyq_bytes"] = [(f.name, bytes(f.getbuffer())) for f in uploads]

        st.caption("Upload previous-year papers to get TF-IDF + Random Forest topic rankings.")

        if st.button("Predict Important Topics", key="predict_topics_btn_tutor"):
            blobs = st.session_state.get("tutor_pyq_bytes") or []
            if not blobs:
                st.warning("Add at least one PDF first.")
            else:
                bio_list = []
                for name, b in blobs:
                    up = io.BytesIO(b)
                    up.name = name
                    bio_list.append(up)
                with st.spinner("Analyzing papers — extracting questions and training RF…"):
                    out = analyze_pyq_ml_pipeline(bio_list)
                    if out.get("error"):
                        st.error(out["error"])
                    else:
                        st.session_state["tutor_pyq_result"] = out
                        sub = predictor_subject.strip() or "your subject"
                        st.success(f"Analyzed — ranked topics for **{html_module.escape(sub)}**.")

        res = st.session_state.get("tutor_pyq_result")
        if res and not res.get("error"):
            st.caption(f"Analyzed **{res.get('n_years', 0)}** years of papers")
            st.caption(
                f"Found **≈ {res.get('topics_found_note', 0)}** surfaced topics "
                f"({res.get('n_questions_detected', 0)} question-like segments)."
            )
            for level, title in (
                ("HIGH", "🔴 HIGH PRIORITY"),
                ("MEDIUM", "🟡 MEDIUM PRIORITY"),
                ("LOW", "🟢 LOW PRIORITY"),
            ):
                block = [r for r in res.get("ranked", []) if r.get("priority") == level][:12]
                if not block:
                    continue
                st.markdown(f"**{title}**")
                for row in block:
                    label = row.get("topic", "")
                    yrs = row.get("years_label", "")
                    safe_key = f"tp_{level}_{sanitize_copy_id(label)[:28]}_{yrs}"
                    if st.button(f"{label} — ({yrs})", key=safe_key):
                        st.session_state["tutor_auto_send"] = (
                            f'Explain "{label}" in depth with examples and practice problems suited for exams.'
                        )
                        st.rerun()

        st.divider()

        st.markdown("### 📚 Session Topics")
        tl = topic_list_for_panel(st.session_state["tutor_topic_status"])
        ongoing_label = plain_subject if canon_focus != "General" else None

        if tl or ongoing_label:
            st.caption("Topics discussed today:")
            for t in tl:
                st.markdown(f"- **{html_module.escape(t)}** ✅")
            if ongoing_label and ongoing_label not in tl and ongoing_label != "General":
                st.markdown(f"- **{html_module.escape(ongoing_label)}** *(focus area)*")
        else:
            st.caption("_Topics will populate as you chat._")

        st.divider()

        st.markdown("### 💡 Suggested Questions")
        last_ai = ""
        for m in reversed(st.session_state["tutor_messages"]):
            if m.get("role") == "assistant":
                last_ai = m.get("content", "")
                break
        for s in suggest_followups(last_ai, plain_subject):
            h = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]
            if st.button((s[:90] + "…") if len(s) > 90 else s, key=f"sg_{h}"):
                st.session_state["tutor_auto_send"] = s
                st.rerun()

        st.divider()

        st.markdown("### 📊 Session Stats")
        cq = count_user_questions()
        nt = len(st.session_state["tutor_topic_status"])
        mins = session_duration_mins()
        st.metric("Questions asked", cq)
        st.metric("Topics covered", nt)
        st.metric("Session duration", f"{mins:.1f} mins")

        st.divider()
        c_clear, c_export = st.columns(2)
        with c_clear:
            if st.button("Clear Chat", key="clear_chat_btn_tutor"):
                st.session_state["tutor_messages"] = []
                st.session_state["tutor_topic_status"] = {}
                st.session_state["tutor_session_start"] = None
                st.session_state["tutor_input_nonce"] = int(st.session_state.get("tutor_input_nonce", 0)) + 1
                st.rerun()
        with c_export:
            from utils.exporter import export_chat_to_pdf

            if st.session_state["tutor_messages"]:
                pdf = export_chat_to_pdf(
                    st.session_state["tutor_messages"],
                    title="RSGSphere AI Tutor Export",
                )
                st.download_button(
                    "Export Chat",
                    data=pdf,
                    file_name="rsgsphere_ai_tutor_chat.pdf",
                    mime="application/pdf",
                )
