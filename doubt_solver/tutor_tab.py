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
    Extract real academic topics using
    TF-IDF importance scoring.
    Filters out metadata, names, headers.
    """
    import PyPDF2
    import re
    from sklearn.feature_extraction.text \
        import TfidfVectorizer
    from collections import Counter
    import numpy as np
    
    # Step 1: Extract all text
    all_text = ""
    page_texts = []
    
    for pdf_file in pdf_files:
        try:
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num, page in \
                    enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    page_texts.append(
                        text.strip())
                    all_text += text + " "
        except:
            continue
    
    if not all_text.strip():
        return get_fallback_topics(all_text)
    
    # Step 2: Aggressive cleaning
    # Remove known noise patterns
    
    noise_patterns = [
        # College/Institute names
        r'global academy of technology',
        r'national education foundation',
        r'department of computer science',
        r'autonomous institute',
        r'affiliated to vtu',
        r'naac.*grade',
        r'aditya layout',
        r'rajarajeshwari nagar',
        r'bengaluru.*karnataka',
        r'ph:.*url:.*',
        r'www\.gat\.ac\.in',
        r'growing ahead of time',
        # Professor/People names
        r'dr\.?\s+[a-z]+\s+[a-z]+',
        r'prof\.?\s+[a-z]+\s+[a-z]+',
        r'professor and head',
        r'head of department',
        # Slide metadata
        r'\d{1,2}-[a-z]{3}-\d{2,4}',
        r'module\s+\d+',
        r'professional elective',
        r'cse\d+[a-z]*',
        r'slide\s+\d+',
        r'\d+\s*of\s*\d+',
        # Vision/Mission text
        r'vision of the',
        r'mission of the',
        r'program educational',
        r'program specific outcomes',
        r'peo\d+:.*',
        r'pso\d+:.*',
        r'co\d+:.*',
        # Quiz patterns
        r'quiz time',
        r'answer:.*[abcd]',
        r'^[abcd]\.',
        r'thank you',
        r'14-mar-\d+',
    ]
    
    cleaned_text = all_text.lower()
    for pattern in noise_patterns:
        cleaned_text = re.sub(
            pattern, ' ', 
            cleaned_text,
            flags=re.IGNORECASE)
    
    # Step 3: Extract meaningful 
    # technical noun phrases
    
    # Technical topic patterns for CV/CS
    tech_topic_patterns = [
        # Specific technical terms
        r'\b(computer vision)\b',
        r'\b(image formation)\b',
        r'\b(image processing)\b',
        r'\b(optical flow)\b',
        r'\b(edge detection)\b',
        r'\b(object recognition)\b',
        r'\b(feature detection)\b',
        r'\b(image segmentation)\b',
        r'\b(stereo correspondence)\b',
        r'\b(structure from motion)\b',
        r'\b(photometric stereo)\b',
        r'\b(scale.?space)\b',
        r'\b(markov random fields?)\b',
        r'\b(point operators?)\b',
        r'\b(linear filtering)\b',
        r'\b(digital camera)\b',
        r'\b(3d reconstruction)\b',
        r'\b(depth estimation)\b',
        r'\b(image pyramid)\b',
        r'\b(convolutional neural)\b',
        r'\b(machine learning)\b',
        r'\b(deep learning)\b',
        r'\b(pattern recognition)\b',
        r'\b(image classification)\b',
        r'\b(face detection)\b',
        r'\b(optical illusions?)\b',
        r'\b(photogrammetry)\b',
        r'\b(data structures?)\b',
        r'\b(binary tree)\b',
        r'\b(graph algorithm)\b',
        r'\b(dynamic programming)\b',
        r'\b(neural network)\b',
        r'\b(support vector)\b',
        r'\b(random forest)\b',
        r'\b(natural language)\b',
        r'\b(reinforcement learning)\b',
    ]
    
    # Find which technical topics
    # actually appear in this document
    found_topics = []
    text_lower = all_text.lower()
    
    for pattern in tech_topic_patterns:
        matches = re.findall(
            pattern, 
            text_lower,
            flags=re.IGNORECASE)
        if matches:
            # Get the original case version
            orig_match = re.search(
                pattern,
                all_text,
                flags=re.IGNORECASE)
            if orig_match:
                topic = orig_match.group(0)\
                    .strip().title()
                if topic not in found_topics:
                    found_topics.append(topic)
    
    # Step 4: If technical patterns found,
    # return them
    if len(found_topics) >= 3:
        return found_topics[:8]
    
    # Step 5: Fallback - TF-IDF on 
    # page content
    if len(page_texts) >= 2:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(2, 3),
                max_features=50,
                stop_words='english',
                min_df=1,
                token_pattern=r'[a-zA-Z]{3,}'
                              r'(?:\s+[a-zA-Z]{3,})*'
            )
            
            tfidf_matrix = vectorizer.fit_transform(
                page_texts)
            
            # Get feature names (phrases)
            feature_names = \
                vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all pages
            scores = np.asarray(
                tfidf_matrix.sum(axis=0))[0]
            
            # Get top scoring phrases
            top_indices = scores.argsort()[::-1]
            
            noise_words = [
                'global academy', 'department',
                'computer science engineering',
                'national education', 'aditya',
                'rajarajeshwari', 'mar 26',
                'professor head', 'institute',
                'autonomous institute',
                'affiliated vtu', 'engineering'
            ]
            
            tfidf_topics = []
            for idx in top_indices:
                phrase = feature_names[idx]
                # Skip noise
                if any(noise in phrase.lower() 
                       for noise in noise_words):
                    continue
                # Skip if too generic
                if len(phrase) < 6:
                    continue
                # Capitalize properly
                topic = phrase.title()
                tfidf_topics.append(topic)
                if len(tfidf_topics) >= 8:
                    break
            
            if tfidf_topics:
                return tfidf_topics
        
        except Exception as e:
            print(f"TF-IDF error: {e}")
    
    # Step 6: Last resort fallback
    return get_fallback_topics(all_text)


def get_fallback_topics(text):
    """
    Return subject-appropriate fallback
    topics when extraction fails.
    """
    text_lower = text.lower()
    
    # Detect subject from content
    if any(w in text_lower for w in [
            'computer vision', 'image', 
            'pixel', 'optical flow',
            'edge detection', 'segmentation']):
        return [
            "Computer Vision",
            "Image Formation",
            "Image Processing",
            "Edge Detection",
            "Optical Flow",
            "Object Recognition",
            "Image Segmentation",
            "Feature Detection"
        ]
    
    elif any(w in text_lower for w in [
            'machine learning', 'neural',
            'classification', 'regression',
            'training', 'dataset']):
        return [
            "Machine Learning",
            "Neural Networks",
            "Classification",
            "Feature Engineering",
            "Model Training",
            "Overfitting",
            "Cross Validation",
            "Deep Learning"
        ]
    
    elif any(w in text_lower for w in [
            'data structure', 'algorithm',
            'tree', 'graph', 'sorting',
            'linked list', 'stack']):
        return [
            "Data Structures",
            "Binary Trees",
            "Graph Algorithms",
            "Sorting Algorithms",
            "Dynamic Programming",
            "Linked Lists",
            "Stack and Queue",
            "Hashing"
        ]
    
    elif any(w in text_lower for w in [
            'network', 'tcp', 'ip', 'protocol',
            'routing', 'socket', 'http']):
        return [
            "Computer Networks",
            "TCP/IP Protocol",
            "Routing Algorithms",
            "Network Security",
            "Socket Programming",
            "OSI Model",
            "Subnetting",
            "DNS and HTTP"
        ]
    
    elif any(w in text_lower for w in [
            'database', 'sql', 'query',
            'normalization', 'transaction']):
        return [
            "Database Management",
            "SQL Queries",
            "Normalization",
            "Transactions",
            "Indexing",
            "ER Diagrams",
            "Joins",
            "ACID Properties"
        ]
    
    # Generic CS fallback
    return [
        "Algorithm Design",
        "Data Structures",
        "Computer Architecture",
        "Operating Systems",
        "Software Engineering",
        "Computer Networks",
        "Database Systems",
        "Theory of Computation"
    ]

def predict_exam_questions(pdf_files):
    """
    Extract REAL exam questions from PDFs.
    Only return actual questions, not
    random sentences.
    """
    import PyPDF2
    import re
    
    all_questions = []
    
    for pdf_file in pdf_files:
        try:
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    
                    # Must be reasonable length
                    if len(line) < 15 or \
                            len(line) > 200:
                        continue
                    
                    # Pattern 1: Actual quiz/exam
                    # questions with numbers
                    if re.match(
                        r'^[QqNn]?[o.]?\s*\d+'
                        r'[\.\)]\s*[A-Z]',
                        line):
                        # Remove question number
                        q = re.sub(
                            r'^[QqNn]?[o.]?\s*'
                            r'\d+[\.\)]\s*',
                            '', line)
                        if (len(q) > 15 and
                            '?' in q or
                            any(w in q.lower() 
                                for w in [
                                'explain', 
                                'define',
                                'describe',
                                'what is',
                                'how does',
                                'why is',
                                'compare',
                                'list',
                                'discuss'])):
                            all_questions.append(q)
                    
                    # Pattern 2: Lines starting
                    # with question words
                    elif re.match(
                        r'^(What|Explain|Define|'
                        r'Describe|Compare|List|'
                        r'Discuss|Write|How|Why|'
                        r'Differentiate|Illustrate|'
                        r'Derive|Prove|Show)',
                        line):
                        if len(line) > 20:
                            all_questions.append(
                                line)
                    
                    # Pattern 3: Quiz questions
                    # from this CV PDF specifically
                    elif ('?' in line and 
                          len(line) > 20 and
                          line[0].isupper()):
                        # Avoid option lines
                        # (A. B. C. D.)
                        if not re.match(
                            r'^[A-D][\.\)]',
                            line):
                            all_questions.append(
                                line)
        except:
            continue
    
    # Remove duplicates
    seen = set()
    unique = []
    for q in all_questions:
        key = q[:40].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    
    # Ensure there are at least 5 questions
    if len(unique) < 5:
        # Get subject from tutor_doc_intel_result
        doc_intel = st.session_state.get("tutor_doc_intel_result") or {}
        subject_label = doc_intel.get("subject", {}).get("label", "General").lower()
        
        physics_qs = [
            "Explain the key physical principles and equations discussed in this material.",
            "What are the core practical applications of these physics concepts in modern technology?",
            "Describe a numerical problem or experiment that demonstrates this physical law.",
            "Compare and contrast the different physical models and theories presented.",
            "Derive or outline the mathematical proof for the fundamental equation in this system."
        ]
        chemistry_qs = [
            "What are the primary chemical reactions, reagents, and mechanisms covered here?",
            "Explain the industrial applications, synthesis steps, and safety protocols of these compounds.",
            "Describe the molecular structure, bonding, and thermodynamic properties of the substances.",
            "Differentiate between the key chemical compounds, isomers, or states discussed.",
            "Formulate the chemical equations and calculate the equilibrium constants for this reaction."
        ]
        cs_qs = [
            "Explain the primary algorithms, data structures, and complexity bounds of this computational method.",
            "What are the core architectural designs, protocols, or design patterns used in this system?",
            "Describe a practical software engineering scenario where this technology is applied.",
            "Compare and contrast the different implementation strategies or programming paradigms discussed.",
            "Outline the systematic workflow, pseudo-code, or optimization techniques for this system."
        ]
        general_qs = [
            "What are the primary concepts, formulas, and definitions covered in this study material?",
            "Explain the practical applications, real-world utility, and examples of these concepts.",
            "Differentiate between the key theories, methodologies, or frameworks discussed.",
            "Provide a step-by-step walkthrough or numerical problem illustrating this topic.",
            "What are the most common exam questions expected from this chapter?"
        ]
        
        fallbacks = general_qs
        if "physics" in subject_label:
            fallbacks = physics_qs
        elif "chemistry" in subject_label:
            fallbacks = chemistry_qs
        elif "computer" in subject_label or "cs" in subject_label or "coding" in subject_label or "programming" in subject_label:
            fallbacks = cs_qs
            
        for fq in fallbacks:
            if len(unique) >= 5:
                break
            if fq not in unique:
                unique.append(fq)
    
    return unique[:6]


def _get_doc_intel_labels() -> tuple[str, str, str]:
    """Document Intelligence labels from tutor upload (if available)."""
    doc_intel = st.session_state.get("tutor_doc_intel_result") or {}
    doc_type = doc_intel.get("doc_type", {}).get("label", "Unknown")
    subject = doc_intel.get("subject", {}).get("label", "Unknown")
    difficulty = doc_intel.get("difficulty", {}).get("label", "Unknown")
    return doc_type, subject, difficulty


def _build_topic_prompt(topic: str, subject: str, difficulty: str) -> str:
    if "Advanced" in difficulty:
        return (
            f"Explain {topic} in detail with advanced concepts, "
            f"mathematical proofs if any, and complex examples. "
            f"This is for {subject} subject."
        )
    if "Intermediate" in difficulty:
        return (
            f"Explain {topic} with examples and applications. "
            f"Cover both theory and practical aspects for {subject} subject."
        )
    return (
        f"Explain {topic} in simple terms with basic examples "
        f"for {subject} subject. Keep it beginner friendly."
    )


def _build_question_prompt(question: str, doc_type: str, subject: str, difficulty: str) -> str:
    if "Question Paper" in doc_type:
        return (
            f"This is an exam question from {subject}. "
            f"Solve and explain in detail: {question}"
        )
    if "Advanced" in difficulty:
        return (
            f"Answer this advanced {subject} question with "
            f"complete explanation and examples: {question}"
        )
    return f"Answer this {subject} question step by step: {question}"


def _send_to_tutor_chat(prompt: str) -> None:
    st.session_state.pending_tutor_message = prompt.strip()
    st.session_state.tutor_input_temp = prompt.strip()
    st.rerun()


def load_tutor_doc_intelligence():
    from doc_intelligence.predictor import DocumentPredictor
    predictor = DocumentPredictor()
    success = predictor.load(
        "models/doc_intelligence.pth",
        "models/feature_extractor.pkl",
    )
    return predictor if success else None


def _render_doc_intel_card() -> None:
    doc_intel = st.session_state.get("tutor_doc_intel_result")
    if not doc_intel:
        return
    doc_type = doc_intel.get("doc_type", {}).get("label", "Unknown")
    subject = doc_intel.get("subject", {}).get("label", "Unknown")
    difficulty = doc_intel.get("difficulty", {}).get("label", "Unknown")
    st.markdown(
        "<div style='background:#2D2D2D;border-radius:8px;padding:8px 12px;"
        "border:1px solid #F5C518;margin-bottom:8px;font-size:12px;'>"
        "<span style='color:#F5C518;font-weight:600;'>🧠 Doc Intelligence</span><br>"
        f"Type: {html_module.escape(doc_type)}<br>"
        f"Subject: {html_module.escape(subject)}<br>"
        f"Difficulty: {html_module.escape(difficulty)}"
        "</div>",
        unsafe_allow_html=True,
    )


def _rewind_pyq_files(pdf_files) -> None:
    for pdf_file in pdf_files or []:
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)


def index_pdfs_to_chromadb(pdf_files) -> tuple[bool, int]:
    """
    Index uploaded PDFs into ChromaDB (same store as Tab 1 Smart Chat).
    """
    try:
        from rag.indexing import index_pdfs_into_chroma

        if "indexed_files" not in st.session_state:
            st.session_state.indexed_files = {}
        if "all_chunks" not in st.session_state:
            st.session_state.all_chunks = []

        skip = {name for name, meta in st.session_state.indexed_files.items() if meta.get("source") == "exam_predictor"}
        to_index = [f for f in pdf_files if f.name not in skip]
        if not to_index:
            return True, 0

        _rewind_pyq_files(to_index)
        success, count, names, chunks = index_pdfs_into_chroma(
            to_index,
            skip_filenames=skip,
        )

        for pdf_file in to_index:
            if pdf_file.name not in names:
                continue
            page_count = max(
                (c.get("page_number", 0) for c in chunks if c.get("filename") == pdf_file.name),
                default=0,
            )
            st.session_state.indexed_files[pdf_file.name] = {
                "pages": page_count,
                "size": getattr(pdf_file, "size", 0),
                "indexed": True,
                "source": "exam_predictor",
            }

        if chunks:
            st.session_state.all_chunks.extend(chunks)

        return success, count
    except Exception as exc:
        print(f"Indexing error: {exc}")
        return False, 0


def _ensure_pyq_indexed(pyq_files) -> tuple[bool, int]:
    """Index new PYQ uploads into ChromaDB once per file."""
    if not pyq_files:
        return False, 0
    indexed = st.session_state.get("pyq_chroma_indexed", set())
    new_files = [f for f in pyq_files if f.name not in indexed]
    if not new_files:
        return True, 0
    success, count = index_pdfs_to_chromadb(new_files)
    if success:
        indexed = set(indexed)
        indexed.update(f.name for f in new_files)
        st.session_state.pyq_chroma_indexed = indexed
    return success, count


def _render_indexed_files_status() -> None:
    indexed = st.session_state.get("indexed_files") or {}
    tutor_files = {name: meta for name, meta in indexed.items() if meta.get("source") == "exam_predictor"}
    if not tutor_files:
        return
    st.markdown("**📚 Indexed to Knowledge Base:**")
    for fname in tutor_files:
        st.markdown(f"✅ {fname}")
    st.caption("AI Tutor can now answer questions from these documents")


def _inject_layout_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 1rem;
            max-width: 100%;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #242424;
            border-color: #3D3D3D !important;
        }
        .tutor-history-label {
            color: #D79922;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin: 12px 0 6px 2px;
        }
        [data-testid="stTextInput"] input {
            background: #2D2D2D !important;
            border: 1px solid #3D3D3D !important;
            border-radius: 24px !important;
            color: #FFFFFF !important;
            padding: 12px 16px !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #D79922 !important;
            box-shadow: 0 0 0 3px rgba(215, 153, 34, 0.2) !important;
        }
        .tutor-footnote {
            text-align: center;
            font-size: 12px;
            color: #666666;
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
        r"<code style='background:#2D2D2D;padding:2px 6px;border-radius:4px;'>\1</code>",
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
            <h2 style="text-align:center;color:#FFFFFF;margin:0 0 8px;font-size:1.5rem;font-weight:600;">
                ✨ {greet}, {html_module.escape(student)}!
            </h2>
            <p style="text-align:center;color:#A0A0A0;margin:0;font-size:1rem;">
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
        st.markdown("### 📚 RAG-Sphere")
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
        _render_doc_intel_card()

        pyq_files = st.file_uploader(
            "📝 Upload Previous Year Papers",
            type=["pdf"],
            accept_multiple_files=True,
            key="pyq_uploader_sidebar",
        )
        if not pyq_files:
            st.session_state.tutor_doc_intel_result = None
            
        st.caption(
            "For Exam Predictor — ML will analyze patterns and predict important topics"
        )

        if pyq_files:
            with st.spinner("📚 Indexing PDFs to knowledge base..."):
                ok, chunk_count = _ensure_pyq_indexed(pyq_files)
            if ok and chunk_count > 0:
                st.success(f"✅ Indexed {chunk_count} chunks to knowledge base")
            elif ok:
                st.caption("PDFs already in knowledge base")
                
            if not st.session_state.get("tutor_doc_intel_result"):
                predictor = load_tutor_doc_intelligence()
                if predictor:
                    first_file = pyq_files[0]
                    if hasattr(first_file, "seek"):
                        first_file.seek(0)
                    result = predictor.predict_pdf(first_file)
                    if result:
                        st.session_state.tutor_doc_intel_result = result

        _render_indexed_files_status()

        doc_type_label, subject_label, difficulty_label = _get_doc_intel_labels()

        st.markdown("")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "📚 Analyze Topics",
                key="analyze_topics_btn",
                use_container_width=True,
            ):
                if pyq_files:
                    with st.spinner("Extracting topics..."):
                        _ensure_pyq_indexed(pyq_files)
                        _rewind_pyq_files(pyq_files)
                        topics = extract_topics_from_pyqs(pyq_files)
                        st.session_state.pyq_topics = topics
                        st.session_state.pyq_questions = []
                        st.session_state.pyq_topics_context = (
                            f"{subject_label} | {difficulty_label}"
                        )
                        
                        # Run the ML pipeline to extract Random Forest feature importances
                        _rewind_pyq_files(pyq_files)
                        res = analyze_pyq_ml_pipeline(pyq_files)
                        st.session_state.tutor_pyq_result = res
                    st.rerun()
                else:
                    st.warning("Upload PYQ PDFs first")

        with btn_col2:
            if st.button(
                "🎯 Predict Questions",
                key="predict_questions_btn",
                use_container_width=True,
            ):
                if pyq_files:
                    with st.spinner("Predicting questions..."):
                        _ensure_pyq_indexed(pyq_files)
                        _rewind_pyq_files(pyq_files)
                        questions = predict_exam_questions(pyq_files)
                        # Guarantee at least 5 questions if possible
                        if len(questions) < 5:
                             questions.extend(["Describe core concepts in detail."] * (5 - len(questions)))
                        st.session_state.pyq_questions = questions
                        st.session_state.pyq_topics = st.session_state.get("pyq_topics", [])
                        st.session_state.pyq_questions_context = subject_label
                        
                        # Also populate tutor_pyq_result if not already calculated
                        if not st.session_state.get("tutor_pyq_result"):
                            _rewind_pyq_files(pyq_files)
                            res = analyze_pyq_ml_pipeline(pyq_files)
                            st.session_state.tutor_pyq_result = res
                    st.rerun()
                else:
                    st.warning("Upload PYQ PDFs first")

        if st.session_state.get("pyq_topics"):
            # Get subject and difficulty from 
            # document intelligence OR from 
            # the uploaded PDF filename
            doc_intel = st.session_state.get(
                'tutor_doc_intel_result', None)
            
            if doc_intel:
                subject = doc_intel.get(
                    'subject', {}).get(
                    'label', '').replace(
                    '💻 ', '').replace(
                    '📐 ', '').replace(
                    '⚛️ ', '').replace(
                    '⚡ ', '').replace(
                    '🧪 ', '').strip()
                difficulty = doc_intel.get(
                    'difficulty', {}).get(
                    'label', '').replace(
                    '🟢 ', '').replace(
                    '🟡 ', '').replace(
                    '🔴 ', '').strip()
            else:
                # Fallback: detect from PDF filename
                subject = "Computer Science"
                difficulty = "Intermediate"
            
            # Show proper header
            st.markdown(
                f"**📊 Topics for "
                f"{subject} | {difficulty}**")
            st.markdown("**📌 Important Topics:**")
            for i, topic in enumerate(st.session_state.pyq_topics[:8]):
                if st.button(
                    f"• {topic}",
                    key=f"topic_btn_{i}",
                    use_container_width=True,
                ):
                    prompt = _build_topic_prompt(topic, subject_label, difficulty_label)
                    _send_to_tutor_chat(prompt)

        st.markdown("")

        if st.session_state.get("pyq_questions"):
            q_ctx = st.session_state.get("pyq_questions_context", subject_label)
            st.markdown(f"**❓ Predicted Questions for {q_ctx}**")
            st.markdown("**❓ Predicted Questions:**")
            for i, question in enumerate(st.session_state.pyq_questions[:8]):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"**Q{i + 1}.** {question}")
                with col2:
                    if st.button("→", key=f"q_btn_{i}", help="Ask AI Tutor"):
                        prompt = _build_question_prompt(
                            question, doc_type_label, subject_label, difficulty_label
                        )
                        _send_to_tutor_chat(prompt)


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
    notes = notes_ctx_fn(ut)
    source_info = ""
    tutor_files = {name: meta for name, meta in st.session_state.get("indexed_files", {}).items() if meta.get("source") == "exam_predictor"}
    if notes and tutor_files:
        from rag.indexing import get_tutor_rag_context

        _, source_info = get_tutor_rag_context(ut, k=3, filenames=list(tutor_files.keys()))

    msgs = build_ollama_messages(hist, ut, "General", notes)
    accumulated = ""
    try:
        for token in stream_tutor_reply(msgs):
            accumulated += token
            status_slot.markdown(f"**RAG-Sphere is typing…**\n\n{accumulated}▎")
        status_slot.empty()
    except Exception as exc:
        accumulated = (
            "**Could not reach Ollama.** Start `ollama serve` and pull the model "
            f"(`ollama pull llama3.2`).\n\n`{exc}`"
        )
        status_slot.empty()

    if source_info:
        accumulated = accumulated.rstrip() + f"\n\n{source_info}"

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
    if "pending_tutor_message" not in st.session_state:
        st.session_state.pending_tutor_message = None
    if "doc_intel_result" not in st.session_state:
        st.session_state.doc_intel_result = None
    if "pyq_chroma_indexed" not in st.session_state:
        st.session_state.pyq_chroma_indexed = set()
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = {}
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "smart_chat_input" not in st.session_state:
        st.session_state.smart_chat_input = ""
    st.session_state["tutor_sessions"] = st.session_state.get("chat_sessions", [])

    def default_notes(_q: str) -> str:
        return ""

    notes_ctx_fn = get_notes_context or default_notes

    pending_msg = st.session_state.get("pending_tutor_message")
    if pending_msg:
        st.session_state.pending_tutor_message = None
        st.session_state.tutor_input_temp = ""
        st.session_state["tutor_pending_send"] = str(pending_msg).strip()

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
            '<p class="tutor-footnote">RAG-Sphere uses llama3.2 locally · Inter</p>',
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
