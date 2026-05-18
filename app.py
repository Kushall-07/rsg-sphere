"""RSGSphere Streamlit app: RAG Smart Chat + AI Tutor (Ollama) + ML dashboard."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from doc_intelligence.predictor import DocumentPredictor
from doc_intelligence.visualizer import (
    load_final_metrics,
    load_training_history,
    plot_accuracy_curves,
    plot_architecture,
    plot_loss_curves,
)
from ml.intent_classifier import load_training_data, predict_intent, train_intent_models
from doubt_solver import render_ai_tutor_tab
from ml.visualizer import (
    class_distribution_figure,
    confusion_matrix_figure,
    model_metrics_figure,
    pca_scatter,
    similarity_heatmap,
)
from rag.embedder import EmbeddingService
from rag.fusion import bm25_search, hybrid_fusion
from rag.generator import build_prompt, generate_answer_stream
from rag.loader import chunk_text, extract_text_from_pdf
from rag.reranker import SVMReranker
from rag.retriever import ChromaRetriever
from utils.confidence import compute_confidence
from utils.exporter import export_chat_to_pdf
from utils.voice import speak_text

st.set_page_config(page_title="RSGSphere", page_icon="📚", layout="wide")


@st.cache_resource
def get_embedder():
    """Create one shared embedding service for the app session."""
    return EmbeddingService()


@st.cache_resource
def get_retriever():
    """Create one shared Chroma retriever for the app session."""
    return ChromaRetriever(db_path="chroma_db")


@st.cache_resource
def get_reranker():
    """Create one shared reranker for the app session."""
    return SVMReranker(model_path="models/reranker_svm.pkl")


@st.cache_resource
def load_doc_intelligence():
    """Load pre-trained document intelligence model (inference only)."""
    predictor = DocumentPredictor()
    success = predictor.load(
        "models/doc_intelligence.pth",
        "models/feature_extractor.pkl",
    )
    return predictor if success else None


def init_state():
    """Initialize Streamlit session-state keys used across tabs."""
    defaults = {
        "chat": [],
        "indexed_files": {},
        "all_chunks": [],
        "question_count": 0,
        "conf_scores": [],
        "timings": [],
        "exam_result": None,
        "intent_training": None,
        "chat_input_nonce": 0,
        "cache_hits": 0,
        "show_splash": True,
        "doc_intel_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme():
    """Inject custom dark theme and chat bubble styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #1A1A1A;
            --bg-secondary: #242424;
            --bg-card: #2D2D2D;
            --accent: #D79922;
            --accent-hover: #C28A1E;
            --accent-muted: #9B6E18;
            --border: #3D3D3D;
            --border-accent: #D79922;
            --text-primary: #FFFFFF;
            --text-secondary: #A0A0A0;
            --text-muted: #666666;
            --shadow: rgba(215, 153, 34, 0.3);
            --glow: rgba(215, 153, 34, 0.15);
        }

        body, .main, .block-container, .stApp {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif;
            background-image: none !important;
        }
        
        /* Headers & Text */
        h1, h2, h3, h4, h5, h6, p, span, div {
            font-family: 'Inter', sans-serif;
        }
        
        /* Chat bubbles */
        .chat-user {
            background: var(--accent);
            padding: 16px 20px;
            border-radius: 20px 20px 0 20px;
            margin: 12px 0 12px 20%;
            font-size: 15px;
            line-height: 1.5;
            color: var(--bg-primary);
            font-weight: 500;
        }
        .chat-bot {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 20px 20px 20px 0;
            margin: 12px 20% 12px 0;
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-primary);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            background: var(--accent) !important;
            color: var(--bg-primary) !important;
            transition: all 0.3s ease !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px var(--shadow) !important;
        }
        .stButton > button:hover {
            background: var(--accent-hover) !important;
            border-color: var(--accent) !important;
            transform: translateY(-2px);
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            color: var(--text-primary);
            transition: border-color 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--glow);
        }
        .stTextInput > div > div > input::placeholder {
            color: var(--text-muted) !important;
        }
        
        /* Tab Container */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-secondary);
            padding: 6px;
            border-radius: 16px;
            border: 1px solid var(--border);
            width: fit-content;
        }

        /* Individual Tab */
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 24px;
            border-radius: 12px;
            border: 1px solid transparent;
            background: transparent;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0.3px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        /* Tab Hover */
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(215, 153, 34, 0.1);
            color: var(--accent);
            border-color: rgba(215, 153, 34, 0.3);
        }

        /* Active Tab */
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(
                135deg, 
                var(--accent) 0%, 
                var(--accent-muted) 100%
            );
            color: var(--bg-primary) !important;
            border-color: transparent;
            box-shadow: 0 4px 15px rgba(215, 153, 34, 0.4);
            font-weight: 600;
        }

        /* Remove default underline indicator */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* Tab panel */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 20px;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
        }
        
        /* Badges */
        .intent-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* Cards */
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
            box-shadow: 0 10px 25px -5px var(--glow);
        }
        .feature-icon {
            font-size: 32px;
            margin-bottom: 12px;
            display: block;
            color: var(--accent);
        }
        .feature-title {
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 8px;
            color: var(--text-primary);
        }
        
        /* Custom Radio Buttons / Quick Questions */
        .stRadio > div[role="radiogroup"] > label {
            background: rgba(215, 153, 34, 0.1);
            border: 1px solid rgba(215, 153, 34, 0.3);
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.2s ease;
            color: var(--accent);
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            background: rgba(215, 153, 34, 0.2);
            border-color: var(--accent);
        }
        
        /* Progress bars / Elements */
        .stProgress > div > div > div > div {
            background-color: var(--accent);
        }
        
        /* Expanders */
        .st-emotion-cache-1z1mbg4 {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded):
    """Persist uploaded file to local data/uploads for offline access."""
    uploads = Path("data/uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    out = uploads / uploaded.name
    with open(out, "wb") as file:
        file.write(uploaded.getbuffer())
    return out


def index_documents(uploaded_files):
    """Extract, chunk, embed, and index uploaded documents in vector DB."""
    from rag.indexing import index_pdfs_into_chroma

    embedder = get_embedder()
    retriever = get_retriever()
    with st.spinner("Indexing documents..."):
        for up in uploaded_files:
            if up.name in st.session_state["indexed_files"]:
                continue
            save_uploaded_file(up)
            if hasattr(up, "seek"):
                up.seek(0)
            _ok, _count, names, chunks = index_pdfs_into_chroma(
                [up],
                embedder=embedder,
                retriever=retriever,
                skip_filenames=set(st.session_state["indexed_files"].keys()),
            )
            if not chunks:
                continue
            st.session_state["all_chunks"].extend(chunks)
            page_count = max((c.get("page_number", 0) for c in chunks), default=0)
            st.session_state["indexed_files"][up.name] = {
                "pages": page_count,
                "size": up.size,
                "indexed": True,
            }


def build_tutor_notes_context(query: str) -> str:
    """Optional excerpts from indexed documents for tutor grounding."""
    if not st.session_state.get("indexed_files"):
        return ""
    from rag.indexing import get_tutor_rag_context

    context, _ = get_tutor_rag_context(
        query, embedder=get_embedder(), retriever=get_retriever(), k=5
    )
    return context


def render_sidebar():
    """Render sidebar upload controls and session stats."""
    st.sidebar.title("📚 RSGSphere")
    st.sidebar.caption("Study Smart. Not Hard.")
    st.sidebar.info("💡 Upload your college PDFs here to chat with them in Smart Chat")
    uploaded = st.sidebar.file_uploader(
        "📚 Upload Study Material",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.sidebar.caption(
        "For Smart Chat — ask questions from your notes/syllabus/textbook"
    )
    if uploaded:
        index_documents(uploaded)

    predictor = load_doc_intelligence()
    if predictor and uploaded:
        for pdf_file in uploaded:
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            result = predictor.predict_pdf(pdf_file)
            if result:
                st.session_state.doc_intel_result = result
                st.sidebar.markdown(
                    """
                    <div style="background:#2D2D2D;border-radius:10px;padding:12px;
                        border:1px solid #3D3D3D;margin-top:8px;">
                        <div style="color:#F5C518;font-size:12px;font-weight:600;
                            margin-bottom:8px;">
                            🧠 Document Intelligence
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.sidebar.metric(
                        "Type",
                        result["doc_type"]["label"],
                        f"{result['doc_type']['confidence']:.0f}%",
                    )
                with col2:
                    st.sidebar.metric(
                        "Subject",
                        result["subject"]["label"],
                        f"{result['subject']['confidence']:.0f}%",
                    )
                st.sidebar.metric(
                    "Difficulty",
                    result["difficulty"]["label"],
                    f"{result['difficulty']['confidence']:.0f}%",
                )

    st.sidebar.markdown("### Uploaded Files")
    to_delete = None
    for name, meta in st.session_state["indexed_files"].items():
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            st.write(f"`{name}`")
            st.caption(f"{meta['pages']} pages | {meta['size']/1024:.1f} KB | ✅ Indexed")
        with col2:
            if st.button("🗑️", key=f"del-{name}"):
                to_delete = name
    if to_delete:
        st.session_state["indexed_files"].pop(to_delete, None)

    if st.sidebar.button("Clear All Documents", key="clear_all_docs_btn"):
        st.session_state["indexed_files"] = {}
        st.session_state["all_chunks"] = []

    if st.session_state["chat"]:
        pdf_bytes = export_chat_to_pdf(st.session_state["chat"])
        st.sidebar.download_button("Export Chat as PDF", data=pdf_bytes, file_name="rsgsphere_chat.pdf")

    st.sidebar.divider()
    avg_conf = np.mean(st.session_state["conf_scores"]) if st.session_state["conf_scores"] else 0.0
    st.sidebar.write(f"Questions asked: {st.session_state['question_count']}")
    st.sidebar.write(f"Documents indexed: {len(st.session_state['indexed_files'])}")
    st.sidebar.write(f"Avg confidence: {avg_conf:.1f}%")


def render_welcome():
    """Render initial landing state before documents are uploaded."""
    st.markdown("<h2 style='text-align: center; font-weight: 700; margin-bottom: 10px; color: #FFFFFF;'>📚 RSGSphere</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 18px; margin-bottom: 40px;'>Study Smart. Not Hard.</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    c1.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📁</span>
            <div class="feature-title">1. Upload PDFs</div>
            <div style="color: #A0A0A0; font-size: 14px;">Add your study materials and syllabus in the sidebar.</div>
        </div>
    """, unsafe_allow_html=True)
    
    c2.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">2. AI Tutor</div>
            <div style="color: #A0A0A0; font-size: 14px;">Ask questions and get instant, context-aware answers.</div>
        </div>
    """, unsafe_allow_html=True)
    
    c3.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">3. See ML Working</div>
            <div style="color: #A0A0A0; font-size: 14px;">Explore the dashboard to see how the models operate.</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; color: #666666; margin-top: 20px;'>👈 Start by uploading your study material in the sidebar.</div>", unsafe_allow_html=True)


def intent_badge(intent: str):
    """Return emoji and color metadata for detected intent display."""
    badges = {
        "fees": ("💰 FEES", "#D79922"),
        "exam": ("📚 EXAM", "#9B6E18"),
        "placement": ("🏢 PLACEMENT", "#3D3D3D"),
        "hostel": ("🏠 HOSTEL", "#D79922"),
        "library": ("📖 LIBRARY", "#9B6E18"),
        "general": ("ℹ️ GENERAL", "#666666"),
    }
    return badges.get(intent, badges["general"])


def process_query(query: str):
    """Run complete RAG pipeline and stream generated response."""
    if not st.session_state["indexed_files"]:
        st.error("Please upload documents first")
        return

    embedder = get_embedder()
    retriever = get_retriever()
    reranker = get_reranker()

    status = st.empty()
    steps = ["📄 Reading", "✂️ Chunking", "🔢 Embedding", "🔍 Retrieving", "🤖 Reranking", "💬 Generating"]
    timing = {}

    t0 = time.time()
    status.info(f"{steps[0]}...")
    time.sleep(0.1)
    timing[steps[0]] = (time.time() - t0) * 1000

    t0 = time.time()
    status.info(f"{steps[1]}...")
    time.sleep(0.1)
    timing[steps[1]] = (time.time() - t0) * 1000

    t0 = time.time()
    status.info(f"{steps[2]}...")
    q_emb = embedder.embed_query(query)
    timing[steps[2]] = (time.time() - t0) * 1000

    t0 = time.time()
    status.info(f"{steps[3]}...")
    sem = retriever.semantic_search(q_emb, k=10)
    timing[steps[3]] = (time.time() - t0) * 1000

    t0 = time.time()
    status.info(f"{steps[4]}...")
    bm = bm25_search(query, sem, k=10)
    fused = hybrid_fusion(sem, bm)
    top = reranker.rerank(query, fused, top_k=3)
    timing[steps[4]] = (time.time() - t0) * 1000

    intent, intent_conf, _ = predict_intent(query)
    prompt = build_prompt(query, top, intent)

    t0 = time.time()
    status.info(f"{steps[5]}...")
    answer_placeholder = st.empty()
    answer = ""
    try:
        for token in generate_answer_stream(prompt):
            answer += token
            answer_placeholder.markdown(f"<div class='chat-bot'>{answer}▌</div>", unsafe_allow_html=True)
    except Exception:
        st.error("Please start Ollama: run 'ollama serve' in terminal")
        return
    timing[steps[5]] = (time.time() - t0) * 1000

    status.success("Pipeline complete")
    conf = compute_confidence(intent_conf, [c.get("fusion_score", 0.0) for c in top])
    st.session_state["conf_scores"].append(conf)
    st.session_state["question_count"] += 1
    st.session_state["timings"].append(timing)

    badge_text, badge_color = intent_badge(intent)
    source = top[0] if top else {"filename": "N/A", "page_number": "N/A"}
    card = f"""
<div class='chat-bot'>
<b><span class='intent-badge' style='background:{badge_color}'>{badge_text}</span> | {conf:.1f}% confident</b><br/>
⚡ Hybrid Retrieval (Vector + BM25)<br/><br/>
{answer}<br/><br/>
📄 Source: {source['filename']} | Page {source['page_number']}<br/>
📊 Confidence: {'█'*int(conf//10)}{'░'*(10-int(conf//10))} {conf:.1f}%<br/>
💡 Related Questions: [Exam policy?] [Fees?] [Schedule?]
</div>
"""
    answer_placeholder.markdown(card, unsafe_allow_html=True)

    ra_col, _ = st.columns([1, 8])
    with ra_col:
        if st.button("🔊 Read Aloud", key=f"speak_live_{st.session_state['question_count']}"):
            with st.spinner("🔊 Speaking..."):
                speak_text(answer)

    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["chat"].append({"role": "user", "content": query, "time": stamp})
    st.session_state["chat"].append({"role": "assistant", "content": answer, "time": stamp})


def tab_chat():
    """Render Smart Chat tab UI and interactions."""
    st.subheader("💬 Smart Chat")
    if not st.session_state["indexed_files"]:
        render_welcome()

    for i, msg in enumerate(st.session_state["chat"]):
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>{msg['content']}<br/><small>{msg['time']}</small></div>",
                unsafe_allow_html=True,
            )
        else:
            answer_text = msg["content"]
            idx = i
            btn_col, msg_col = st.columns([1, 11])
            with btn_col:
                if st.button("🔊 Read Aloud", key=f"speak_chat_{idx}"):
                    with st.spinner("🔊 Speaking..."):
                        speak_text(answer_text)
            with msg_col:
                st.markdown(
                    f"<div class='chat-bot'>{msg['content']}<br/><small>{msg['time']}</small></div>",
                    unsafe_allow_html=True,
                )

    nonce = int(st.session_state.get("chat_input_nonce", 0))
    q_key = f"q_input_{nonce}"
    if q_key not in st.session_state:
        st.session_state[q_key] = ""

    cols = st.columns([8, 1])
    with cols[0]:
        question = st.text_input(
            "Ask anything from your uploaded documents...",
            key=q_key,
        )
    with cols[1]:
        send = st.button("Send", key="send_btn_chat")

    chips = ["📅 Exam Schedule", "💰 Fee Details", "📚 Attendance Rules", "🏢 Placements", "📖 Library Timings"]
    selected_chip = st.radio("Quick questions", chips, horizontal=True, label_visibility="collapsed")
    if st.button("Use quick question", key="use_quick_question_btn_chat"):
        question = selected_chip

    if send and question.strip():
        st.markdown(f"<div class='chat-user'>{question}</div>", unsafe_allow_html=True)
        process_query(question.strip())
        st.session_state["chat_input_nonce"] = nonce + 1


def tab_training_dashboard():
    """Render ML training dashboard for model transparency."""
    st.subheader("🧠 ML Training Dashboard")
    st.markdown("### Section A: Intent Classifier")

    if st.button("🚀 Train Intent Classifier", key="train_intent_btn"):
        data = load_training_data()
        with st.status("Training intent models...", expanded=True) as status:
            st.write("Step 1 - Load Data")
            st.write("80 training examples | 6 classes")
            trained = train_intent_models(data)
            st.session_state["intent_training"] = {"data": data, "trained": trained}
            status.update(label="Intent models trained", state="complete")

    info = st.session_state.get("intent_training")
    if info:
        data = info["data"]
        trained = info["trained"]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(class_distribution_figure(data), use_container_width=True)
        st.markdown("Step 2 - Preprocess")
        c1, c2 = st.columns(2)
        c1.code(df.iloc[0]["text"])
        c2.code(df.iloc[0]["text"].lower())
        st.caption("Lowercase -> Remove stopwords -> TF-IDF ngrams (1,2)")
        st.markdown("Step 3 - Split")
        st.write(f"Train: {len(trained['X_train'])} examples | Test: {len(trained['X_test'])} examples")
        st.markdown("Step 4/5 - Compare Models")
        st.plotly_chart(model_metrics_figure(trained["results"]), use_container_width=True)
        best = trained["best_model_name"]
        st.success(f"✅ Best Model: {best} with {trained['results'][best]['accuracy']*100:.2f}% accuracy")
        st.markdown("Step 6 - Confusion Matrix")
        st.plotly_chart(confusion_matrix_figure(trained["confusion_matrix"], trained["labels"]), use_container_width=True)
        st.markdown("Step 7 - ROC Curves")
        st.info("Multi-class ROC can be added from probability outputs; placeholder kept lightweight for exhibition flow.")
        st.markdown("Step 8 - Feature Importance")
        st.info("Top class-wise TF-IDF terms can be extracted from linear models in an extended version.")

    st.markdown("### Section B: AI Tutor PYQ Analyzer (RF feature importances)")
    tpyq = st.session_state.get("tutor_pyq_result")
    fi = []
    if tpyq and not tpyq.get("error") and tpyq.get("feature_importances") and tpyq.get("feature_names"):
        names = tpyq["feature_names"]
        vals = tpyq["feature_importances"]
        if len(names) == len(vals):
            fi = [{"feature": names[i], "importance": float(vals[i])} for i in range(len(names))]
    if fi:
        st.plotly_chart(
            px.bar(pd.DataFrame(fi), x="feature", y="importance", title="Exam topic model — RF importances"),
            use_container_width=True,
        )

    st.markdown("### Section C: Vector Space Explorer")
    records = get_retriever().fetch_all_chunks()
    if records:
        vectors = np.array([r["embedding"] for r in records if r["embedding"] is not None])
        if len(vectors) >= 5:
            pca = PCA(n_components=2, random_state=42)
            points = pca.fit_transform(vectors)
            labels = [r["filename"] for r in records[: len(points)]]
            snippets = [r["text"][:100] for r in records[: len(points)]]
            st.caption(f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
            st.plotly_chart(pca_scatter(points, labels, snippets), use_container_width=True)
            n = min(20, len(vectors))
            st.plotly_chart(similarity_heatmap(cosine_similarity(vectors[:n], vectors[:n])), use_container_width=True)
            if len(vectors) >= 3:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                klabels = km.fit_predict(vectors)
                st.write(f"Silhouette score (k=3): {silhouette_score(vectors, klabels):.3f}")

    st.markdown("### Section D: System Performance")
    total_q = st.session_state["question_count"]
    avg_conf = np.mean(st.session_state["conf_scores"]) if st.session_state["conf_scores"] else 0.0
    avg_resp = np.mean([sum(t.values()) for t in st.session_state["timings"]]) if st.session_state["timings"] else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries answered", total_q)
    c2.metric("Average response time (ms)", f"{avg_resp:.1f}")
    c3.metric("Average confidence score", f"{avg_conf:.1f}%")
    c4.metric("Cache hit rate", f"{st.session_state['cache_hits']}%")
    if st.session_state["timings"]:
        frame = pd.DataFrame(st.session_state["timings"]).mean().reset_index()
        frame.columns = ["step", "ms"]
        st.plotly_chart(px.bar(frame, x="step", y="ms", title="Pipeline Timing Breakdown"), use_container_width=True)

    st.markdown("---")
    st.markdown("## 🧠 Section E: Document Intelligence Neural Network")

    history = load_training_history()
    metrics = load_final_metrics()

    if not history or not metrics:
        st.warning(
            "Model not trained yet.\n\n"
            "Run this command first:\n\n"
            "`python train_doc_model.py`"
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("DocType Accuracy", f"{metrics['doc_type_acc']:.1f}%")
        with col2:
            st.metric("Subject Accuracy", f"{metrics['subject_acc']:.1f}%")
        with col3:
            st.metric("Difficulty Accuracy", f"{metrics['difficulty_acc']:.1f}%")
        with col4:
            st.metric("Total Parameters", f"{metrics['total_params']:,}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_loss_curves(history), use_container_width=True)
        with col2:
            st.plotly_chart(plot_accuracy_curves(history), use_container_width=True)

        st.markdown("### 🏗️ Network Architecture")
        st.markdown(plot_architecture(), unsafe_allow_html=True)

        st.markdown("### 🔍 Test the Model")
        st.markdown("Upload any PDF to classify:")
        test_pdf = st.file_uploader(
            "Upload PDF for classification",
            type=["pdf"],
            key="doc_intel_test_pdf",
        )
        if test_pdf:
            predictor = load_doc_intelligence()
            if predictor:
                with st.spinner("🧠 Classifying..."):
                    result = predictor.predict_pdf(test_pdf)
                if result:
                    st.success("Classification Complete!")

                    st.markdown("**Document Type:**")
                    for label, conf in result["doc_type"]["all_probs"]:
                        st.progress(conf / 100, text=f"{label}: {conf:.1f}%")

                    st.markdown("**Subject:**")
                    for label, conf in result["subject"]["all_probs"]:
                        st.progress(conf / 100, text=f"{label}: {conf:.1f}%")

                    st.markdown("**Difficulty:**")
                    for label, conf in result["difficulty"]["all_probs"]:
                        st.progress(conf / 100, text=f"{label}: {conf:.1f}%")


def render_splash() -> None:
    """Full-screen landing splash shown once per browser session."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        .stTabs { display: none !important; }
        header { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        footer { visibility: hidden !important; }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        .splash-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #1A1A1A;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes scaleIn {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(245, 197, 24, 0.4); }
            70% { box-shadow: 0 0 0 30px rgba(245, 197, 24, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 197, 24, 0); }
        }

        .splash-logo {
            font-size: 80px;
            margin-bottom: 24px;
            border-radius: 50%;
            padding: 10px;
            animation: scaleIn 0.5s ease-out, pulse 2s infinite 0.5s;
        }

        .splash-title {
            font-family: 'Inter', sans-serif;
            font-size: 52px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -1px;
            animation: slideUp 0.6s ease-out 0.2s both;
            margin-bottom: 12px;
        }

        .splash-title span { color: #F5C518; }

        .splash-tagline {
            font-family: 'Inter', sans-serif;
            font-size: 18px;
            font-weight: 300;
            color: #A0A0A0;
            letter-spacing: 2px;
            text-transform: uppercase;
            animation: slideUp 0.6s ease-out 0.4s both;
            margin-bottom: 60px;
        }

        .splash-loader {
            width: 200px;
            height: 3px;
            background: #2D2D2D;
            border-radius: 3px;
            overflow: hidden;
            animation: slideUp 0.6s ease-out 0.6s both;
        }

        .splash-loader-bar {
            height: 100%;
            background: linear-gradient(90deg, #F5C518, #E6B800);
            border-radius: 3px;
            animation: loading 2s ease-in-out forwards;
        }

        @keyframes loading {
            0% { width: 0%; }
            100% { width: 100%; }
        }

        .splash-powered {
            position: absolute;
            bottom: 40px;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: #3D3D3D;
            letter-spacing: 1px;
            animation: slideUp 0.6s ease-out 0.8s both;
        }

        .splash-powered span { color: #666666; }
        </style>

        <div class="splash-container">
            <div class="splash-logo">📚</div>
            <div class="splash-title">RSG<span>Sphere</span></div>
            <div class="splash-tagline">Study Smart · Not Hard</div>
            <div class="splash-loader">
                <div class="splash-loader-bar"></div>
            </div>
            <div class="splash-powered">
                Powered by <span>llama3.2 · ChromaDB · scikit-learn</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(2)
    st.session_state.show_splash = False
    st.rerun()


def render_main_app() -> None:
    """Main application: sidebar and all tabs."""
    render_sidebar()
    t1, t2, t3 = st.tabs([
        "💬  Smart Chat",
        "🤖  AI Tutor",
        "🧠  ML Dashboard",
    ])
    with t1:
        tab_chat()
    with t2:
        render_ai_tutor_tab(get_notes_context=build_tutor_notes_context)
    with t3:
        tab_training_dashboard()


def main():
    """Run splash once per session, then the full application."""
    init_state()
    apply_theme()

    if "show_splash" not in st.session_state:
        st.session_state.show_splash = True

    if st.session_state.show_splash:
        render_splash()
    else:
        render_main_app()


if __name__ == "__main__":
    main()
