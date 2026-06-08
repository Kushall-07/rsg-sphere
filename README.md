# RAG-Sphere 📚

**Study Smart. Not Hard.**

RAG-Sphere is an intelligent, offline AI study companion that combines Retrieval-Augmented Generation (RAG), document intelligence, and an AI tutor to help you learn from your own study materials — completely private, zero-cost, and running locally on your machine.

![RAG-Sphere Screenshot](https://via.placeholder.com/800x400?feat=RAG-Sphere+Dashboard) <!-- Replace with actual screenshot if available -->

## ✨ Features

### 💬 Smart Chat Tab
- **Upload PDFs** (syllabus, notes, textbooks, lab manuals, previous year papers)
- **Hybrid Retrieval**: Combines semantic search (vector embeddings) + BM25 keyword search for accurate results
- **ML Reranking**: Uses a trained SVM to re-rank retrieved chunks for better relevance
- **Streaming Answers**: Powered by Ollama (`llama3.2`) for real-time, token-by-token responses
- **Intent Detection**: Automatically classifies your query (fees, exam, placement, hostel, library, general) to tailor responses
- **Source Citations**: Every answer includes the filename and page number so you can verify the source
- **Confidence Scores**: See how confident the model is in its answer
- **Chat History**: Export your entire conversation as a PDF for later review

### 🎓 AI Tutor Tab
- **Previous Year Paper (PYQ) Analyzer**: Upload past exam papers to extract key topics and predict likely questions
- **Smart Topic Extraction**: Uses TF-IDF + technical pattern recognition to identify important concepts
- **Question Generation**: Generates exam-style questions based on your uploaded materials
- **Document Intelligence**: Auto-classifies uploaded PYQs by type (textbook, question paper, notes, etc.), subject, and difficulty level
- **Contextual Tutoring**: Get explanations, solutions, and study guidance tailored to your materials
- **Session Management**: Maintain multiple chat sessions with persistence across browser sessions

### 🧠 ML Dashboard Tab
- **Document Intelligence Metrics**:
  - View training history (loss/accuracy curves over 30 epochs)
  - Confusion matrices for document type, subject, and difficulty classification
  - Feature correlation heatmap (12 handcrafted linguistic features)
  - Model architecture visualization
- **Intent Classifier Metrics**:
  - Compare Naive Bayes, SVM, and Logistic Regression performance
  - ROC curves and AUC scores for each intent class
  - Confusion matrix and feature importance (TF-IDF weights)
- **System Performance**:
  - Real-time statistics: total queries, average response time, confidence scores
  - Pipeline timing breakdown (loading, chunking, embedding, retrieval, reranking, generation)
  - Vector space exploration: PCA projections, similarity heatmaps, clustering analysis

## 🛠️ Tech Stack

| Component          | Technology                                                                 |
|--------------------|----------------------------------------------------------------------------|
| **Frontend**       | Streamlit with custom CSS (dark theme, interactive UI)                     |
| **Embeddings**     | SentenceTransformers (`all-MiniLM-L6-v2`)                                  |
| **Vector DB**      | ChromaDB (local persistence)                                               |
| **LLM**            | Ollama (`llama3.2` 3B parameter model)                                     |
| **Document Intelligence** | PyTorch CNN (3-task classification)                                   |
| **Intent Classification** | scikit-learn (SVM, Logistic Regression, Naive Bayes)               |
| **NLP Utilities**  | rank-bm25, NLTK, PyPDF2, textstat                                          |
| **Visualization**  | Plotly (interactive charts)                                                |
| **Export**         | ReportLab (PDF export)                                                     |
| **Voice I/O**      | SpeechRecognition, Pyttsx3 (optional voice input/output)                   |

## 🚀 Getting Started

### Prerequisites
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/) (for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kushall-07/rsg-sphere.git
   cd rsg-sphere
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the LLM model**
   ```bash
   ollama pull llama3.2
   ```

4. **Start Ollama service**
   ```bash
   ollama serve
   ```
   *(Leave this running in a separate terminal)*

5. **Train the models (if not already trained)**
   ```bash
   # Train document intelligence model
   python train_doc_model.py
   
   # Intent classifier is trained on first use, or you can trigger it from the ML Dashboard tab
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
rsg-sphere/
├── app.py                  # Main Streamlit application (3 tabs)
├── requirements.txt        # Python dependencies
├── train_doc_model.py      # Script to train the document intelligence model
├── models/                 # Trained ML models
│   ├── doc_intelligence.pth
│   ├── feature_extractor.pkl
│   └── intent_clf.pkl
├── data/                   # Training data and user uploads
├── chroma_db/              # Vector database (ChromaDB persistence)
├── doc_intelligence/       # Document classification (type/subject/difficulty)
├── ml/                     # Intent classifier & topic extraction
├── rag/                    # RAG pipeline (embedder, retriever, reranker, generator)
├── doubt_solver/           # AI Tutor tab (PYQ analysis, chat)
└── utils/                  # Confidence scoring, PDF export, voice I/O
```

## 🎯 How It Works

1. **Document Processing**
   - When you upload a PDF, the system extracts text using PyPDF2
   - Text is chunked into semantically meaningful pieces
   - Each chunk is embedded into a 384-dimensional vector using SentenceTransformers
   - Vectors are stored in ChromaDB for efficient similarity search

2. **Query Processing**
   - Your question is embedded using the same model
   - Hybrid search retrieves relevant chunks (vector + BM25)
   - An SVM reranker re-orders results based on learned relevance patterns
   - The top chunks are combined with an intent-aware prompt
   - The prompt is sent to Ollama's `llama3.2` model for generation
   - Responses are streamed back in real-time

3. **Document Intelligence**
   - Uploaded PDFs are analyzed using a CNN that predicts:
     - **Document Type**: Textbook, Question Paper, Notes, Lab Manual, Research Paper, Syllabus
     - **Subject**: Mathematics, Physics, Chemistry, CS, Biology, English, History, Economics, etc.
     - **Difficulty**: Beginner, Intermediate, Advanced
   - These predictions help contextualize both the Smart Chat and AI Tutor responses

4. **AI Tutor & PYQ Analysis**
   - Previous year papers are processed to extract:
     - Key technical topics using TF-IDF and domain-specific patterns
     - Likely exam questions based on question patterns and Bloom's taxonomy
   - The analyzer uses a Random Forest model to rank topic importance
   - Generated questions and topic explanations are grounded in your actual materials

## 🔒 Privacy & Offline-First Design

- **100% Local**: All processing happens on your machine
- **No Data Leaves**: Neither your documents nor your queries are sent to any external server
- **No Account Needed**: Zero registration, zero tracking
- **Model Control**: You choose which LLM to run via Ollama (we recommend `llama3.2`)
- **Vector DB Persistence**: ChromaDB stores embeddings locally in `./chroma_db/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the incredible frontend framework
- [Ollama](https://ollama.com/) for making local LLMs accessible
- [SentenceTransformers](https://www.sbert.net/) for state-of-the-art embeddings
- [ChromaDB](https://www.trychroma.com/) for the vector database
- The open-source ML community for scikit-learn, PyTorch, and NLTK

---

**Made with ❤️ for students everywhere.**  
Transform your study materials into an interactive learning experience — no internet required after initial setup.