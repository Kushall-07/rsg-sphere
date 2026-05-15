"""Embedding helpers for RSGSphere RAG."""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """Manage embedding model load and inference."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with the model name and lazy model state."""
        self.model_name = model_name
        self.model = None
    def load_model(self):
        """Load sentence-transformer model once and reuse it."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model
    def embed_texts(self, texts):
        """Embed a list of texts into dense numpy vectors."""
        model = self.load_model()
        return np.array(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
    def embed_query(self, query: str):
        """Embed a single user query into one dense vector."""
        model = self.load_model()
        return np.array(model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0])
