"""ChromaDB retrieval layer for semantic search."""
from __future__ import annotations
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings

class ChromaRetriever:
    """Wrap ChromaDB operations for indexing and semantic retrieval."""
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "rag_chunks"):
        """Create or connect to persistent ChromaDB collection."""
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(collection_name)
    def index_documents(self, chunks: List[Dict], embeddings):
        """Store chunk text, metadata, and embeddings in ChromaDB."""
        if not chunks:
            return
        self.collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"filename": c["filename"], "page_number": c["page_number"]} for c in chunks],
            embeddings=embeddings.tolist(),
        )
    def semantic_search(self, query_embedding, k: int = 10, filenames: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve top-k semantically similar chunks with similarity scores."""
        query_kwargs = {"query_embeddings": [query_embedding.tolist()], "n_results": k}
        if filenames is not None:
            if len(filenames) == 0:
                return []
            elif len(filenames) == 1:
                query_kwargs["where"] = {"filename": filenames[0]}
            else:
                query_kwargs["where"] = {"filename": {"$in": list(filenames)}}
        result = self.collection.query(**query_kwargs)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out = []
        for i, doc in enumerate(docs):
            out.append({
                "text": doc,
                "filename": metas[i].get("filename", "unknown.pdf") if i < len(metas) else "unknown.pdf",
                "page_number": metas[i].get("page_number", 0) if i < len(metas) else 0,
                "semantic_score": max(0.0, 1.0 - float(distances[i])) if i < len(distances) else 0.0,
            })
        return out
    def fetch_all_chunks(self) -> List[Dict]:
        """Return all indexed chunks for dashboard exploration views."""
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        docs, metas, embeds = data.get("documents", []), data.get("metadatas", []), data.get("embeddings", [])
        return [{"text": docs[i], "filename": metas[i].get("filename", "unknown.pdf"), "page_number": metas[i].get("page_number", 0), "embedding": embeds[i]} for i in range(len(docs))]
