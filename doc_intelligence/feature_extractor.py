"""TF-IDF and handcrafted features for document classification."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class DocumentFeatureExtractor:
    def __init__(self, max_features: int = 1000):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=2,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _handcrafted_features(self, texts):
        features = []
        for text in texts:
            words = text.lower().split()
            sentences = [s for s in text.split(".") if s.strip()]

            qp_words = ["marks", "answer", "attempt", "module", "part"]
            tb_words = ["chapter", "definition", "theorem", "exercise"]
            lab_words = ["aim", "procedure", "apparatus", "observation"]
            rp_words = ["abstract", "methodology", "references", "proposed"]

            feat = [
                len(words),
                len(sentences),
                np.mean([len(w) for w in words]) if words else 0,
                np.mean([len(s.split()) for s in sentences]) if sentences else 0,
                text.count("?"),
                sum(1 for w in words if w.isdigit()),
                sum(1 for w in words if w in qp_words),
                sum(1 for w in words if w in tb_words),
                sum(1 for w in words if w in lab_words),
                sum(1 for w in words if w in rp_words),
                sum(1 for w in words if len(w) > 10),
                sum(1 for w in text.split() if w.isupper()) / max(len(words), 1),
            ]
            features.append(feat)
        return np.array(features)

    def fit(self, texts):
        self.tfidf.fit(texts)
        hc = self._handcrafted_features(texts)
        self.scaler.fit(hc)
        self.is_fitted = True

    def transform(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform.")
        tfidf_feat = self.tfidf.transform(texts).toarray()
        hc_feat = self._handcrafted_features(texts)
        hc_scaled = self.scaler.transform(hc_feat)
        return np.hstack([tfidf_feat, hc_scaled])

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def extract_from_pdf(self, pdf_file):
        import PyPDF2

        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        if not text.strip():
            text = "unknown document"
        features = self.transform([text])
        return features, text

    def save(self, path: str = "models/feature_extractor.pkl"):
        import joblib

        joblib.dump(
            {
                "tfidf": self.tfidf,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str = "models/feature_extractor.pkl"):
        import joblib

        data = joblib.load(path)
        self.tfidf = data["tfidf"]
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]
