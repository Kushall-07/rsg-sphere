"""Intent classifier training and inference utilities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
INTENT_MODEL_PATH = Path("models/intent_clf.pkl")

def load_training_data(path: str = "data/training_data.json"):
    """Load labeled training examples for intent classification."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pipeline(model):
    """Create shared TF-IDF + classifier pipeline configuration."""
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")), ("clf", model)])

def train_intent_models(data) -> Dict:
    """Train NB, SVM, LR and return metrics, artifacts, and best model."""
    X = [d["text"] for d in data]; y = [d["intent"] for d in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {"Naive Bayes": _pipeline(MultinomialNB()), "SVM": _pipeline(SVC(kernel="rbf", probability=True, random_state=42)), "Logistic Regression": _pipeline(LogisticRegression(max_iter=1000, random_state=42))}
    results = {}; best_name = None; best_acc = -1; best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train); preds = model.predict(X_test); acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        results[name] = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "preds": preds, "model": model}
        if acc > best_acc:
            best_acc = acc; best_name = name; best_model = model
    INTENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True); joblib.dump(best_model, INTENT_MODEL_PATH)
    labels = sorted(list(set(y))); conf = confusion_matrix(y_test, results[best_name]["preds"], labels=labels)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "results": results, "best_model_name": best_name, "best_model": best_model, "labels": labels, "confusion_matrix": conf}

def load_or_train_intent_model() -> Pipeline:
    """Load saved intent model or train it automatically on first use."""
    return joblib.load(INTENT_MODEL_PATH) if INTENT_MODEL_PATH.exists() else train_intent_models(load_training_data())["best_model"]

def predict_intent(query: str):
    """Predict intent label and confidence score for a user query."""
    model = load_or_train_intent_model(); pred = model.predict([query])[0]; probs = model.predict_proba([query])[0]
    return pred, float(np.max(probs)), {c: float(p) for c, p in zip(model.classes_, probs)}
