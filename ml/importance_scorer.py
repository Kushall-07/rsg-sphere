"""Random Forest-based topic importance scoring."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
FEATURE_NAMES = ["PYQ Frequency", "Notes Mentions", "Heading/Bold Signal", "Technical Complexity", "Syllabus Position", "Co-occurrence Score"]

def _build_features(topics: List[str], pyq_stats: Dict, notes_corpus: str):
    """Create six numeric features for each candidate topic."""
    notes_lower = notes_corpus.lower(); total_topics = max(1, len(topics)); X = []; y = []
    for idx, topic in enumerate(topics):
        t = topic.lower(); pyq_ratio = pyq_stats.get(topic, {}).get("ratio", 0.0); mentions = notes_lower.count(t)
        heading = 1.0 if topic.istitle() or len(topic.split()) <= 3 else 0.0
        complexity = min(1.0, len(topic.split()) / 4 + len(topic) / 30); position = 1.0 - (idx / total_topics)
        cooccur = min(1.0, (mentions + pyq_ratio * 10) / 12)
        feats = [pyq_ratio, mentions / 10.0, heading, complexity, position, cooccur]
        X.append(feats); y.append(0.5 * pyq_ratio + 0.2 * min(1.0, mentions / 8) + 0.3 * cooccur)
    return np.array(X), np.array(y)

def train_and_score_topics(topics: List[str], pyq_result: Dict, notes_corpus: str) -> Dict:
    """Train Random Forest regressor and produce ranked topic scores."""
    X, y = _build_features(topics, pyq_result.get("frequency", {}), notes_corpus)
    if len(topics) < 3:
        raise ValueError("Need at least 3 topics to train predictor.")
    model = RandomForestRegressor(n_estimators=100, random_state=42); model.fit(X, y); preds = model.predict(X)
    norm = 100 * (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
    ranked = []
    for i, topic in enumerate(topics):
        score = float(norm[i]); priority = "HIGH" if score > 70 else ("MEDIUM" if score >= 40 else "LOW")
        f = pyq_result.get("frequency", {}).get(topic, {"appeared_count": 0, "total_papers": 1})
        ranked.append({"topic": topic, "score": round(score, 2), "priority": priority, "appeared": f"{f['appeared_count']}/{f['total_papers']}"})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {
        "model": model,
        "ranked_topics": ranked,
        "feature_importances": dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
        "metrics": {"r2": float(r2_score(y, preds)), "mae": float(mean_absolute_error(y, preds)), "rmse": float(mean_squared_error(y, preds) ** 0.5)},
        "features": X,
        "targets": y,
        "predictions": preds,
    }
