"""Plotly visualizations used across the app dashboards."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def class_distribution_figure(data):
    """Build class-distribution bar chart for intent dataset."""
    df = pd.DataFrame(data); counts = df["intent"].value_counts().reset_index(); counts.columns = ["intent", "count"]
    return px.bar(counts, x="intent", y="count", color="intent", title="Class Distribution")

def model_metrics_figure(results):
    """Build grouped bar chart comparing model classification metrics."""
    rows = []
    for m, v in results.items():
        rows += [{"Model": m, "Metric": "Accuracy", "Score": v["accuracy"]}, {"Model": m, "Metric": "Precision", "Score": v["precision"]}, {"Model": m, "Metric": "Recall", "Score": v["recall"]}, {"Model": m, "Metric": "F1", "Score": v["f1"]}]
    return px.bar(pd.DataFrame(rows), x="Model", y="Score", color="Metric", barmode="group", title="Model Metrics")

def confusion_matrix_figure(matrix, labels):
    """Build confusion-matrix heatmap for best intent classifier."""
    return go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Viridis"), layout=go.Layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"))

def topic_bar_figure(ranked_topics):
    """Create horizontal ranking chart for predicted exam topics."""
    df = pd.DataFrame(ranked_topics); return px.bar(df.sort_values("score"), x="score", y="topic", color="priority", orientation="h", title="Topic Importance Ranking")

def topic_pie_figure(ranked_topics):
    """Create priority distribution pie chart for topic classes."""
    df = pd.DataFrame(ranked_topics); dist = df["priority"].value_counts().reset_index(); dist.columns = ["priority", "count"]
    return px.pie(dist, names="priority", values="count", title="Priority Distribution")

def feature_importance_figure(feature_importances):
    """Create feature importance chart for random forest predictor."""
    df = pd.DataFrame({"feature": list(feature_importances.keys()), "importance": list(feature_importances.values())})
    return px.bar(df, x="importance", y="feature", orientation="h", title="Feature Importances")

def heatmap_year_topic(year_rows):
    """Build year-vs-topic appearance heatmap for PYQ analysis."""
    if not year_rows:
        return go.Figure(layout=go.Layout(title="Year-wise Topic Heatmap"))
    df = pd.DataFrame(year_rows); years = df["year"].tolist(); cols = [c for c in df.columns if c != "year"]
    return go.Figure(data=go.Heatmap(z=df[cols].values, x=cols, y=years, colorscale="Blues"), layout=go.Layout(title="Year-wise Topic Heatmap"))

def pca_scatter(points, labels, snippets, title="Embedding Space (PCA)"):
    """Create embedding scatter plot for vector-space explorer panels."""
    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "label": labels, "snippet": snippets})
    return px.scatter(df, x="x", y="y", color="label", hover_data=["snippet"], title=title)

def similarity_heatmap(sim_matrix):
    """Create cosine similarity heatmap for top chunk comparisons."""
    return go.Figure(data=go.Heatmap(z=sim_matrix, colorscale="Viridis"), layout=go.Layout(title="Similarity Heatmap"))
