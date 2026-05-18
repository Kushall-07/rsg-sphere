"""Training history charts and architecture diagram for the ML dashboard."""

import plotly.graph_objects as go


def load_training_history():
    import json

    try:
        with open("models/training_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_final_metrics():
    import json

    try:
        with open("models/final_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def plot_loss_curves(history):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["train_loss"],
            name="Training Loss",
            line=dict(color="#F5C518", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["val_loss"],
            name="Validation Loss",
            line=dict(color="#E64833", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#2D2D2D",
        font_color="#FFFFFF",
        legend=dict(bgcolor="#2D2D2D", bordercolor="#3D3D3D"),
    )
    return fig


def plot_accuracy_curves(history):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["doc_type_acc"],
            name="Document Type",
            line=dict(color="#F5C518", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["subject_acc"],
            name="Subject",
            line=dict(color="#90AEAD", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["difficulty_acc"],
            name="Difficulty",
            line=dict(color="#E64833", width=2),
        )
    )
    fig.update_layout(
        title="Accuracy per Output Head",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#2D2D2D",
        font_color="#FFFFFF",
        legend=dict(bgcolor="#2D2D2D", bordercolor="#3D3D3D"),
    )
    return fig


_ARCHITECTURE_HTML = (
    "<div style='background:#2D2D2D;"
    "border-radius:12px;padding:24px;"
    "text-align:center;font-family:"
    "Inter,sans-serif;color:#FFFFFF;'>"
    "<h3 style='color:#F5C518;'>"
    "Neural Network Architecture</h3>"
    "<p>Input(1012) → Dense(512) → "
    "Dense(256) → Dense(128)</p>"
    "<div style='display:flex;"
    "justify-content:center;gap:16px;"
    "margin-top:16px;'>"
    "<div style='background:#1A1A1A;"
    "padding:10px;border-radius:8px;"
    "border:1px solid #F5C518;'>"
    "<b style='color:#F5C518;'>Head 1</b>"
    "<br>Dense(64)→Dense(6)"
    "<br><small>Document Type</small>"
    "</div>"
    "<div style='background:#1A1A1A;"
    "padding:10px;border-radius:8px;"
    "border:1px solid #90AEAD;'>"
    "<b style='color:#90AEAD;'>Head 2</b>"
    "<br>Dense(64)→Dense(8)"
    "<br><small>Subject</small>"
    "</div>"
    "<div style='background:#1A1A1A;"
    "padding:10px;border-radius:8px;"
    "border:1px solid #E64833;'>"
    "<b style='color:#E64833;'>Head 3</b>"
    "<br>Dense(32)→Dense(3)"
    "<br><small>Difficulty</small>"
    "</div>"
    "</div>"
    "</div>"
)


def plot_architecture():
    return _ARCHITECTURE_HTML
