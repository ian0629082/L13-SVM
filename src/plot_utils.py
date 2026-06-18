from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CLASS_COLORS = {
    0: "#2563eb",
    1: "#dc2626",
}


def make_mesh(X: np.ndarray, resolution: int = 100, padding: float = 0.8):
    # 建立規則網格，用來計算每個 2D 位置的分類分數。
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_2d_decision_boundary(model, X: np.ndarray, y: np.ndarray, resolution: int):
    xx, yy, grid = make_mesh(X, resolution=resolution)
    scores = model.decision_function(grid).reshape(xx.shape)
    predictions = model.predict(grid).reshape(xx.shape)
    support_vectors = model.support_vectors_

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=predictions,
            colorscale=[
                [0.0, "rgba(37, 99, 235, 0.12)"],
                [0.5, "rgba(37, 99, 235, 0.12)"],
                [0.5, "rgba(220, 38, 38, 0.12)"],
                [1.0, "rgba(220, 38, 38, 0.12)"],
            ],
            showscale=False,
            line_width=0,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=scores,
            contours=dict(
                start=-1,
                end=1,
                size=1,
                coloring="none",
                showlabels=True,
            ),
            line=dict(color="#111827", width=2),
            showscale=False,
            name="Margin and boundary",
        )
    )

    for class_id in np.unique(y):
        mask = y == class_id
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(
                    color=CLASS_COLORS[int(class_id)],
                    size=8,
                    line=dict(color="white", width=1),
                ),
                name=f"Class {class_id}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode="markers",
            marker=dict(
                color="rgba(0,0,0,0)",
                size=14,
                line=dict(color="#111827", width=2),
            ),
            name="Support vectors",
        )
    )
    fig.update_layout(
        height=650,
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title_text="",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_3d_decision_function(model, X: np.ndarray, y: np.ndarray, resolution: int):
    xx, yy, grid = make_mesh(X, resolution=resolution)
    scores = model.decision_function(grid).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=scores,
            colorscale="RdBu",
            opacity=0.78,
            colorbar=dict(title="score"),
            name="decision_function",
        )
    )

    sample_scores = model.decision_function(X)
    for class_id in np.unique(y):
        mask = y == class_id
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=sample_scores[mask],
                mode="markers",
                marker=dict(
                    color=CLASS_COLORS[int(class_id)],
                    size=4,
                    line=dict(color="white", width=0.5),
                ),
                name=f"Class {class_id}",
            )
        )

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=np.zeros_like(scores),
            showscale=False,
            opacity=0.25,
            colorscale=[[0, "#111827"], [1, "#111827"]],
            name="score = 0",
        )
    )
    fig.update_layout(
        height=720,
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="decision_function(x)",
        ),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig


def plot_simplified_kernel_mapping(
    original_X: np.ndarray,
    mapped_X: np.ndarray,
    y: np.ndarray,
    ):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Original 2D input", "Simplified 3D teaching map"),
    )

    for class_id in np.unique(y):
        mask = y == class_id
        fig.add_trace(
            go.Scatter(
                x=original_X[mask, 0],
                y=original_X[mask, 1],
                mode="markers",
                marker=dict(
                    color=CLASS_COLORS[int(class_id)],
                    size=7,
                    line=dict(color="white", width=1),
                ),
                name=f"Class {class_id}",
                legendgroup=f"class-{class_id}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=mapped_X[mask, 0],
                y=mapped_X[mask, 1],
                z=mapped_X[mask, 2],
                mode="markers",
                marker=dict(
                    color=CLASS_COLORS[int(class_id)],
                    size=4,
                    line=dict(color="white", width=0.5),
                ),
                name=f"Class {class_id} mapped",
                legendgroup=f"class-{class_id}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    x_plane = np.linspace(original_X[:, 0].min() - 0.4, original_X[:, 0].max() + 0.4, 2)
    y_plane = np.linspace(original_X[:, 1].min() - 0.4, original_X[:, 1].max() + 0.4, 2)
    px, py = np.meshgrid(x_plane, y_plane)
    pz = np.full_like(px, np.median(mapped_X[:, 2]))
    fig.add_trace(
        go.Surface(
            x=px,
            y=py,
            z=pz,
            opacity=0.22,
            showscale=False,
            colorscale=[[0, "#16a34a"], [1, "#16a34a"]],
            name="Concept plane",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="x1", row=1, col=1)
    fig.update_yaxes(title_text="x2", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(
        height=680,
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="teaching z",
        ),
    )
    return fig
