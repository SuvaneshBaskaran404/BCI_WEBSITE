"""Plotly figures for the linked lab view and the trial inspector, plus selection parsing.

Okabe-Ito colours keep Left/Right distinguishable under colour-vision deficiency.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from ml_pipeline.metrics import roc_points

BLUE = "#0072B2"       # Okabe-Ito blue: Left Hand
ORANGE = "#E69F00"     # Okabe-Ito orange: Right Hand
VERMILION = "#D55E00"  # Okabe-Ito vermilion: misclassified marker outline
GREEN = "#009E73"
INK = "#0B1C30"
INDIGO = "#4F46E5"
GREY = "#777587"

_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color="#464555"),
    margin=dict(l=48, r=16, t=36, b=44),
)


def selected_trials_from_event(event):
    """Trial indices from a Streamlit plotly selection event (customdata holds trial ids)."""
    if not event:
        return []
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return []
    found = set()
    for p in points or []:
        if not isinstance(p, dict):
            continue
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd):
            cd = cd[0]
        if cd is None:
            cd = p.get("point_index")
        if cd is not None:
            found.add(int(cd))
    return sorted(found)


def projection_figure(coords, y, pred, class_names, highlight=()):
    """Plot 1: 2-D tangent-space projection with a surrogate decision-boundary map."""
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = len(y)

    surrogate = LogisticRegression(max_iter=500).fit(coords, y)
    pad = 0.6
    xs = np.linspace(coords[:, 0].min() - pad, coords[:, 0].max() + pad, 120)
    ys = np.linspace(coords[:, 1].min() - pad, coords[:, 1].max() + pad, 120)
    xx, yy = np.meshgrid(xs, ys)
    zz = surrogate.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    hl = set(int(i) for i in highlight)
    correct = pred == y
    colors = [BLUE if yi == 0 else ORANGE for yi in y]
    symbols = ["circle" if c else "x" for c in correct]
    sizes = [15 if i in hl else 9 for i in range(n)]
    line_colors = [INK if i in hl else (VERMILION if not correct[i] else "#FFFFFF") for i in range(n)]
    line_widths = [2.5 if i in hl else 1.0 for i in range(n)]
    hover = [f"Trial {i} · {class_names[int(y[i])]} · predicted {class_names[int(pred[i])]}"
             for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=zz, zmin=0, zmax=1, showscale=False, hoverinfo="skip", opacity=0.9,
        colorscale=[[0, "#CFE5F4"], [0.5, "#FFFFFF"], [1, "#FBE7C0"]],
        contours=dict(coloring="heatmap", showlines=False),
    ))
    fig.add_trace(go.Scatter(
        x=coords[:, 0].tolist(), y=coords[:, 1].tolist(), mode="markers",
        customdata=list(range(n)), hovertext=hover, hoverinfo="text",
        marker=dict(color=colors, symbol=symbols, size=sizes,
                    line=dict(color=line_colors, width=line_widths)),
    ))
    fig.update_layout(**_LAYOUT, height=380, showlegend=False, dragmode="select",
                      xaxis_title="Riemannian component ξ₁", yaxis_title="Riemannian component ξ₂")
    fig.add_annotation(
        x=0.01, y=1.08, xref="paper", yref="paper", showarrow=False, align="left",
        text=(f"<span style='color:{BLUE}'>●</span> Left Hand &nbsp; "
              f"<span style='color:{ORANGE}'>●</span> Right Hand &nbsp; "
              f"<span style='color:{VERMILION}'>✕</span> misclassified"),
    )
    return fig


def confusion_figure(cm, class_names):
    cm = np.asarray(cm)
    names = [class_names[0], class_names[1]]
    text = [[str(int(v)) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred: {n}" for n in names], y=[f"True: {n}" for n in names],
        text=text, texttemplate="%{text}",
        textfont=dict(size=22, color=INK, family="JetBrains Mono, monospace"),
        colorscale=[[0, "#FFFFFF"], [1, INDIGO]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>count = %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**_LAYOUT, height=300)
    return fig


def roc_figure(y, proba, auc):
    fpr, tpr, _ = roc_points(y, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=INDIGO, width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=GREY)))
    auc_text = "n/a" if auc != auc else f"{auc:.3f}"
    fig.update_layout(**_LAYOUT, height=300, showlegend=False,
                      title=dict(text=f"ROC · AUC = {auc_text}", x=0.02),
                      xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def waveform_figure(t, traces, names):
    colors = {"C3": BLUE, "C4": ORANGE, "Cz": GREEN, "Fz": VERMILION}
    fig = go.Figure()
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(x=t, y=traces[k], mode="lines", name=name, hoverinfo="name",
                                 line=dict(width=1.2, color=colors.get(name, GREY))))
        fig.add_annotation(x=t[0], y=traces[k][0], text=name, showarrow=False,
                           xanchor="right", xshift=-6, font=dict(size=9, color=GREY))
    layout = {**_LAYOUT, "margin": dict(l=60, r=16, t=16, b=44)}
    fig.update_layout(**layout, height=340, showlegend=False, xaxis_title="Time (s)")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


def topomap_figure(xy, values, names):
    """Electrode scalp map of per-channel values (no interpolation; positions from 10-20)."""
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    r = max(0.1, float(np.nanmax(np.abs(xy)))) * 1.15 if ok.any() else 1.0
    theta = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=(r * np.cos(theta)).tolist(),
                             y=(r * np.sin(theta)).tolist(), mode="lines",
                             line=dict(color=GREY, width=1), hoverinfo="skip"))
    idx = np.where(ok)[0]
    fig.add_trace(go.Scatter(
        x=xy[idx, 0].tolist(), y=xy[idx, 1].tolist(), mode="markers+text",
        text=[names[i] for i in idx], textposition="top center", textfont=dict(size=9),
        hovertext=[f"{names[i]}: {values[i]:.2e} V²" for i in idx], hoverinfo="text",
        marker=dict(size=18, color=values[idx].tolist(),
                    colorscale=[[0, "#EFF4FF"], [1, INDIGO]], showscale=False,
                    line=dict(color="#FFFFFF", width=1)),
    ))
    fig.update_xaxes(visible=False, range=[-r, r], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[-r, r])
    fig.update_layout(**_LAYOUT, height=260, showlegend=False,
                      title=dict(text="Mu-band (8–12 Hz) power", x=0.02, font=dict(size=12)))
    return fig
