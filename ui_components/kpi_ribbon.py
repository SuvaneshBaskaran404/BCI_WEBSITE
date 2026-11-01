"""Four-card KPI ribbon: CV accuracy, Cohen's kappa, ROC-AUC and runtime."""
import streamlit as st

from ml_pipeline.metrics import kappa_label
from ui_components.theme_styles import kpi_card_html


def render_kpis(m, k_folds, runtime_ms, chance=0.5):
    cols = st.columns(4)
    acc = m["cv_acc"]
    auc = m["roc_auc"]
    cards = [
        (f"{k_folds}-fold CV accuracy (τ 0.50)", f"{acc:.1%}",
         f"{(acc - chance) * 100:+.1f} pts vs chance · σ {m['cv_std']:.3f}",
         "ok" if acc > chance else "err"),
        ("Cohen's κ", f"{m['kappa']:.3f}", kappa_label(m["kappa"]),
         "ok" if m["kappa"] >= 0.4 else "err"),
        ("ROC-AUC", "n/a" if auc != auc else f"{auc:.3f}",
         "Threshold-independent", "info"),
        ("Last CV runtime", f"{runtime_ms:,.0f} ms", "Cached per parameter set", "info"),
    ]
    for col, (label, value, sub, tone) in zip(cols, cards):
        with col:
            st.markdown(kpi_card_html(label, value, sub, tone), unsafe_allow_html=True)
