"""EEG Classification Explorer: Streamlit entry point.

Launch with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from mne_skeleton.data_loader import load_moabb_bnci2014_001, load_uploaded, make_synthetic_mi
from mne_skeleton.epoch_inspector import band_power, electrode_positions, waveform_offsets
from mne_skeleton.preprocessing import preprocess
from ml_pipeline.classifier import MDM_NAME, build_pipeline, normalize_params
from ml_pipeline.features import tangent_space_2d
from ml_pipeline.metrics import (QUADRANT_LABELS, binary_metrics, diagnose_generalization,
                                 explain_trial, quadrant_indices, threshold_predict)
from ml_pipeline.validation import cross_validate_model
from ui_components.explain_quiz import render_concepts, render_diagnostics, render_quiz
from ui_components.exporter import build_config_json, build_pipeline_script
from ui_components.kpi_ribbon import render_kpis
from ui_components.linked_plots import (confusion_figure, projection_figure, roc_figure,
                                        selected_trials_from_event, topomap_figure,
                                        waveform_figure)
from ui_components.sidebar import SOURCE_MOABB, SOURCE_SYNTH, SOURCE_UPLOAD, render_sidebar
from ui_components.theme_styles import badge_html, inject_css, section_header

st.set_page_config(page_title="EEG Classification Explorer", page_icon="🧠", layout="wide")
inject_css()

params, upload = render_sidebar()
params = normalize_params(params)


@st.cache_data(show_spinner="Loading EEG data…")
def load_dataset(source, file_bytes=None, file_name=None):
    if source == SOURCE_MOABB:
        return load_moabb_bnci2014_001(subject=1)
    if source == SOURCE_UPLOAD:
        return load_uploaded(file_bytes, file_name)
    return make_synthetic_mi()


@st.cache_data(show_spinner=False)
def preprocess_cached(X, sfreq, l_freq, h_freq, reject_uv):
    return preprocess(X, sfreq, l_freq, h_freq, reject_uv=reject_uv)


@st.cache_data(show_spinner=False)
def projection_cached(X):
    return tangent_space_2d(X)


@st.cache_data(show_spinner="Training & cross-validating…")
def run_experiment(X, y, feature, classifier, shrinkage, C, kernel, metric, n_folds, stratified):
    import time

    t0 = time.perf_counter()

    def build():
        return build_pipeline(feature, classifier, C=C, shrinkage=shrinkage,
                              kernel=kernel, metric=metric)

    cv = cross_validate_model(X, y, build, n_folds=n_folds, stratified=stratified)
    cv["runtime_ms"] = (time.perf_counter() - t0) * 1000
    return cv


# ---------- data ----------
if params["source"] == SOURCE_UPLOAD and upload is None:
    st.info("Upload a .fif (MNE Epochs), .csv or .npy file in the sidebar, or choose a sample dataset.")
    st.stop()

file_bytes = upload.getvalue() if upload is not None else None
file_name = upload.name if upload is not None else None
try:
    dataset = load_dataset(params["source"], file_bytes, file_name)
except Exception as exc:  # noqa: BLE001 - show any loader problem in the UI
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

Xf, artifact_ok = preprocess_cached(dataset.X, dataset.sfreq, *params["band"], params["reject_uv"])

# ---------- header ----------
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown(section_header("Module 8 · Advanced EEG Classification"), unsafe_allow_html=True)
    st.markdown("<p class='hero-title'>EEG Classification Explorer</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='hero-sub'>Riemannian tangent space, CSP and MDM decoders · "
                f"{dataset.source} · {len(dataset.y)} trials · {len(dataset.ch_names)} channels"
                f" · {dataset.sfreq:.0f} Hz</p>", unsafe_allow_html=True)
with head_r:
    recompute = st.button("Re-compute pipeline", type="primary", use_container_width=True)
if recompute:
    run_experiment.clear()

# ---------- experiment ----------
try:
    cv = run_experiment(
        Xf, dataset.y, feature=params["feature"], classifier=params["classifier"],
        shrinkage=params["shrinkage"], C=params["C"], kernel=params["kernel"],
        metric=params["metric"], n_folds=params["n_folds"], stratified=params["stratified"],
    )
except ValueError as exc:
    st.error(f"Cross-validation could not run: {exc}. Try fewer folds or more trials.")
    st.stop()

thr = params["threshold"]
pred = threshold_predict(cv["oof_proba"], thr)
m = binary_metrics(dataset.y, pred, cv["oof_proba"])
m.update(cv_acc=cv["cv_acc"], cv_std=cv["cv_std"], train_acc=cv["train_acc"])
diag = diagnose_generalization(cv["train_acc"], cv["cv_acc"])
st.markdown(badge_html(f"Pipeline ready · {cv['runtime_ms']:.0f} ms", "ok"), unsafe_allow_html=True)
render_kpis(m, params["n_folds"], cv["runtime_ms"])

# ---------- tabs ----------
tab_lab, tab_theory, tab_export = st.tabs(
    ["Interactive Lab & Projections", "Theory & Concept Self-Check", "Export Pipeline Code"])

with tab_lab:
    # Selections: points clicked in Plot 1 win; otherwise a confusion quadrant (chosen below).
    quad_name = st.session_state.get("quad_select", "None")
    quad_idx = quadrant_indices(dataset.y, pred, QUADRANT_LABELS.get(quad_name))
    point_sel = selected_trials_from_event(st.session_state.get("proj_plot"))
    selected = point_sel if point_sel else [int(i) for i in quad_idx]

    row1_l, row1_r = st.columns([7, 5], gap="large")
    with row1_l:
        st.markdown(section_header("Plot 1 · Tangent-space projection & decision boundary",
                                   "Drag to select trials; selections load the inspector below"),
                    unsafe_allow_html=True)
        coords = projection_cached(Xf)
        st.plotly_chart(projection_figure(coords, dataset.y, pred, dataset.class_names, selected),
                        on_select="rerun", key="proj_plot")
    with row1_r:
        st.markdown(section_header("Plot 2 · Classification performance",
                                   "Confusion matrix and ROC, toggle below"), unsafe_allow_html=True)
        view = st.radio("View", ["Confusion matrix", "ROC curve"], horizontal=True,
                        key="view_mode", label_visibility="collapsed")
        st.radio("Highlight a quadrant in Plot 1", list(QUADRANT_LABELS), horizontal=True,
                 key="quad_select")
        if view == "ROC curve":
            st.plotly_chart(roc_figure(dataset.y, cv["oof_proba"], m["roc_auc"]), key="roc_plot")
        else:
            st.plotly_chart(confusion_figure(m["confusion"], dataset.class_names), key="cm_plot")
        s1, s2, s3 = st.columns(3)
        s1.metric("Sensitivity", f"{m['sensitivity']:.1%}")
        s2.metric("Specificity", f"{m['specificity']:.1%}")
        s3.metric("F1", f"{m['f1']:.3f}")

    row2_l, row2_r = st.columns([7, 5], gap="large")
    misclassified = [int(i) for i in (pred != dataset.y).nonzero()[0]]
    focus = selected[0] if selected else (misclassified[0] if misclassified else 0)
    with row2_l:
        trial = Xf[focus]
        correct = bool(pred[focus] == dataset.y[focus])
        st.markdown(section_header(f"Trial #{focus} inspector",
                                   f"True: {dataset.class_names[int(dataset.y[focus])]} · "
                                   f"P(Right) = {cv['oof_proba'][focus]:.2f} · "
                                   f"{'correct' if correct else 'misclassified'}"),
                    unsafe_allow_html=True)
        wf_col, tp_col = st.columns([3, 2])
        with wf_col:
            t, traces = waveform_offsets(trial, dataset.sfreq, dataset.tmin)
            st.plotly_chart(waveform_figure(t, traces, dataset.ch_names), key="wave")
        with tp_col:
            xy = electrode_positions(dataset.ch_names)
            vals = band_power(trial, dataset.sfreq, 8.0, 12.0)
            st.plotly_chart(topomap_figure(xy, vals, dataset.ch_names), key="topo")
        st.markdown(f"<div class='insight'><b>Root cause:</b> "
                    f"{explain_trial(trial, dataset.ch_names, correct, params['reject_uv'] * 1e-6)}"
                    "</div>", unsafe_allow_html=True)
    with row2_r:
        render_diagnostics(diag, m, thr)
        st.caption(f"Model: {params['classifier']} · Feature: {params['feature']} · "
                   f"Split: {'stratified' if params['stratified'] else 'chronological'} · "
                   f"Artifact-flagged trials: {int((~artifact_ok).sum())}")

with tab_theory:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(section_header("Concept cards"), unsafe_allow_html=True)
        render_concepts()
    with right:
        st.markdown(section_header("Self-check quiz", "Three questions, instant grading"),
                    unsafe_allow_html=True)
        render_quiz()

with tab_export:
    script = build_pipeline_script(params)
    config = build_config_json(params)
    st.markdown(section_header("Generated pipeline",
                               "Updates live with your active hyperparameters"),
                unsafe_allow_html=True)
    st.code(script, language="python")
    d1, d2 = st.columns(2)
    d1.download_button("Download pipeline_export.py", script, file_name="pipeline_export.py",
                       mime="text/x-python", use_container_width=True)
    d2.download_button("Download config_export.json", config, file_name="config_export.json",
                       mime="application/json", use_container_width=True)
