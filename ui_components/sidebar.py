"""Left sidebar: dataset source, feature and classifier controls, validation settings."""
import streamlit as st

from ml_pipeline.classifier import (CLASSIFIER_OPTIONS, KERNEL_OPTIONS, LDA, MDM_METRIC_OPTIONS,
                                    MDM_NAME, RF, SVM)
from ml_pipeline.features import FEATURE_OPTIONS

SOURCE_SYNTH = "synthetic"
SOURCE_MOABB = "moabb"
SOURCE_UPLOAD = "upload"
SOURCE_LABELS = {
    SOURCE_SYNTH: "Synthetic motor imagery (offline)",
    SOURCE_MOABB: "MOABB BNCI2014-001, subject 1 (downloads once)",
    SOURCE_UPLOAD: "Upload my own file",
}


def render_sidebar():
    """Draw the sidebar and return (params dict, uploaded file or None)."""
    with st.sidebar:
        st.markdown("### EEG Workbench")
        st.caption("Module 8 · Classification")

        st.markdown("#### Dataset")
        source = st.radio("Source", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get,
                          label_visibility="collapsed")
        upload = None
        if source == SOURCE_UPLOAD:
            upload = st.file_uploader("Epochs (.fif), table (.csv) or array (.npy)",
                                      type=["fif", "csv", "npy"])

        st.markdown("#### Feature extraction")
        feature = st.selectbox("Feature pipeline", FEATURE_OPTIONS)
        band = st.slider("Bandpass (Hz)", 1.0, 40.0, (8.0, 30.0), step=0.5)
        reject_uv = st.slider("Artifact flag threshold (µV)", 50, 300, 150, step=10)

        st.markdown("#### Classifier")
        classifier = st.selectbox("Model", CLASSIFIER_OPTIONS)
        shrinkage = st.slider("Shrinkage λ (LDA)", 0.0, 1.0, 0.10, 0.05,
                              disabled=classifier != LDA)
        C = st.select_slider("Penalty C (SVM)", options=[0.01, 0.1, 1.0, 10.0, 100.0],
                             value=1.0, disabled=classifier != SVM)
        kernel = st.selectbox("Kernel (SVM)", KERNEL_OPTIONS, disabled=classifier != SVM)
        metric = st.selectbox("Distance metric (MDM)", MDM_METRIC_OPTIONS,
                              disabled=classifier != MDM_NAME)

        st.markdown("#### Validation")
        n_folds = st.slider("K folds", 2, 10, 5)
        split = st.radio("Split strategy", ["Stratified (shuffled)", "Chronological (no shuffle)"])
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

    params = dict(
        source=source,
        feature=feature,
        band=(float(band[0]), float(band[1])),
        reject_uv=float(reject_uv),
        classifier=classifier,
        shrinkage=float(shrinkage),
        C=float(C),
        kernel=kernel,
        metric=metric,
        n_folds=int(n_folds),
        stratified=split.startswith("Stratified"),
        threshold=float(threshold),
    )
    return params, upload
