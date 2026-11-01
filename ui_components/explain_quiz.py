"""Explain panel: live diagnostics, expandable concept cards and a graded self-check quiz."""
import streamlit as st

from ui_components.theme_styles import badge_html, card_html

CONCEPTS = [
    ("Riemannian geometry",
     "EEG spatial covariance matrices live on the cone of symmetric positive definite (SPD) "
     "matrices. Averaging them with Euclidean arithmetic inflates their eigenvalues (the "
     "swelling effect) and washes out mu-band desynchronisation. Riemannian distances respect "
     "the cone's geometry: δ(P₁, P₂) = ‖log(P₁^{-1/2} P₂ P₁^{-1/2})‖_F."),
    ("Shrinkage LDA",
     "With many channels and few trials the empirical covariance is noisy and often singular. "
     "Shrinkage blends it with a structured target (a scaled identity): Σ = (1−λ)S + λ·μI. "
     "λ = 0 trusts the data fully; λ = 1 ignores it. Pick λ by cross-validation, not by eye."),
    ("BCI data leakage",
     "Consecutive EEG trials share electrode impedance, drift and fatigue. A shuffled split puts "
     "near-duplicate trials in both train and test folds, so accuracy measures memorised session "
     "structure. Chronological splits test on later, unseen recording time and give honest numbers."),
]

QUIZ = [
    {
        "q": "A large frontal EOG blink corrupts Trial #104. Which preprocessing step resolves it "
             "without destroying the mu rhythm?",
        "options": ["Severe bandpass below 15 Hz",
                    "Adaptive EOG regression or ICA-based artifact removal"],
        "answer": 1,
        "why": "Blinks are broadband and frontal. ICA or EOG regression removes that component "
               "and keeps the 8–12 Hz mu rhythm. A low-pass filter below 15 Hz would delete the "
               "mu band itself.",
    },
    {
        "q": "Why can a shuffled K-fold split inflate accuracy on session-recorded EEG?",
        "options": ["It makes the training folds smaller",
                    "Trials from one session leak across folds, so the model learns session drift "
                    "that the test fold shares",
                    "Shuffling swaps the labels"],
        "answer": 1,
        "why": "Non-stationary signals are shared within a session. Random splits place near-identical "
               "trials on both sides, so the score reflects memorised drift. Chronological splits "
               "test on unseen time.",
    },
    {
        "q": "What does raising the shrinkage λ in shrinkage LDA do?",
        "options": ["Pulls the covariance estimate toward a scaled identity, lowering variance when "
                    "trials are few",
                    "Adds more CSP filters"],
        "answer": 0,
        "why": "Shrinkage regularises the covariance matrix. It trades a little bias for much lower "
               "variance, which is usually the right trade with few, high-dimensional trials.",
    },
]


def render_diagnostics(diag, m, threshold):
    tone = "ok" if diag["status"] == "Healthy generalisation" else "err" \
        if diag["status"] in ("Overfitting", "Underfitting") else "info"
    st.markdown(card_html(
        f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:6px'>"
        f"<b>Live model diagnostics</b> {badge_html(diag['status'], tone)}</div>"
        f"<div style='font-size:13px'>{diag['message']}</div>"
        f"<div style='font-size:12px;color:#464555;margin-top:8px'>"
        f"At τ = {threshold:.2f}: sensitivity {m['sensitivity']:.1%}, "
        f"specificity {m['specificity']:.1%}, F1 {m['f1']:.3f}. Raising τ trades sensitivity "
        f"for specificity.</div>"
    ), unsafe_allow_html=True)


def render_concepts():
    for title, body in CONCEPTS:
        with st.expander(title):
            st.write(body)


def render_quiz():
    with st.form("quiz_form"):
        for i, item in enumerate(QUIZ):
            st.markdown(f"**Q{i + 1}. {item['q']}**")
            st.radio(f"Answer {i + 1}", item["options"], index=None, key=f"quiz_{i}",
                     label_visibility="collapsed")
        submitted = st.form_submit_button("Check answers")
    if submitted:
        score = 0
        for i, item in enumerate(QUIZ):
            chosen = st.session_state.get(f"quiz_{i}")
            if chosen is None:
                st.warning(f"Q{i + 1}: no answer selected.")
                continue
            ok = chosen == item["options"][item["answer"]]
            score += int(ok)
            (st.success if ok else st.error)(f"Q{i + 1}: {'Correct' if ok else 'Not quite'}. {item['why']}")
        st.info(f"Score: {score} / {len(QUIZ)}")
