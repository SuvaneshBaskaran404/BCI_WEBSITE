"""Binary classification metrics, diagnostics and trial-level explanations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

QUADRANT_LABELS = {
    "None": None,
    "TP: Right Hand correct": "tp",
    "FN: Right Hand missed": "fn",
    "FP: Left Hand called Right": "fp",
    "TN: Left Hand correct": "tn",
}


def threshold_predict(proba, threshold):
    return (np.asarray(proba) >= float(threshold)).astype(int)


def binary_metrics(y_true, y_pred, proba):
    """Positive class is 1 (Right Hand)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan")
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        sensitivity=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        tp=tp, tn=tn, fp=fp, fn=fn,
        confusion=cm.tolist(),
    )


def kappa_label(k):
    """Landis & Koch (1977) agreement bands."""
    if k < 0:
        return "Poor"
    for bound, label in ((0.2, "Slight"), (0.4, "Fair"), (0.6, "Moderate"), (0.8, "Substantial")):
        if k < bound:
            return label
    return "Almost perfect"


def quadrant_indices(y, pred, quad):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    masks = {
        "tp": (y == 1) & (pred == 1),
        "tn": (y == 0) & (pred == 0),
        "fp": (y == 0) & (pred == 1),
        "fn": (y == 1) & (pred == 0),
    }
    if quad not in masks:
        return np.array([], dtype=int)
    return np.where(masks[quad])[0]


def diagnose_generalization(train_acc, cv_acc):
    """Bias/variance diagnosis from the train vs cross-validation accuracy gap."""
    gap = train_acc - cv_acc
    if train_acc < 0.6 and cv_acc < 0.6:
        status = "Underfitting"
        message = ("Training and validation accuracy are both near chance. The model cannot "
                   "separate the classes: try a richer feature set, a wider band, or more trials.")
    elif gap > 0.10:
        status = "Overfitting"
        message = (f"Training accuracy exceeds validation accuracy by {gap:.1%}. The model is "
                   "memorising trial-specific noise. Lower C, raise λ, or use more trials.")
    elif gap < 0.03:
        status = "Healthy generalisation"
        message = (f"Training and validation accuracy agree within {abs(gap):.1%}, so the model "
                   "generalises to held-out trials.")
    else:
        status = "Mild variance"
        message = (f"A {gap:.1%} gap between training and validation accuracy. Watch this as you "
                   "make the model more flexible.")
    return dict(status=status, gap=float(gap), message=message)


def roc_points(y, proba):
    fpr, tpr, thr = roc_curve(np.asarray(y, dtype=int), np.asarray(proba, dtype=float))
    return fpr, tpr, thr


def explain_trial(X_trial, ch_names, correct, amp_threshold_v=150e-6):
    """Rule-based explanation for one trial, based on its largest peak-to-peak channel."""
    ptp = X_trial.max(axis=1) - X_trial.min(axis=1)
    worst = int(np.argmax(ptp))
    if ptp[worst] > amp_threshold_v:
        return (f"Artifact flag: {ch_names[worst]} peak-to-peak amplitude is "
                f"{ptp[worst] * 1e6:.0f} µV, above the {amp_threshold_v * 1e6:.0f} µV limit. "
                "Ocular or muscle contamination can dominate this trial's covariance estimate.")
    if not correct:
        return ("No gross artifact was flagged, so the error most likely reflects a weak "
                "lateralised mu/beta contrast. Compare band power over C3 and C4 below.")
    return "Correctly classified and no artifact flagged."
