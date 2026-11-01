"""Classifier factory and parameter normalisation."""
from __future__ import annotations

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .features import build_feature_extractor

LDA = "LDA"
SVM = "SVM (SVC)"
RF = "Random Forest"
MDM_NAME = "Riemannian MDM"
CLASSIFIER_OPTIONS = (LDA, SVM, RF, MDM_NAME)
KERNEL_OPTIONS = ("rbf", "linear")
MDM_METRIC_OPTIONS = ("riemann", "logeuclid", "euclid")


def normalize_params(p):
    """Reset hyperparameters that the chosen model ignores, so caches and exports stay clean."""
    q = dict(p)
    clf = q["classifier"]
    if clf != LDA:
        q["shrinkage"] = 0.0
    if clf != SVM:
        q["C"] = 1.0
        q["kernel"] = "rbf"
    if clf != MDM_NAME:
        q["metric"] = "riemann"
    if clf == MDM_NAME:
        q["feature"] = "n/a (MDM uses covariances)"
    return q


def build_pipeline(feature, classifier, C=1.0, shrinkage=0.1, kernel="rbf",
                   metric="riemann", cov_estimator="lwf", seed=0):
    """Return an unfitted scikit-learn pipeline taking (n_trials, n_channels, n_times)."""
    if classifier == MDM_NAME:
        return make_pipeline(Covariances(estimator=cov_estimator), MDM(metric=metric))
    features = build_feature_extractor(feature, cov_estimator=cov_estimator)
    if classifier == LDA:
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=float(shrinkage))
    elif classifier == SVM:
        # Platt-scaled probabilities; SVC(probability=True) is deprecated in scikit-learn 1.9.
        clf = CalibratedClassifierCV(SVC(C=float(C), kernel=kernel), method="sigmoid", cv=3)
    elif classifier == RF:
        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return make_pipeline(features, clf)
