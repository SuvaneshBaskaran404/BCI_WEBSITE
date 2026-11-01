"""Feature extractors: CSP, Riemannian tangent space and flattened raw ERP."""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURE_OPTIONS = ("CSP", "Riemann Tangent Space", "Raw ERP (flattened)")
CSP_COMPONENTS = 6
ERP_DOWNSAMPLE = 4


def flatten_erp(X, step=ERP_DOWNSAMPLE):
    X = np.asarray(X)
    return X[:, :, ::step].reshape(X.shape[0], -1)


def build_feature_extractor(name, cov_estimator="lwf"):
    if name == "CSP":
        return CSP(n_components=CSP_COMPONENTS, log=True)
    if name == "Riemann Tangent Space":
        return make_pipeline(Covariances(estimator=cov_estimator), TangentSpace(metric="riemann"))
    if name == "Raw ERP (flattened)":
        return make_pipeline(FunctionTransformer(flatten_erp), StandardScaler())
    raise ValueError(f"Unknown feature pipeline: {name}")


def tangent_space_2d(X, cov_estimator="lwf"):
    """Unsupervised 2-D projection (PCA on tangent-space features) for Plot 1."""
    ts = make_pipeline(Covariances(estimator=cov_estimator), TangentSpace(metric="riemann"))
    return PCA(n_components=2, random_state=0).fit_transform(ts.fit_transform(X))
