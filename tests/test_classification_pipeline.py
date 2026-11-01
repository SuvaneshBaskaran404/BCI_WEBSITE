"""Tests for the classifier pipelines, cross-validation, metrics and the export generator."""
import ast
import json

import numpy as np
import pytest

from ml_pipeline.classifier import (CLASSIFIER_OPTIONS, MDM_NAME, build_pipeline,
                                    normalize_params)
from ml_pipeline.features import FEATURE_OPTIONS
from ml_pipeline.metrics import (binary_metrics, diagnose_generalization, quadrant_indices,
                                 threshold_predict)
from ml_pipeline.validation import cross_validate_model
from mne_skeleton.data_loader import make_synthetic_mi
from ui_components.exporter import build_config_json, build_pipeline_script


@pytest.fixture(scope="module")
def ds():
    return make_synthetic_mi(n_trials=60, duration=2.0, seed=5)


@pytest.mark.parametrize("classifier", ["LDA", "SVM (SVC)", "Random Forest"])
@pytest.mark.parametrize("feature", FEATURE_OPTIONS)
def test_pipeline_fits_and_predicts_probabilities(ds, classifier, feature):
    model = build_pipeline(feature, classifier)
    model.fit(ds.X, ds.y)
    proba = model.predict_proba(ds.X)
    assert proba.shape == (len(ds.y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_mdm_pipeline_runs_with_each_metric(ds, metric):
    model = build_pipeline("CSP", MDM_NAME, metric=metric).fit(ds.X, ds.y)
    assert model.predict(ds.X).shape == (len(ds.y),)


def test_cross_validation_covers_every_trial_once(ds):
    cv = cross_validate_model(
        ds.X, ds.y, lambda: build_pipeline("Riemann Tangent Space", "LDA"),
        n_folds=4, stratified=False,
    )
    assert not np.isnan(cv["oof_proba"]).any()
    assert 0.0 <= cv["cv_acc"] <= 1.0
    assert len(cv["fold_test_acc"]) == 4


def test_threshold_moves_sensitivity_and_specificity_in_opposite_directions():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.35, 0.7, 0.9])
    low = binary_metrics(y, threshold_predict(proba, 0.2), proba)
    high = binary_metrics(y, threshold_predict(proba, 0.8), proba)
    assert low["sensitivity"] > high["sensitivity"]
    assert low["specificity"] < high["specificity"]


def test_binary_metrics_on_known_confusion_matrix():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    proba = np.array([0.2, 0.6, 0.8, 0.9])
    m = binary_metrics(y, pred, proba)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 1, 1, 0)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 0.5
    assert m["accuracy"] == 0.75
    assert quadrant_indices(y, pred, "fp").tolist() == [1]


def test_generalization_diagnosis_labels():
    assert diagnose_generalization(0.95, 0.60)["status"] == "Overfitting"
    assert diagnose_generalization(0.55, 0.52)["status"] == "Underfitting"
    assert diagnose_generalization(0.80, 0.79)["status"] == "Healthy generalisation"


def test_normalize_params_resets_ignored_hyperparameters():
    p = dict(feature="CSP", classifier="LDA", shrinkage=0.3, C=10.0, kernel="linear",
             metric="logeuclid", n_folds=5, stratified=True, threshold=0.5, band=(8.0, 30.0),
             reject_uv=150.0, source="synthetic")
    q = normalize_params(p)
    assert q["shrinkage"] == 0.3
    assert q["C"] == 1.0 and q["kernel"] == "rbf"
    assert q["metric"] == "riemann"


@pytest.mark.parametrize("classifier", CLASSIFIER_OPTIONS)
def test_generated_script_is_valid_python_and_config_roundtrips(classifier):
    params = dict(feature="CSP", classifier=classifier, shrinkage=0.1, C=1.0, kernel="rbf",
                  metric="riemann", n_folds=5, stratified=True, threshold=0.5,
                  band=(8.0, 30.0), reject_uv=150.0, source="synthetic")
    script = build_pipeline_script(normalize_params(params))
    ast.parse(script)
    assert "pipeline_export" in script
    assert json.loads(build_config_json(params))["parameters"]["classifier"] == classifier
