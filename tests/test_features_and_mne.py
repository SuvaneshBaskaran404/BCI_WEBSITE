"""Tests for feature dimensionality and the MNE preprocessing / inspection helpers."""
import numpy as np
import pytest

from ml_pipeline.features import build_feature_extractor, flatten_erp
from mne_skeleton.data_loader import make_synthetic_mi
from mne_skeleton.epoch_inspector import band_power, electrode_positions, waveform_offsets
from mne_skeleton.preprocessing import bandpass, preprocess


@pytest.fixture(scope="module")
def ds():
    return make_synthetic_mi(n_trials=40, duration=2.0, seed=3)


def test_csp_output_dimension(ds):
    feats = build_feature_extractor("CSP").fit_transform(ds.X, ds.y)
    assert feats.shape == (40, 6)


def test_tangent_space_dimension_matches_upper_triangle(ds):
    n_ch = ds.X.shape[1]
    feats = build_feature_extractor("Riemann Tangent Space").fit_transform(ds.X, ds.y)
    assert feats.shape == (40, n_ch * (n_ch + 1) // 2)


def test_flattened_erp_dimension(ds):
    flat = flatten_erp(ds.X, step=4)
    assert flat.shape == (40, 22 * (ds.X.shape[2] // 4 + (ds.X.shape[2] % 4 > 0)))


def test_bandpass_preserves_shape_and_rejects_out_of_band_tone():
    sfreq = 250.0
    t = np.arange(1000) / sfreq
    tone_50hz = np.sin(2 * np.pi * 50.0 * t)
    X = np.tile(tone_50hz, (2, 3, 1))
    out = bandpass(X, sfreq, 8.0, 30.0)
    assert out.shape == X.shape
    assert out.std() < 0.1 * X.std()


def test_preprocess_flags_large_artifacts_without_removing_trials(ds):
    Xf, ok = preprocess(ds.X, ds.sfreq, 8.0, 30.0, reject_uv=150.0)
    assert Xf.shape == ds.X.shape
    assert ok.shape == (40,)
    assert ok.dtype == bool


def test_electrode_positions_and_band_power(ds):
    xy = electrode_positions(ds.ch_names)
    assert xy.shape == (22, 2)
    assert np.isfinite(xy).all(axis=1).sum() >= 20
    power = band_power(ds.X[0], ds.sfreq, 8.0, 12.0)
    assert power.shape == (22,)
    assert np.all(power >= 0)


def test_waveform_offsets_shape(ds):
    t, traces = waveform_offsets(ds.X[0], ds.sfreq, ds.tmin)
    assert t.shape == (ds.X.shape[2],)
    assert traces.shape == ds.X[0].shape
