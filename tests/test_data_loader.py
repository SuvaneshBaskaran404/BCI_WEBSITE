"""Tests for data loading: synthetic generator, uploads and label handling."""
import io

import numpy as np
import pandas as pd
import pytest

from mne_skeleton.data_loader import Dataset, load_uploaded, make_synthetic_mi


def test_synthetic_dataset_shapes_and_labels():
    ds = make_synthetic_mi(n_trials=40, duration=2.0)
    assert isinstance(ds, Dataset)
    assert ds.X.shape == (40, 22, 500)
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.sfreq == 250.0
    assert len(ds.ch_names) == 22


def test_csv_upload_roundtrip_preserves_values():
    ds = make_synthetic_mi(n_trials=12, duration=1.0, seed=1)
    cols = {"label": np.where(ds.y == 0, "left_hand", "right_hand")}
    for c, ch in enumerate(ds.ch_names):
        for t in range(ds.X.shape[2]):
            cols[f"{ch}:{t}"] = ds.X[:, c, t]
    buf = io.BytesIO(pd.DataFrame(cols).to_csv(index=False).encode())
    loaded = load_uploaded(buf.getvalue(), "trials.csv")
    assert loaded.X.shape == ds.X.shape
    assert loaded.ch_names == ds.ch_names
    np.testing.assert_array_equal(loaded.y, ds.y)
    np.testing.assert_allclose(loaded.X, ds.X, rtol=1e-9, atol=1e-15)


def test_npy_dict_upload_and_three_class_rejection():
    ds = make_synthetic_mi(n_trials=10, duration=1.0, seed=2)
    buf = io.BytesIO()
    np.save(buf, {"X": ds.X, "y": ds.y, "sfreq": 250.0})
    loaded = load_uploaded(buf.getvalue(), "trials.npy")
    assert loaded.X.shape == ds.X.shape

    bad = io.BytesIO()
    np.save(bad, {"X": ds.X, "y": np.arange(10) % 3})
    with pytest.raises(ValueError, match="exactly 2 classes"):
        load_uploaded(bad.getvalue(), "three.npy")


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded(b"", "data.txt")
