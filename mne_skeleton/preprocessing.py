"""Filtering and artifact flagging, using MNE's filter engine."""
from __future__ import annotations

import numpy as np
import mne


def bandpass(X, sfreq, l_freq, h_freq, order=4):
    """Zero-phase Butterworth bandpass applied per trial and channel (no labels involved)."""
    return mne.filter.filter_data(
        np.asarray(X, dtype=np.float64), float(sfreq), float(l_freq), float(h_freq),
        method="iir", iir_params=dict(order=order, ftype="butter"), verbose=False,
    )


def artifact_mask(X, threshold_v=150e-6):
    """True for trials whose peak-to-peak amplitude stays below ``threshold_v`` on every channel."""
    ptp = X.max(axis=2) - X.min(axis=2)
    return ptp.max(axis=1) < threshold_v


def preprocess(X, sfreq, l_freq, h_freq, reject_uv=150.0):
    """Bandpass the epochs and flag artifact trials. Trials are flagged, not removed."""
    Xf = bandpass(X, sfreq, l_freq, h_freq)
    ok = artifact_mask(Xf, reject_uv * 1e-6)
    return Xf, ok
