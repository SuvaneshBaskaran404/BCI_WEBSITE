"""Epoch inspection helpers: electrode layout, stacked waveforms and scalp-map values."""
from __future__ import annotations

import numpy as np
import mne


def make_info(ch_names, sfreq):
    info = mne.create_info(list(ch_names), float(sfreq), ch_types="eeg")
    try:
        montage = mne.channels.make_standard_montage("colin27_1020")
    except ValueError:  # MNE < 1.8 only knows the older name
        montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def electrode_positions(ch_names):
    """(n_channels, 2) x/y electrode coordinates from the 10-20 montage; NaN if unknown."""
    info = make_info(ch_names, 250.0)
    montage = info.get_montage()
    pos = montage.get_positions()["ch_pos"] if montage is not None else {}
    return np.array([pos.get(ch, [np.nan, np.nan, np.nan])[:2] for ch in ch_names], dtype=float)


def waveform_offsets(X_trial, sfreq, tmin=0.0):
    """Centre each channel and offset it vertically so channels stack without overlapping."""
    X = np.asarray(X_trial, dtype=float)
    n_ch, n_t = X.shape
    t = tmin + np.arange(n_t) / float(sfreq)
    centred = X - X.mean(axis=1, keepdims=True)
    spacing = 4.0 * np.std(centred) if np.std(centred) > 0 else 1.0
    offsets = (np.arange(n_ch)[::-1] * spacing)[:, None]
    return t, centred + offsets


def band_power(X_trial, sfreq, fmin=8.0, fmax=12.0):
    """Mean Welch power per channel in a frequency band, shape (n_channels,)."""
    from mne.time_frequency import psd_array_welch

    X = np.asarray(X_trial, dtype=float)
    n_fft = min(256, X.shape[-1])
    psd, _ = psd_array_welch(X, float(sfreq), fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    return psd.mean(axis=-1)
