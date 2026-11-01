"""Dataset loading: synthetic motor imagery, MOABB benchmark data and user uploads.

Conventions
-----------
* Epoched data has shape (n_trials, n_channels, n_times) and is stored in Volts (MNE convention).
* Labels are binary integers: 0 = first class alphabetically (Left Hand), 1 = second (Right Hand).
* Trials are assumed to be in acquisition (chronological) order.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
CLASS_NAMES = {0: "Left Hand", 1: "Right Hand"}
DEFAULT_SFREQ = 250.0


@dataclass
class Dataset:
    """Epoched EEG with binary labels."""

    X: np.ndarray
    y: np.ndarray
    sfreq: float
    ch_names: list
    tmin: float
    source: str
    class_names: dict = field(default_factory=lambda: dict(CLASS_NAMES))

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (trials, channels, times); got shape {self.X.shape}")
        if len(self.y) != self.X.shape[0]:
            raise ValueError("Number of labels does not match number of trials")
        if len(self.ch_names) != self.X.shape[1]:
            raise ValueError("Number of channel names does not match number of channels")


def _default_names(n_channels):
    if n_channels == len(DEFAULT_CH_NAMES):
        return list(DEFAULT_CH_NAMES)
    return [f"Ch{i + 1}" for i in range(n_channels)]


def _binary_labels(raw):
    """Map two arbitrary labels to 0/1 (sorted order) and build display names."""
    labels = [str(v) for v in np.asarray(raw).ravel()]
    uniq = sorted(set(labels))
    if len(uniq) != 2:
        raise ValueError(
            f"Binary classification needs exactly 2 classes; found {len(uniq)}: {uniq[:6]}"
        )
    y = np.array([uniq.index(v) for v in labels], dtype=int)
    names = {i: u.replace("_", " ").title() for i, u in enumerate(uniq)}
    return y, names


def make_synthetic_mi(n_trials=120, sfreq=DEFAULT_SFREQ, duration=4.0, tmin=0.0,
                      seed=42, blink_rate=0.08):
    """Synthetic 22-channel left/right motor imagery with mu-band ERD and occasional blinks.

    Left-hand imagery desynchronises the right-hemisphere mu rhythm (C4) and vice versa,
    which is the physiological effect CSP and Riemannian methods exploit.
    """
    rng = np.random.default_rng(seed)
    ch_names = list(DEFAULT_CH_NAMES)
    n_times = int(round(duration * sfreq))
    t = tmin + np.arange(n_times) / sfreq
    y = rng.integers(0, 2, size=n_trials)
    X = rng.normal(0.0, 8e-6, size=(n_trials, len(ch_names), n_times))
    c3, c4, fz = ch_names.index("C3"), ch_names.index("C4"), ch_names.index("Fz")

    for i in range(n_trials):
        phase = rng.uniform(0, 2 * np.pi)
        erd = np.ones(len(ch_names))
        if y[i] == 0:
            erd[c4] = 0.35
        else:
            erd[c3] = 0.35
        # Trial-to-trial amplitude variability keeps the task realistically hard.
        mu = 1.5e-6 * rng.lognormal(0.0, 0.5) * np.sin(2 * np.pi * 10.0 * t + phase)
        X[i] += erd[:, None] * mu[None, :]
        if rng.uniform() < blink_rate:
            blink_t = tmin + rng.uniform(0.3 * duration, 0.7 * duration)
            X[i, fz] += 80e-6 * np.exp(-((t - blink_t) ** 2) / (2 * 0.08 ** 2))

    return Dataset(X, y, float(sfreq), ch_names, float(tmin),
                   "Synthetic motor imagery (offline)")


def load_moabb_bnci2014_001(subject=1, cache_dir=".cache/moabb", tmin=0.5, tmax=3.5,
                            fmin=8.0, fmax=32.0):
    """Left vs right hand motor imagery from BNCI2014-001 via MOABB.

    The first call downloads the dataset (a large download); the epoched result is then cached
    as .npz in ``cache_dir``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"bnci2014_001_s{subject}_{tmin}_{tmax}.npz")
    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=False) as f:
            X, labels = f["X"], f["labels"]
            sfreq, ch_names = float(f["sfreq"]), [str(c) for c in f["ch_names"]]
    else:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import LeftRightImagery

        dataset = BNCI2014_001()
        paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        X, raw_labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        sessions = dataset.get_data(subjects=[subject])[subject]
        first_raw = next(iter(next(iter(sessions.values())).values())).copy().pick("eeg")
        sfreq = float(first_raw.info["sfreq"])
        ch_names = list(first_raw.ch_names)
        labels = np.asarray(raw_labels).astype(str)
        np.savez(cache_file, X=X, labels=labels, sfreq=sfreq, ch_names=np.array(ch_names))

    y, names = _binary_labels(labels)
    return Dataset(X, y, sfreq, ch_names, float(tmin),
                   f"MOABB BNCI2014-001, subject {subject}", names)


def _from_npy(data, filename, sfreq_default):
    obj = np.load(io.BytesIO(data), allow_pickle=True)
    if obj.dtype != object or obj.ndim != 0:
        raise ValueError(
            ".npy uploads must hold a dict saved with np.save, e.g. "
            "np.save('trials.npy', {'X': X, 'y': y, 'sfreq': 250.0})"
        )
    d = obj.item()
    if "X" not in d or "y" not in d:
        raise ValueError(".npy dict needs 'X' (trials, channels, times) and 'y' (labels)")
    X = np.asarray(d["X"], dtype=float)
    y, names = _binary_labels(d["y"])
    sfreq = float(d.get("sfreq", sfreq_default))
    ch_names = list(d.get("ch_names", _default_names(X.shape[1])))
    return Dataset(X, y, sfreq, ch_names, 0.0, f"Upload: {filename}", names)


def _from_csv(data, filename, sfreq_default):
    """CSV layout: a 'label' column plus one column per 'CHANNEL:SAMPLE_INDEX', one row per trial.

    Example header: label,Fz:0,Fz:1,...,C3:0,C3:1,... Values are in Volts.
    """
    df = pd.read_csv(io.BytesIO(data))
    if "label" not in df.columns:
        raise ValueError("CSV needs a 'label' column")
    feat_cols = [c for c in df.columns if c != "label"]
    parsed = [c.rsplit(":", 1) for c in feat_cols]
    if any(len(p) != 2 or not p[1].isdigit() for p in parsed):
        raise ValueError("CSV feature columns must be named '<channel>:<sample>', e.g. C3:0, C3:1")
    chs = list(dict.fromkeys(p[0] for p in parsed))
    times = sorted({int(p[1]) for p in parsed})
    expected = [f"{ch}:{t}" for ch in chs for t in times]
    if sorted(feat_cols) != sorted(expected):
        raise ValueError("Every channel must have the same set of sample indices")
    X = df[expected].to_numpy(dtype=float).reshape(len(df), len(chs), len(times))
    y, names = _binary_labels(df["label"].to_numpy())
    return Dataset(X, y, float(sfreq_default), chs, 0.0, f"Upload: {filename}", names)


def _from_fif(data, filename):
    import mne

    with tempfile.NamedTemporaryFile(suffix=".fif", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        epochs = mne.read_epochs(path, preload=True, verbose=False)
    except Exception as exc:  # noqa: BLE001 - surface any MNE read error to the user
        raise ValueError(f"Could not read '{filename}' as MNE Epochs: {exc}") from exc
    finally:
        os.remove(path)
    inverse = {code: name for name, code in epochs.event_id.items()}
    labels = [inverse.get(code, str(code)) for code in epochs.events[:, 2]]
    y, names = _binary_labels(labels)
    return Dataset(epochs.get_data(copy=True), y, float(epochs.info["sfreq"]),
                   list(epochs.ch_names), float(epochs.tmin), f"Upload: {filename}", names)


def load_uploaded(data: bytes, filename: str, sfreq_default: float = DEFAULT_SFREQ) -> Dataset:
    """Parse a user upload. Supported: .fif (MNE Epochs), .csv (see _from_csv), .npy (dict)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npy":
        return _from_npy(data, filename, sfreq_default)
    if ext == ".csv":
        return _from_csv(data, filename, sfreq_default)
    if ext == ".fif":
        return _from_fif(data, filename)
    raise ValueError(f"Unsupported file type '{ext}'. Use .fif, .csv or .npy.")
