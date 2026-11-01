"""EEG data loading, preprocessing and epoch inspection built on MNE-Python."""
import mne

mne.set_log_level("WARNING")  # keep CSP / covariance chatter out of the console

from .data_loader import Dataset, load_moabb_bnci2014_001, load_uploaded, make_synthetic_mi
from .preprocessing import artifact_mask, bandpass, preprocess
from .epoch_inspector import band_power, electrode_positions, make_info, waveform_offsets

__all__ = [
    "Dataset", "load_moabb_bnci2014_001", "load_uploaded", "make_synthetic_mi",
    "artifact_mask", "bandpass", "preprocess",
    "band_power", "electrode_positions", "make_info", "waveform_offsets",
]
