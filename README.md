# EEG Classification Explorer (BCI_WEBSITE)

Module 8 of an interactive Brain-Computer Interface teaching app. It classifies left vs right hand
motor imagery with CSP, Riemannian tangent-space and raw-ERP features, and four classifiers (LDA,
SVM, Random Forest, Riemannian MDM). Everything updates live as you change parameters, and each
plot links to a trial inspector so you can see why a trial was misclassified.

---

## User guide

### Run the app (one command)

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and works offline with a synthetic dataset.

### Install

Option A, pip (Python 3.10 to 3.12):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Option B, conda:

```bash
conda env create -f environment.yml
conda activate bci-website
```

### Datasets

| Source | What it is | Notes |
| --- | --- | --- |
| Synthetic motor imagery | 120 trials, 22 channels, 250 Hz, mu-band desynchronisation plus eye blinks | Default. Offline, instant. |
| MOABB BNCI2014-001 | Real left vs right hand imagery, subject 1 | Downloads once (a large file), then cached in `.cache/moabb/`. Needs internet. |
| Upload | Your own data (see formats below) | Binary labels only. |

**Upload formats**

- **`.fif`**: an MNE `Epochs` file. Labels come from event names. Raw (continuous) files are not supported.
- **`.csv`**: one row per trial, a `label` column, then one column per `CHANNEL:SAMPLE` (for example `C3:0`, `C3:1`). Values are in Volts.
- **`.npy`**: a dictionary saved with `np.save`, for example `np.save("trials.npy", {"X": X, "y": y, "sfreq": 250.0, "ch_names": names})`. `X` has shape (trials, channels, times) in Volts. `ch_names` is optional.

Labels must have exactly two classes. Sorting the labels alphabetically sets the class order, so the first class becomes 0 (Left Hand in the built-in datasets) and the second becomes 1 (Right Hand). Right Hand is the positive class in all metrics.

### Controls

- **Feature pipeline**: CSP (log-variance of spatial filters), Riemannian tangent space (SPD covariances mapped to a Euclidean tangent plane), or flattened raw ERP.
- **Bandpass** and **artifact threshold**: filter band in Hz, and the peak-to-peak level in µV above which a trial is flagged. Flagged trials are reported, not removed.
- **Model**: LDA (shrinkage λ), SVM (penalty C, kernel), Random Forest, or Riemannian MDM (distance metric). Controls that don't apply to the chosen model are greyed out.
- **K folds** (2 to 10): number of cross-validation folds.
- **Split strategy**: *Stratified (shuffled)* gives every fold the same class balance but ignores time. *Chronological* uses contiguous, unshuffled blocks, which is the honest choice for session-recorded data. Compare the two to see data leakage in action.
- **Decision threshold** (0.10 to 0.90): shifts sensitivity against specificity without retraining, because it is applied to the cross-validated probabilities.

### Reading the lab

1. **Plot 1** shows the 2-D tangent-space projection. The shaded background is the decision boundary of a 2-D surrogate classifier fitted to that projection, so it is a visual aid, not your full model. Crosses mark misclassified trials. Drag to select trials; the inspector loads the first one.
2. **Plot 2** shows the confusion matrix or ROC curve. Use *Highlight a quadrant* to light up, for example, all false positives in Plot 1. Point selections in Plot 1 take priority over the quadrant.
3. **Trial inspector** shows the stacked EEG traces, a scalp map of mu-band (8–12 Hz) power, and a rule-based root-cause note.
4. **Explain** panel gives a live bias/variance diagnosis from the train vs. validation accuracy gap, concept cards (Riemannian geometry, shrinkage LDA, data leakage), and a three-question graded quiz.
5. **Export Pipeline Code** shows a standalone script (`pipeline_export.py`) and a JSON config (`config_export.json`), both built from your current settings.

### Running the exported script

Copy `pipeline_export.py` into the project root and run `python pipeline_export.py`. It prints fold accuracies, the confusion matrix, Cohen's κ, and ROC-AUC, and writes `config_export.json`.

---

## Developer guide

### Repository layout

```
app.py                       Streamlit entry point (layout, caching, wiring)
mne_skeleton/                MNE-Python data and preprocessing layer
  data_loader.py               synthetic MI, MOABB BNCI2014-001, .fif/.csv/.npy uploads
  preprocessing.py             MNE IIR bandpass, artifact flagging
  epoch_inspector.py           10-20 montage, stacked waveforms, band-power scalp values
ml_pipeline/                 scikit-learn / pyRiemann modelling layer
  features.py                  CSP, Riemannian tangent space, flattened ERP, tangent-space PCA
  classifier.py                classifier factory, normalize_params()
  validation.py                stratified and chronological cross-validation
  metrics.py                   confusion metrics, kappa, diagnostics, quadrant selection
ui_components/               Streamlit / Plotly presentation layer
  theme_styles.py              Figma design tokens as CSS and HTML helpers
  sidebar.py                   controls, returns a params dict
  kpi_ribbon.py                four KPI cards
  linked_plots.py              Plot 1, Plot 2, waveform and topomap figures; selection parsing
  explain_quiz.py              diagnostics, concept cards, quiz
  exporter.py                  pipeline_export.py and config_export.json generators
tests/                       pytest suite (data, features/MNE, classification pipeline)
.streamlit/config.toml       light theme matching the Figma palette
```

### Design rules

- **Data units**: volts everywhere (MNE convention). UI thresholds are in µV and converted at the boundary.
- **Caching**: `st.cache_data` wraps dataset loading, preprocessing, the 2-D projection, and cross-validation. Changing only the decision threshold reuses the cached probabilities, which keeps interactions fast.
- **No leakage**: bandpass filtering and the 2-D projection are unsupervised and run on all trials. Everything that uses labels (CSP, LDA, SVM, MDM, RF) runs inside each fold.
- **Parameter hygiene**: `normalize_params` resets hyperparameters that the selected model ignores, so cache keys and exports stay consistent.

### Tests

```bash
pytest
```

The suite covers data loading and round-trips, feature dimensions, MNE preprocessing, every classifier and feature combination, cross-validation coverage, threshold behaviour, metric correctness, diagnostics, and export script validity.

### Extending

- **New dataset**: add a loader to `mne_skeleton/data_loader.py` that returns a `Dataset`, then add an option in `ui_components/sidebar.py` and a branch in `load_dataset` in `app.py`.
- **P300 ERP**: the design sketch mentions P300. Add a MOABB `P300` paradigm loader and feed it through the same pipeline. The ERP feature extractor already handles it.
- **New classifier**: add it to `build_pipeline` and `CLASSIFIER_OPTIONS`, and extend `normalize_params` and `_model_lines` in `exporter.py`.

### Known limitations

- The first MOABB load downloads a large dataset and needs internet access.
- Plot 2 quadrant selection uses a radio control. Plotly heatmaps don't emit selection events in Streamlit, so clicking a cell is not supported.
- The Plot 1 decision boundary comes from a 2-D surrogate, not the full model.
- Live EEG streaming (LSL) and GPU metering shown in the Figma mockup are out of scope and are not implemented.

## Licence

Academic teaching project. Add a licence before public release.
