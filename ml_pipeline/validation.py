"""Cross-validation with stratified (shuffled) or chronological (unshuffled) folds."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validate_model(X, y, build_fn, n_folds=5, stratified=True, seed=0):
    """Out-of-fold probabilities plus fold-averaged train and test accuracy (threshold 0.5).

    Chronological mode uses contiguous, unshuffled blocks, so every test block lies in a
    different stretch of recording time. Stratified mode shuffles trials, which can place
    trials from the same session in both train and test folds (a leakage path).
    """
    y = np.asarray(y, dtype=int)
    if stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=False)

    oof = np.full(len(y), np.nan)
    train_accs, test_accs = [], []
    for tr, te in splitter.split(X, y):
        model = build_fn()
        model.fit(X[tr], y[tr])
        train_accs.append(accuracy_score(y[tr], model.predict(X[tr])))
        proba = model.predict_proba(X[te])[:, 1]
        oof[te] = proba
        test_accs.append(accuracy_score(y[te], (proba >= 0.5).astype(int)))

    return dict(
        oof_proba=oof,
        train_acc=float(np.mean(train_accs)),
        cv_acc=float(np.mean(test_accs)),
        cv_std=float(np.std(test_accs)),
        fold_test_acc=[float(a) for a in test_accs],
        n_folds=int(n_folds),
        stratified=bool(stratified),
    )
