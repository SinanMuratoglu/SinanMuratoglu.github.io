"""Multi-label classifier on the extracted features + evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

from .io import CLASSES


def make_model():
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          class_weight="balanced", random_state=0)
    return MultiOutputClassifier(base)


def _proba(model, X):
    return np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X))
                            for p in model.predict_proba(X)])


def tune_thresholds(y, p):
    """Per-class threshold maximising F1 on validation data."""
    grid = np.linspace(0.05, 0.95, 19)
    return np.array([grid[np.argmax([f1_score(y[:, j], p[:, j] >= t, zero_division=0)
                                     for t in grid])] for j in range(y.shape[1])])


def evaluate(y, p, thr) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(CLASSES):
        yj, pj = y[:, j], p[:, j]
        has_both = 0 < yj.sum() < len(yj)
        rows.append({
            "class": c, "positives": int(yj.sum()),
            "roc_auc": roc_auc_score(yj, pj) if has_both else np.nan,
            "avg_precision": average_precision_score(yj, pj) if has_both else np.nan,
            "f1": f1_score(yj, pj >= thr[j], zero_division=0), "threshold": thr[j],
        })
    df = pd.DataFrame(rows)
    yhat = p >= thr
    df.loc[len(df)] = {"class": "MACRO", "positives": int(y.sum()),
                       "roc_auc": df["roc_auc"].mean(), "avg_precision": df["avg_precision"].mean(),
                       "f1": f1_score(y, yhat, average="macro", zero_division=0), "threshold": np.nan}
    return df


def train_and_evaluate(feats: pd.DataFrame, labels: pd.DataFrame, splits: dict[str, list[str]], out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = feats.join(labels, how="inner")
    fcols = list(feats.columns)
    part = {k: data.loc[data.index.intersection(v)] for k, v in splits.items()}
    for k, d in part.items():
        print(f"{k}: {len(d)} images")

    model = make_model().fit(part["train"][fcols], part["train"][CLASSES].values)
    pv = _proba(model, part["val"][fcols])
    thr = tune_thresholds(part["val"][CLASSES].values, pv)
    report = evaluate(part["test"][CLASSES].values, _proba(model, part["test"][fcols]), thr)

    joblib.dump({"model": model, "features": fcols, "thresholds": thr}, out / "model.joblib")
    report.to_csv(out / "test_report.csv", index=False)
    imp = _importance(model, part["val"][fcols], part["val"][CLASSES].values)
    imp.to_csv(out / "feature_importance.csv")
    (out / "summary.json").write_text(json.dumps(
        {k: len(v) for k, v in part.items()} | {"macro_auc": float(report.iloc[-1]["roc_auc"])}, indent=2))
    return report, imp


def _importance(model, X, y, n_repeats=3):
    from sklearn.inspection import permutation_importance
    cols = {}
    for j, (est, c) in enumerate(zip(model.estimators_, CLASSES)):
        if 0 < y[:, j].sum() < len(y):
            r = permutation_importance(est, X, y[:, j], scoring="roc_auc",
                                       n_repeats=n_repeats, random_state=0)
            cols[c] = r.importances_mean
    return pd.DataFrame(cols, index=X.columns)
