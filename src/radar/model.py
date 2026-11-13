"""Replier classifier: leave-one-campaign-out training and evaluation.

Rules that keep the numbers honest:
* negatives are downsampled in TRAINING folds only; test folds keep the real class ratio;
* the decision threshold is picked on a validation split of the training data, never on test;
* the headline metric is PR-AUC (average precision) at the real class ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import feature_group


def make_model(name: str, seed: int = C.RANDOM_STATE):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(name)


MODEL_NAMES = ("logreg", "rf", "hgb")


def downsample(idx: np.ndarray, y: np.ndarray, ratio: int, rng) -> np.ndarray:
    """Keep all positives among `idx` and up to `ratio` negatives per positive.
    `y` holds labels for the whole dataset; `idx` selects rows from it."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_neg = min(len(neg), ratio * len(pos))
    return np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])


def _best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def fit_with_threshold(name, X, y, seed=C.RANDOM_STATE):
    """Pick a threshold on an inner validation split, then refit on all training rows."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = make_model(name, seed).fit(X_tr, y_tr)
    thr = _best_f1_threshold(y_va, m.predict_proba(X_va)[:, 1])
    return make_model(name, seed).fit(X, y), thr


def score_metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    order = np.argsort(-p)
    out = {
        "n_test": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "threshold": thr,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()), "tn": int(((pred == 0) & (y == 0)).sum()),
    }
    for k in C.PRECISION_AT_K:
        out[f"precision_at_{k}"] = float(y[order[:k]].mean()) if len(y) >= k else np.nan
    return out


def loco(df: pd.DataFrame, cols: list[str], model: str, group_col="campaign_group",
         seed=C.RANDOM_STATE, importance=False):
    """Leave-one-campaign-out. Returns (metrics per held-out campaign, out-of-fold scores,
    group permutation importance or None)."""
    rng = np.random.default_rng(seed)
    X = df[cols].to_numpy(dtype=float)
    y = df[C.COL_LABEL].to_numpy()
    groups = df[group_col].to_numpy()
    oof = np.full(len(df), np.nan)
    rows, imps = [], []
    for g in sorted(pd.unique(groups)):
        test = groups == g
        if y[test].sum() == 0 or y[~test].sum() == 0:
            rows.append({"held_out": g, "note": "skipped: no positives on one side"})
            continue
        tr_idx = downsample(np.flatnonzero(~test), y, C.NEG_PER_POS_TRAIN, rng)
        m, thr = fit_with_threshold(model, X[tr_idx], y[tr_idx], seed)
        p = m.predict_proba(X[test])[:, 1]
        oof[test] = p
        rows.append({"held_out": g, "model": model, **score_metrics(y[test], p, thr)})
        if importance:
            imps.append(group_importance(m, X[test], y[test], cols, rng).assign(held_out=g))
    imp = pd.concat(imps, ignore_index=True) if imps else None
    return pd.DataFrame(rows), oof, imp


def group_importance(m, X, y, cols, rng, repeats=3) -> pd.DataFrame:
    """Drop in PR-AUC when all columns of a feature group are shuffled together."""
    if len(y) > C.IMPORTANCE_MAX_ROWS:  # keep all positives, subsample negatives
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep = np.concatenate([pos, rng.choice(neg, C.IMPORTANCE_MAX_ROWS - len(pos), replace=False)])
        X, y = X[keep], y[keep]
    base = average_precision_score(y, m.predict_proba(X)[:, 1])
    groups = pd.Series([feature_group(c) for c in cols])
    out = []
    for grp, idx in groups.groupby(groups).groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, list(idx)] = Xp[rng.permutation(len(Xp))][:, list(idx)]
            drops.append(base - average_precision_score(y, m.predict_proba(Xp)[:, 1]))
        out.append({"group": grp, "n_features": len(idx), "pr_auc_drop": float(np.mean(drops))})
    return pd.DataFrame(out)


def replicate_balanced_cv(df, cols, model="rf", seed=C.RANDOM_STATE) -> dict:
    """The paper's setup (1:1 downsampling, 10-fold CV). For comparison only:
    NOT the real-world performance."""
    rng = np.random.default_rng(seed)
    y_all = df[C.COL_LABEL].to_numpy()
    idx = downsample(np.arange(len(df)), y_all, 1, rng)
    X, y = df[cols].to_numpy(dtype=float)[idx], y_all[idx]
    aucs, f1s = [], []
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        m = make_model(model, seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
    return {"setup": "balanced 1:1, 10-fold (paper replication)", "model": model,
            "roc_auc": float(np.mean(aucs)), "f1": float(np.mean(f1s))}
