"""The deployable model: one classifier trained on every campaign, saved to models/.

    python -m radar.deploy            # train + save (about a minute)

The leave-one-campaign-out run (radar.pipeline) is how the model is *evaluated*; this file
builds the model that is *used*: the same HistGradientBoosting setup, trained on all
campaigns with the same 10:1 negative downsampling, threshold picked on a held-out 20 %.
Inputs are the 76 per-account features in the format of the released dataset
(RQ3_replier_classifier_features.csv), so anyone with data in that format can score it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config as C
from .data import data_dir, feature_columns, load_features
from .model import downsample, fit_with_threshold

MODEL_DIR = C.ROOT / "models"
MODEL_FILE = MODEL_DIR / "radar_hgb.joblib"
META_FILE = MODEL_DIR / "radar_hgb.json"


def train(sample: bool = False) -> dict:
    df = load_features(data_dir(sample))
    cols = feature_columns(df)
    y = df[C.COL_LABEL].to_numpy()
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = downsample(np.arange(len(df)), y, C.NEG_PER_POS_TRAIN, rng)
    model, thr = fit_with_threshold("hgb", df.iloc[idx][cols], y[idx])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    meta = {
        "model": "HistGradientBoostingClassifier", "features": cols, "threshold": thr,
        "trained_on": "all campaigns", "n_train": int(len(idx)), "n_pos": int(y[idx].sum()),
        "neg_per_pos": C.NEG_PER_POS_TRAIN, "sklearn": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "is_sample": sample,
        "note": "Threshold picked on a 10:1 sample; on real data, where IO accounts are far rarer, "
                "most accounts above it are false alarms. Use the score to rank for review.",
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    return meta


class Scorer:
    """Load the saved model and score accounts given in the released-features format."""

    def __init__(self, model_file: Path = MODEL_FILE, meta_file: Path = META_FILE):
        self.model = joblib.load(model_file)
        self.meta = json.loads(meta_file.read_text())
        self.features: list[str] = self.meta["features"]
        self.threshold: float = self.meta["threshold"]

    @staticmethod
    def available() -> bool:
        return MODEL_FILE.exists() and META_FILE.exists()

    def check(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """(missing feature columns, columns with text that is not a number). Converts number-like
        text (e.g. "0,5") in place; empty cells are allowed."""
        missing = [c for c in self.features if c not in df.columns]
        bad = []
        for c in self.features:
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                raw = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
                num = pd.to_numeric(raw.replace({"": None, "nan": None}), errors="coerce")
                if (num.isna() & df[c].notna() & (raw != "")).any():
                    bad.append(c)
                else:
                    df[c] = num
        return missing, bad

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].replace([np.inf, -np.inf], np.nan).astype(float)
        return self.model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    m = train(ap.parse_args().sample)
    print(json.dumps({k: v for k, v in m.items() if k != "features"}, indent=2))
    print(f"saved {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e6:.1f} MB)")
