"""Loading and joining the dataset files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def data_dir(sample: bool) -> Path:
    return C.SAMPLE_DIR if sample else C.RAW_DIR


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` "
            "(or `python scripts/make_sample_data.py` and pass --sample)."
        )
    return path


def _read_ids(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read a CSV whose ID columns should be int64; fall back to strings if they are not."""
    df = pd.read_csv(_require(path), usecols=usecols)
    for col in usecols:
        if col.endswith("id") and df[col].dtype == object:
            df[col] = df[col].astype(str)
    return df


def tokens(name: str) -> set[str]:
    return set(name.lower().replace("-", "_").split("_"))


def load_features(d: Path) -> pd.DataFrame:
    df = pd.read_csv(_require(d / C.FEATURES_FILE))
    missing = {C.COL_USER, C.COL_LABEL} - set(df.columns)
    if missing:
        raise ValueError(
            f"{C.FEATURES_FILE} is missing {missing}. Columns found: {list(df.columns)[:20]}... "
            "Fix the names in config.py."
        )
    if df[C.COL_USER].duplicated().any():
        raise ValueError(f"{C.FEATURES_FILE}: a replier appears in more than one row.")
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df[C.COL_LABEL] = df[C.COL_LABEL].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are not IDs or labels. Refuses anything named like a label."""
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in C.NON_FEATURE_COLS and "label" not in c.lower() and not c.lower().endswith("id")
    ]
    if not cols:
        raise ValueError("No feature columns found.")
    return cols


def profile_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if tokens(c) & C.PROFILE_TOKENS and not c.startswith("graph_")]


def feature_group(col: str) -> str:
    t = tokens(col)
    for group, keys in C.FEATURE_GROUP_TOKENS:
        if t & keys:
            return group
    return "other"


def load_replies(d: Path) -> pd.DataFrame:
    """One row per (target/control tweet, reply). Label is kept for evaluation only."""
    cols = [C.COL_TWEET, C.COL_USER, C.COL_CAMPAIGN, C.COL_TYPE, C.COL_LABEL, C.COL_REPLY]
    return _read_ids(d / C.REPLIES_FILE, cols)


def load_delays(d: Path) -> pd.DataFrame | None:
    p = d / C.DELAY_FILE
    return pd.read_csv(p) if p.exists() else None


def _mode_by(df: pd.DataFrame, key: str, val: str) -> pd.DataFrame:
    """Most frequent `val` per `key`, plus how many distinct values the key had."""
    per = df.groupby([key, val]).size().rename("n").reset_index()
    per = per.sort_values([key, "n"], ascending=[True, False])
    n_distinct = per.groupby(key).size().rename(f"n_{val}s")
    return per.drop_duplicates(key).set_index(key)[[val]].join(n_distinct)


def tweet_campaigns(replies: pd.DataFrame) -> pd.Series:
    """Campaign of each target tweet = campaign of the IO accounts that replied to it.

    In the real file the `campaign` column is the *replier's* campaign and every normal
    replier is labelled 'remain', so a normal account's campaign must come from the tweets
    it replied to. The label is used here only to define evaluation groups, never as a feature.
    """
    io = replies[(replies[C.COL_LABEL] == 1) & (replies[C.COL_TYPE] == C.TARGET_TYPE)]
    return _mode_by(io, C.COL_TWEET, C.COL_CAMPAIGN)[C.COL_CAMPAIGN]


def account_campaigns(replies: pd.DataFrame, top_n: int = C.TOP_CAMPAIGNS) -> pd.DataFrame:
    """Assign each replier the campaign of the target tweets it replied to most;
    pool small campaigns into 'other'. Campaigns ranked by number of target tweets."""
    tc = tweet_campaigns(replies)
    r = replies.loc[replies[C.COL_TYPE] == C.TARGET_TYPE, [C.COL_USER, C.COL_TWEET]]
    r = r.assign(**{C.COL_CAMPAIGN: r[C.COL_TWEET].map(tc)}).dropna(subset=[C.COL_CAMPAIGN])
    main = _mode_by(r, C.COL_USER, C.COL_CAMPAIGN)
    top = tc.value_counts().nlargest(top_n).index
    main["campaign_group"] = main[C.COL_CAMPAIGN].where(main[C.COL_CAMPAIGN].isin(top), "other")
    return main.reset_index()
