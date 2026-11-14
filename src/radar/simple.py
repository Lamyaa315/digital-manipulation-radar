"""The simple-input model: features anyone can compute from a plain list of replies.

Input (one row per reply):
    account_id, tweet_id                       required
    reply_time  (or reply_id)                  required — when the reply was posted
    tweet_time                                 optional — derived from tweet_id when it is a tweet ID
    account_created_at, followers, following   optional — account profile
    reply_text                                 optional — used only by a rule (near-copies), not the model

Only features that can be rebuilt *exactly* from that input are used, so the model sees the
same numbers in training and in use. Checked on the dataset: the reply-delay statistics match
the released ones on 99.99 % of accounts; account age matches to about half a day.
Text similarity is NOT in this model (the released one needs the LaBSE language model and
every reply to every tweet); it is shown as a separate rule instead.

    python -m radar.simple     # build training table, evaluate leave-one-campaign-out, save model
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .model import downsample, fit_with_threshold, make_model, score_metrics

TWEPOCH_MS = 1288834974657
MODEL_FILE = C.ROOT / "models" / "radar_simple.joblib"
META_FILE = C.ROOT / "models" / "radar_simple.json"
STATS = ["mean", "min", "25%", "50%", "75%", "max", "range", "iqr", "entropy"]
GRAPH = ["graph_degree", "graph_strength", "graph_community_size"]
FEATURES = ([f"{s}_delay_h" for s in STATS]
            + ["age_years", "log_followers", "log_following", "n_replies", "n_tweets"] + GRAPH)


def snowflake_ms(ids) -> np.ndarray:
    return (pd.Series(ids).astype("int64").to_numpy() >> 22) + TWEPOCH_MS


def _entropy(x: np.ndarray) -> float:
    _, c = np.unique(x, return_counts=True)
    p = c / len(x)
    return float(-(p * np.log2(p)).sum())


def delay_stats(g: pd.Series) -> pd.Series:
    """Same statistics as the released dataset (pandas describe + range, iqr, log2 entropy)."""
    x = g.to_numpy(dtype=float)
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return pd.Series([x.mean(), x.min(), q[0], q[1], q[2], x.max(), x.max() - x.min(), q[2] - q[0], _entropy(x)],
                     index=[f"{s}_delay_h" for s in STATS])


def features(rep: pd.DataFrame) -> pd.DataFrame:
    """rep columns: account, tweet, reply_ms, tweet_ms, [created_ms, followers, following].
    Delay in hours rounded up, as in the released dataset."""
    rep = rep.copy()
    rep["delay_h"] = np.ceil((rep["reply_ms"] - rep["tweet_ms"]) / 3.6e6).clip(lower=0)
    g = rep.groupby("account")
    out = g["delay_h"].apply(delay_stats).unstack()
    out["n_replies"] = g.size()
    out["n_tweets"] = g["tweet"].nunique()
    first = g["reply_ms"].min()
    if "created_ms" in rep:
        out["age_years"] = (first - g["created_ms"].first()) / (365.25 * 86400e3)
    else:
        out["age_years"] = np.nan
    for c in ("followers", "following"):
        out[f"log_{c}"] = np.log1p(g[c].first()) if c in rep else np.nan
    return out.join(graph_feats(rep))[FEATURES]


def graph_feats(rep: pd.DataFrame) -> pd.DataFrame:
    """Co-reply network features, built exactly like the pipeline's (>= 2 tweets per account,
    >= 3 shared tweets per link, Louvain communities). Meaningful when the upload holds all the
    replies to the tweets under investigation, as the dataset does."""
    from .graph import build_coreply_edges, communities, to_graph
    r = pd.DataFrame({C.COL_USER: rep["account"], C.COL_TWEET: rep["tweet"], C.COL_TYPE: C.TARGET_TYPE})
    e = build_coreply_edges(r)
    users = pd.Index(rep["account"].unique())
    out = pd.DataFrame(0.0, index=users, columns=GRAPH)
    if len(e):
        g = to_graph(e)
        mem = pd.Series(communities(g))
        out.loc[list(g.nodes), "graph_degree"] = pd.Series(dict(g.degree()))
        out.loc[list(g.nodes), "graph_strength"] = pd.Series(dict(g.degree(weight="weight")))
        out.loc[mem.index, "graph_community_size"] = mem.map(mem.value_counts()).astype(float)
    return out


# -- training ------------------------------------------------------------------------------
def training_table() -> pd.DataFrame:
    """Rebuild the simple features for every account in the dataset."""
    r = pd.read_pickle(C.ARTIFACTS_DIR / "replies.pkl")
    r = r[r[C.COL_TYPE] == C.TARGET_TYPE]
    rep = pd.DataFrame({"account": r[C.COL_USER].to_numpy(), "tweet": r[C.COL_TWEET].to_numpy(),
                        "reply_ms": snowflake_ms(r[C.COL_REPLY]), "tweet_ms": snowflake_ms(r[C.COL_TWEET])})
    rep["delay_h"] = np.ceil((rep["reply_ms"] - rep["tweet_ms"]) / 3.6e6).clip(lower=0)
    # vectorised version of features() for 2.6M rows
    g = rep.groupby("account")["delay_h"]
    q = g.quantile([0.25, 0.5, 0.75]).unstack()
    ent = rep.groupby(["account", "delay_h"]).size().rename("n").reset_index()
    ent["p"] = ent["n"] / ent.groupby("account")["n"].transform("sum")
    ent = (-(ent["p"] * np.log2(ent["p"]))).groupby(ent["account"]).sum()
    t = pd.DataFrame({"mean_delay_h": g.mean(), "min_delay_h": g.min(), "25%_delay_h": q[0.25],
                      "50%_delay_h": q[0.5], "75%_delay_h": q[0.75], "max_delay_h": g.max()})
    t["range_delay_h"] = t["max_delay_h"] - t["min_delay_h"]
    t["iqr_delay_h"] = t["75%_delay_h"] - t["25%_delay_h"]
    t["entropy_delay_h"] = ent
    t["n_replies"] = rep.groupby("account").size()
    t["n_tweets"] = rep.groupby("account")["tweet"].nunique()
    acc = pd.read_pickle(C.ARTIFACTS_DIR / "accounts.pkl").set_index(C.COL_USER)
    t = t.join(acc[[C.COL_LABEL, "campaign_group", "age", *GRAPH]], how="inner").rename(columns={"age": "age_years"})
    info = pd.read_csv(C.RAW_DIR / C.REPLIER_INFO_FILE, usecols=[C.COL_USER, "followers_count", "following_count"])
    info = info.drop_duplicates(C.COL_USER).set_index(C.COL_USER)
    t["log_followers"] = np.log1p(info["followers_count"].reindex(t.index))
    t["log_following"] = np.log1p(info["following_count"].reindex(t.index))
    return t


def evaluate(t: pd.DataFrame) -> pd.DataFrame:
    y = t[C.COL_LABEL].to_numpy()
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = []
    for g in sorted(t["campaign_group"].unique()):
        test = np.flatnonzero(t["campaign_group"].to_numpy() == g)
        train = np.flatnonzero(t["campaign_group"].to_numpy() != g)
        tr = downsample(train, y, C.NEG_PER_POS_TRAIN, rng)
        m, thr = fit_with_threshold("hgb", t.iloc[tr][FEATURES], y[tr])
        p = m.predict_proba(t.iloc[test][FEATURES])[:, 1]
        rows.append({"held_out": g, **score_metrics(y[test], p, thr)})
    return pd.DataFrame(rows)


def example_accounts() -> set:
    """Accounts in the example files shipped with the app. They are kept out of training so the
    examples are an honest test on accounts the model never saw."""
    ids = set()
    for f, col in (("simple_sample.csv", "account_id"), ("sample_accounts.csv", C.COL_USER)):
        p = C.ROOT / "models" / f
        if p.exists():
            ids |= set(pd.read_csv(p, usecols=[col])[col].astype("int64"))
    return ids


def train_and_save() -> dict:
    t = training_table()
    ev = evaluate(t)
    held = example_accounts()
    t = t[~t.index.isin(held)]
    y = t[C.COL_LABEL].to_numpy()
    idx = downsample(np.arange(len(t)), y, C.NEG_PER_POS_TRAIN, np.random.default_rng(C.RANDOM_STATE))
    model, thr = fit_with_threshold("hgb", t.iloc[idx][FEATURES], y[idx])
    normal = t.loc[t[C.COL_LABEL] == 0, FEATURES]
    spread = (normal.quantile(.75) - normal.quantile(.25))
    spread = spread.where(spread > 0, normal.quantile(.95) - normal.quantile(.05)).replace(0, 1)
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    meta = {"model": "HistGradientBoostingClassifier (simple inputs)", "features": FEATURES, "threshold": thr,
            "n_train": int(len(idx)), "n_pos": int(y[idx].sum()), "excluded_example_accounts": len(held),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "normal_median": normal.median().round(6).to_dict(), "normal_spread": spread.round(6).to_dict(),
            "loco": ev.round(4).to_dict(orient="records"),
            "loco_mean_pr_auc": float(ev["pr_auc"].mean())}
    META_FILE.write_text(json.dumps(meta, indent=2))
    return meta


class SimpleScorer:
    def __init__(self):
        self.model = joblib.load(MODEL_FILE)
        self.meta = json.loads(META_FILE.read_text())
        self.threshold = self.meta["threshold"]
        self.ns = {"med": pd.Series(self.meta["normal_median"]), "iqr": pd.Series(self.meta["normal_spread"])}

    @staticmethod
    def available() -> bool:
        return MODEL_FILE.exists() and META_FILE.exists()

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feats[FEATURES].astype(float))[:, 1]


if __name__ == "__main__":
    m = train_and_save()
    print(pd.DataFrame(m["loco"])[["held_out", "n_pos", "prevalence", "pr_auc", "roc_auc", "precision_at_100"]]
          .to_string(index=False))
    print("mean PR-AUC", round(m["loco_mean_pr_auc"], 3), "threshold", round(m["threshold"], 3))
