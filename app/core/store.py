"""Data layer for the Radar app. No Streamlit and no UI code here.

Everything shown in the app comes from the pipeline's artifacts (artifacts/*.pkl|csv)
or from the released dataset files. Three kinds of numbers exist, and every function
here says which one it returns (the UI shows the matching tag):

    MODEL    held-out prediction of the trained classifier (radar.pipeline, LOCO)
    FEATURE  a feature released with the dataset (e.g. reply cosine similarity)
    RULE     computed here with a transparent rule (timing, shared targets, graph)

Times: tweet IDs are Twitter "snowflake" IDs, so each ID encodes the posting time to the
millisecond. That is how the timeline gets real timestamps. (The dataset's `diff_min`
column turned out to be the reply delay in *hours*, rounded up — checked against these
timestamps on 96 % of replies.)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from radar import config as C  # noqa: E402
from radar.data import feature_group  # noqa: E402

A = C.ARTIFACTS_DIR
MAIN_SET = "behavior (all released features)"
TWEPOCH_MS = 1288834974657
CASE_MIN_SIZE = 5
LOUVAIN_RESOLUTION = 1.5     # >1 splits the giant co-reply blobs into reviewable cases
SYNC_WINDOW_MIN = 60         # "synchronized" = replied within this many minutes of the post

PRETTY = {
    "saudi_arabia_112019": "Saudi Arabia · Nov 2019",
    "turkey_052020": "Turkey · May 2020",
    "sa_eg_ae_022020": "Saudi Arabia / Egypt / UAE · Feb 2020",
    "serbia_022020": "Serbia · Feb 2020",
    "honduras_022020": "Honduras · Feb 2020",
    "other": "Other operations (pooled)",
}


def snowflake_time(ids) -> pd.Series:
    ids = pd.Series(ids).astype("int64").to_numpy()
    return pd.to_datetime((ids >> 22) + TWEPOCH_MS, unit="ms", utc=True)


def short(uid) -> str:
    return f"acct-{str(uid)[-6:]}"


def short_tweet(tid) -> str:
    return f"tweet-{str(tid)[-6:]}"


def tier(score: float) -> str:
    """Review priority from the model score. A ranking aid, never a verdict."""
    return "HIGH" if score >= 0.7 else "ELEVATED" if score >= 0.4 else "LOW"


# ----------------------------------------------------------------------------------------
@dataclass
class Store:
    info: dict
    acc: pd.DataFrame          # one row per account, indexed by user id
    edges: pd.DataFrame        # u, v, shared
    replies: pd.DataFrame      # target replies with real timestamps
    metrics: pd.DataFrame
    imp: pd.DataFrame | None
    texts: pd.Series           # tweet id -> English text (only some targets)
    thresholds: pd.Series      # campaign_group -> decision threshold (MODEL)
    feats: list[str]
    baseline: pd.Series        # medians over accounts with >= 2 target replies (label-free)
    baseline_cos: pd.Series    # sample of their max_cosine, for the box plot
    _cases: dict = field(default_factory=dict)

    # -- loading -----------------------------------------------------------------------
    @classmethod
    def load(cls) -> "Store":
        info = json.loads((A / "run_info.json").read_text())
        acc = pd.read_pickle(A / "accounts.pkl")
        metrics = pd.read_csv(A / "metrics.csv")
        best = info["best_model"]
        thr = metrics.query("model == @best and feature_set == @MAIN_SET").set_index("held_out")["threshold"]
        acc = acc.set_index(C.COL_USER)
        acc["flagged"] = acc["score"] >= acc["campaign_group"].map(thr)
        acc["rank"] = acc.groupby("campaign_group")["score"].rank(ascending=False, method="min")

        r = pd.read_pickle(A / "replies.pkl")
        r = r.loc[r[C.COL_TYPE] == C.TARGET_TYPE, [C.COL_TWEET, C.COL_USER, C.COL_REPLY]]
        r["t"] = snowflake_time(r[C.COL_REPLY]).to_numpy()
        r["t_post"] = snowflake_time(r[C.COL_TWEET]).to_numpy()
        r["delay_min"] = (r["t"] - r["t_post"]).dt.total_seconds() / 60
        r = r.sort_values([C.COL_TWEET, "t"]).reset_index(drop=True)

        texts = pd.Series(dtype=str)
        p = C.RAW_DIR / C.TEXT_FILE
        if not info["is_sample"] and p.exists():
            t = pd.read_csv(p, usecols=["tweetid", "english_text"])
            t = t.dropna()
            ids = t["tweetid"].astype(str).str.strip()
            ok = ids.str.fullmatch(r"\d{1,19}") & ((ids.str.len() < 19) | (ids < "9223372036854775807"))
            t = t[ok].assign(tweetid=ids[ok].map(int).astype("int64"))
            texts = t.drop_duplicates("tweetid").set_index("tweetid")["english_text"]

        feats = [c for c in info["feature_columns"] if c in acc.columns]
        key = ["max_cosine", "mean_cosine", "mean_num_url", "age", "activity_rate"]
        active = r.groupby(C.COL_USER).size()
        active = acc.index.intersection(active.index[active >= 2])
        imp = pd.read_csv(A / "importance.csv") if (A / "importance.csv").exists() else None
        return cls(info, acc, pd.read_pickle(A / "edges.pkl"), r, metrics, imp, texts, thr,
                   feats, acc.loc[active, key].median(), acc.loc[active, "max_cosine"].sample(20000, random_state=1))

    # -- datasets ------------------------------------------------------------------------
    def campaigns(self) -> pd.DataFrame:
        """Selectable datasets = held-out campaign groups (each was scored by a model that
        never saw it)."""
        g = self.acc.groupby("campaign_group")
        t = self.replies.assign(cg=self.replies[C.COL_USER].map(self.acc["campaign_group"]))
        out = pd.DataFrame({
            "accounts": g.size(),
            "flagged": g["flagged"].sum(),
            "in_graph": g["community"].count(),
            "replies": t.groupby("cg").size(),
            "targets": t.groupby("cg")[C.COL_TWEET].nunique(),
            "first": t.groupby("cg")["t"].min(),
            "last": t.groupby("cg")["t"].max(),
        })
        m = self.metrics.query("model == @self.info['best_model'] and feature_set == @MAIN_SET")
        out = out.join(m.set_index("held_out")[["pr_auc", "precision_at_100", "prevalence"]])
        out["name"] = [PRETTY.get(i, i) for i in out.index]
        return out.sort_values("accounts", ascending=False)

    # -- detection (RULE: graph communities; MODEL: scores) --------------------------------
    def detect(self, campaign: str, progress=None) -> pd.DataFrame:
        """Find co-reply groups inside one held-out campaign and summarise each as a case.

        The grouping and the evidence are computed now; the per-account scores are the
        pipeline's held-out predictions (training a model takes minutes, not seconds)."""
        if campaign in self._cases:
            return self._cases[campaign]
        say = progress or (lambda *_: None)
        members = self.acc.index[self.acc["campaign_group"] == campaign]
        say(0.1, f"Selecting the co-reply links among {len(members):,} accounts")
        e = self.edges[self.edges["u"].isin(members) & self.edges["v"].isin(members)]
        say(0.3, f"Building the co-reply network ({len(e):,} links)")
        g = nx.from_pandas_edgelist(e, "u", "v", "shared")
        say(0.45, "Detecting communities (Louvain)")
        comms = nx.community.louvain_communities(g, weight="shared", seed=C.RANDOM_STATE,
                                                 resolution=LOUVAIN_RESOLUTION)
        comms = [sorted(c) for c in comms if len(c) >= CASE_MIN_SIZE]
        say(0.75, f"Measuring evidence for {len(comms)} groups")
        rows = []
        for c in comms:
            rows.append(self._summarise(c, e))
        cases = pd.DataFrame(rows)
        if cases.empty:
            self._cases[campaign] = cases
            return cases
        # review priority = model score, broken by synchronisation
        cases = cases.sort_values(["score", "sync_share"], ascending=False).reset_index(drop=True)
        cases.insert(0, "case_id", [f"{campaign[:3].upper()}-{i + 1:03d}" for i in range(len(cases))])
        cases["campaign"] = campaign
        self._cases[campaign] = cases
        say(1.0, "Done")
        return cases

    def _summarise(self, ids: list, e: pd.DataFrame) -> dict:
        a = self.acc.loc[ids]
        rep = self.replies[self.replies[C.COL_USER].isin(ids)]
        per_t = rep.groupby(C.COL_TWEET)[C.COL_USER].nunique()
        shared_t = per_t[per_t >= 2]
        sub = e[e["u"].isin(ids) & e["v"].isin(ids)]
        n = len(ids)
        sync = (rep["delay_min"] <= SYNC_WINDOW_MIN).mean() if len(rep) else 0.0
        return {
            "members": ids, "size": n,
            "score": float(a["score"].mean()),                        # MODEL
            "flagged": int(a["flagged"].sum()),                        # MODEL
            "posts": int(len(rep)),
            "targets": int(rep[C.COL_TWEET].nunique()),
            "shared_targets": int(len(shared_t)),                      # RULE
            "density": 2 * len(sub) / (n * (n - 1)) if n > 1 else 0,  # RULE
            "mean_shared": float(sub["shared"].mean()) if len(sub) else 0,
            "sync_share": float(sync),                                 # RULE
            "median_delay_min": float(rep["delay_min"].median()) if len(rep) else np.nan,
            "max_cosine": float(a["max_cosine"].median()),             # FEATURE
            "mean_cosine": float(a["mean_cosine"].median()),           # FEATURE
            "mean_num_url": float(a["mean_num_url"].median()),         # FEATURE
            "first": rep["t"].min(), "last": rep["t"].max(),
            "n_io": int(a[C.COL_LABEL].sum()),                         # ground truth (eval only)
        }

    # -- case detail ---------------------------------------------------------------------
    def title(self, case: pd.Series) -> str:
        """Auto-generated case title from the strongest RULE/FEATURE signal."""
        sim_hi = case["max_cosine"] >= max(0.5, 2 * self.baseline["max_cosine"])
        if case["sync_share"] >= 0.5:
            return "Synchronized reply swarm"
        if sim_hi:
            return "Mirror-text reply cluster"
        if case["density"] >= 0.5:
            return "Tight shared-target ring"
        return "Recurring co-reply network"

    def case_graph(self, case: pd.Series, max_nodes: int = 160, max_links: int = 900) -> dict:
        ids = case["members"]
        a = self.acc.loc[ids]
        e = self.edges[self.edges["u"].isin(ids) & self.edges["v"].isin(ids)]
        strength = pd.concat([e.groupby("u")["shared"].sum(), e.groupby("v")["shared"].sum()]).groupby(level=0).sum()
        keep = strength.reindex(ids).fillna(0).sort_values(ascending=False).index[:max_nodes]
        e = e[e["u"].isin(keep) & e["v"].isin(keep)].nlargest(max_links, "shared")
        return graph_payload(self.acc, {case["case_id"]: list(keep)}, e)

    def case_matrix(self, case: pd.Series, n_acc: int = 24, n_tw: int = 18) -> pd.DataFrame:
        """Accounts x target tweets they replied to (RULE: common targets)."""
        rep = self.replies[self.replies[C.COL_USER].isin(case["members"])]
        top_t = rep.groupby(C.COL_TWEET)[C.COL_USER].nunique().nlargest(n_tw).index
        top_a = rep[rep[C.COL_TWEET].isin(top_t)].groupby(C.COL_USER).size().nlargest(n_acc).index
        m = (rep[rep[C.COL_TWEET].isin(top_t) & rep[C.COL_USER].isin(top_a)]
             .groupby([C.COL_USER, C.COL_TWEET]).size().unstack(fill_value=0))
        return m.reindex(index=top_a, columns=top_t, fill_value=0)

    def case_replies(self, case: pd.Series) -> pd.DataFrame:
        rep = self.replies[self.replies[C.COL_USER].isin(case["members"])].copy()
        rep["score"] = rep[C.COL_USER].map(self.acc["score"])
        rep["account"] = rep[C.COL_USER].map(short)
        rep["tweet"] = rep[C.COL_TWEET].map(short_tweet)
        return rep

    def sync_baseline(self, case: pd.Series) -> tuple[float, float]:
        """Share of replies within SYNC_WINDOW_MIN of the post: case members vs everyone
        else who replied to the same target tweets (RULE)."""
        rep = self.replies
        t = rep.loc[rep[C.COL_USER].isin(case["members"]), C.COL_TWEET].unique()
        on = rep[rep[C.COL_TWEET].isin(t)]
        inside = on[C.COL_USER].isin(case["members"])
        fast = on["delay_min"] <= SYNC_WINDOW_MIN
        return float(fast[inside].mean()), float(fast[~inside].mean()) if (~inside).any() else np.nan

    # -- accounts ------------------------------------------------------------------------
    def account(self, uid) -> dict:
        row = self.acc.loc[uid]
        e = self.edges[(self.edges["u"] == uid) | (self.edges["v"] == uid)]
        other = np.where(e["u"] == uid, e["v"], e["u"])
        nb = (pd.DataFrame({"uid": other, "shared": e["shared"].to_numpy()})
              .nlargest(12, "shared"))
        nb["score"] = nb["uid"].map(self.acc["score"])
        nb["account"] = nb["uid"].map(short)
        rep = self.replies[self.replies[C.COL_USER] == uid].copy()
        rep["tweet"] = rep[C.COL_TWEET].map(short_tweet)
        rep["text"] = rep[C.COL_TWEET].map(self.texts)
        return {"row": row, "neighbors": nb, "replies": rep}

    def unusual(self, uid) -> pd.Series:
        """Robust distance from typical normal accounts, averaged per feature group.
        Describes the account (FEATURE); it is not the model's internal reasoning."""
        normal = self._normal_stats()
        z = (self.acc.loc[uid, self.feats].astype(float) - normal["med"]) / normal["iqr"]
        z = z.clip(-50, 50)
        return (z.abs().rename("z").to_frame().assign(group=[feature_group(f) for f in self.feats])
                .groupby("group")["z"].mean().sort_values())

    def _normal_stats(self):
        if not hasattr(self, "_ns"):
            n = self.acc.loc[self.acc[C.COL_LABEL] == 0, self.feats]
            spread = (n.quantile(.75) - n.quantile(.25))
            spread = spread.where(spread > 0, n.quantile(.95) - n.quantile(.05))
            spread = spread.where(spread > 0, n.std()).replace(0, np.nan)
            self._ns = {"med": n.median(), "iqr": spread}
        return self._ns

    # -- target tweets -------------------------------------------------------------------
    def tweets(self, campaign: str | None = None) -> pd.DataFrame:
        if not hasattr(self, "_tw"):
            r = self.replies.assign(score=self.replies[C.COL_USER].map(self.acc["score"]),
                                    flag=self.replies[C.COL_USER].map(self.acc["flagged"]),
                                    cg=self.replies[C.COL_USER].map(self.acc["campaign_group"]))
            tw = r.groupby(C.COL_TWEET).agg(replies=("t", "size"), flagged=("flag", "sum"),
                                            posted=("t_post", "first"), first=("t", "min"),
                                            campaign=("cg", lambda s: s.mode().iat[0]))
            tw["text"] = tw.index.map(self.texts)
            tw["tweet"] = tw.index.map(short_tweet)
            self._tw = tw.sort_values("flagged", ascending=False)
        tw = self._tw
        return tw if campaign is None else tw[tw["campaign"] == campaign]

    def tweet_replies(self, tid) -> pd.DataFrame:
        r = self.replies[self.replies[C.COL_TWEET] == tid].copy()
        r["score"] = r[C.COL_USER].map(self.acc["score"])
        r["flagged"] = r[C.COL_USER].map(self.acc["flagged"])
        r["account"] = r[C.COL_USER].map(short)
        return r


# ----------------------------------------------------------------------------------------
def graph_payload(acc: pd.DataFrame, groups: dict[str, list], e: pd.DataFrame,
                  spread: float = 1.0) -> dict:
    """Nodes + links + precomputed positions for the canvas network component.

    Groups are laid out on a sunflower spiral; inside each group a spring layout.
    Nothing here is random decoration: every node is an account, every link a pair of
    accounts that replied to >= 3 of the same target tweets."""
    nodes, links = [], []
    pos_all = {}
    order = list(groups)
    for gi, gid in enumerate(order):
        sub = e[e["u"].isin(groups[gid]) & e["v"].isin(groups[gid])]
        linked = set(sub["u"]) | set(sub["v"])
        if linked:
            groups[gid] = [u for u in groups[gid] if u in linked]
        ids = groups[gid]
        g = nx.Graph()
        g.add_nodes_from(ids)
        g.add_weighted_edges_from(sub[["u", "v", "shared"]].itertuples(index=False))
        p = nx.spring_layout(g, seed=7, weight="weight", iterations=60) if len(ids) > 1 else {ids[0]: np.zeros(2)}
        r = 0.26 + 0.06 * np.sqrt(len(ids))
        ang = gi * 2.39996
        cx, cy = (np.cos(ang) * np.sqrt(gi) * 0.9 * spread, np.sin(ang) * np.sqrt(gi) * 0.9 * spread) if len(order) > 1 else (0, 0)
        for uid, (x, y) in p.items():
            pos_all[uid] = (cx + x * r, cy + y * r)
    deg = pd.concat([e["u"], e["v"]]).value_counts()
    for gid in order:
        for uid in groups[gid]:
            x, y = pos_all[uid]
            row = acc.loc[uid]
            nodes.append({"id": str(uid), "label": short(uid), "group": gid,
                          "x": round(float(x), 4), "y": round(float(y), 4),
                          "score": round(float(row["score"]), 3), "flag": bool(row["flagged"]),
                          "deg": int(deg.get(uid, 0))})
    for u, v, s in e[["u", "v", "shared"]].itertuples(index=False):
        if u in pos_all and v in pos_all:
            links.append([str(u), str(v), int(s)])
    return {"nodes": nodes, "links": links, "groups": order}
