"""The review workflow: build a queue of accounts, explain each one, record decisions.

No Streamlit here. An Investigation is created either from a campaign in the dataset
(scores = held-out predictions, groups = co-reply communities) or from files the user
uploads (scores = the deployed model, groups = co-reply communities of the uploaded replies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from app.core.store import C, PRETTY, Store, short
from radar.data import feature_group
from radar.graph import build_coreply_edges

DECISIONS = {"suspicious": "Suspicious", "normal": "Looks normal", "unsure": "Unsure"}
UPLOAD_MIN_SHARED = 2      # uploads are small, so two shared tweets already link two accounts
GROUP_MIN = 3


@dataclass
class Investigation:
    name: str
    source: str                         # "dataset" | "upload"
    queue: pd.DataFrame                 # index = account id
    groups: dict                        # group id -> list of account ids
    feats: pd.DataFrame                 # features of queued accounts (may be partial)
    edges: pd.DataFrame                 # co-reply links among grouped accounts
    replies: pd.DataFrame | None        # per-reply rows (tweet, user, t, delay_min) if known
    threshold: float
    labels: pd.Series | None = None     # known labels, evaluation only (dataset source)
    all_scores: pd.Series | None = None # score of every account we know (for group views)
    n_scored: int = 0                   # accounts the model scored
    n_above: int = 0                    # of those, above the review threshold
    campaign: str = ""                  # dataset campaign, if any
    notes: dict = field(default_factory=dict)

    # -- decisions -----------------------------------------------------------------------
    def decide(self, uids, decision: str, note: str | None = None) -> None:
        uids = [uids] if not isinstance(uids, (list, tuple, pd.Index, np.ndarray)) else uids
        for u in uids:
            if u in self.queue.index:
                self.queue.at[u, "decision"] = decision
                if note is not None:
                    self.queue.at[u, "note"] = note

    def progress(self) -> tuple[int, int]:
        return int((self.queue["decision"] != "").sum()), len(self.queue)

    def next_open(self, after=None):
        q = self.queue
        open_ = q.index[q["decision"] == ""]
        if not len(open_):
            return None
        if after is not None and after in q.index:
            later = [u for u in open_ if q.index.get_loc(u) > q.index.get_loc(after)]
            if later:
                return later[0]
        return open_[0]

    def decisions_table(self) -> pd.DataFrame:
        q = self.queue.rename_axis(None).reset_index(names="account_id")
        q["decision"] = q["decision"].map(lambda d: DECISIONS.get(d, "Not reviewed"))
        return q[["rank", "account_id", "account", "score", "group", "decision", "note", "reasons"]]


# -- explanations --------------------------------------------------------------------------
GROUP_TEXT = {
    "text similarity": "Its replies are more alike than a typical account's",
    "reply delay": "Its reply timing is unusual compared with typical accounts",
    "profile": "Its profile is unusual: account age, follower ratio or activity",
    "entities": "It uses hashtags, mentions or links unusually often",
    "engagement": "Its replies get unusual numbers of likes or retweets",
}


def reasons(feats_row: pd.Series | None, ns: dict, group: str, group_size: int,
            replies: pd.DataFrame | None = None) -> list[str]:
    """Plain-language reasons, strongest first. Describes the account; it is not the
    model's internal reasoning."""
    out: list[tuple[float, str]] = []
    if feats_row is not None:
        cols = [c for c in ns["med"].index if c in feats_row.index]
        z = ((feats_row[cols].astype(float) - ns["med"][cols]) / ns["iqr"][cols]).abs().clip(upper=50)
        zg = z.groupby([feature_group(c) for c in cols]).mean().dropna().sort_values(ascending=False)
        for g, v in zg.items():
            if v >= 1 and g in GROUP_TEXT:
                txt = GROUP_TEXT[g]
                if g == "text similarity" and "max_cosine" in feats_row:
                    m = feats_row["max_cosine"]
                    txt = ("Some of its replies are near copies of each other" if m >= 0.8 else txt) + f" (similarity {m:.2f} of 1)"
                out.append((v, txt))
    if group:
        out.append((3.0, f"Replies to the same tweets as {group_size - 1} other accounts (group {group})"))
    if replies is not None and len(replies):
        fast = (replies["delay_min"] <= 60).mean()
        if fast >= 0.5 and len(replies) >= 2:
            out.append((2.5, f"{fast:.0%} of its replies came within an hour of the tweet"))
    out.sort(key=lambda x: -x[0])
    return [t for _, t in out[:4]] or ["No single signal stands out; the score comes from several weaker signals"]


# -- builders ------------------------------------------------------------------------------
def from_campaign(store: Store, campaign: str, n: int = 100) -> Investigation:
    cases = store.detect(campaign)
    acc = store.acc[store.acc["campaign_group"] == campaign]
    top = acc["score"].nlargest(n)
    gid_of, groups = {}, {}
    for i, (_, c) in enumerate(cases.iterrows()):
        gid = f"G{i + 1}"
        groups[gid] = list(c["members"])
        for u in c["members"]:
            gid_of[u] = gid
    q = pd.DataFrame({"account": [short(u) for u in top.index], "score": top.values}, index=top.index)
    q["group"] = [gid_of.get(u, "") for u in q.index]
    q["group_size"] = q["group"].map(lambda g: len(groups.get(g, [])))
    q["rank"] = np.arange(1, len(q) + 1)
    q["decision"], q["note"] = "", ""
    feats = store.fx.loc[store.fx.index.intersection(q.index), store.feats]
    rep = store.replies[store.replies[C.COL_USER].isin(q.index)]
    ns = store._normal_stats()
    q["reasons"] = [" · ".join(reasons(feats.loc[u] if u in feats.index else None, ns, q.at[u, "group"],
                                       q.at[u, "group_size"], rep[rep[C.COL_USER] == u])) for u in q.index]
    used = {g: m for g, m in groups.items() if g in set(q["group"])}
    members = {u for m in used.values() for u in m}
    e = store.edges[store.edges["u"].isin(members) & store.edges["v"].isin(members)]
    thr = float(store.thresholds.get(campaign, 0.5))
    return Investigation(PRETTY.get(campaign, campaign), "dataset", q, used, feats, e, rep, thr,
                         labels=store.acc.loc[q.index, C.COL_LABEL],
                         all_scores=store.acc.loc[list(members | set(q.index)), "score"],
                         n_scored=len(acc), n_above=int(acc["flagged"].sum()), campaign=campaign)


def from_upload(scorer, ns: dict, feats_df: pd.DataFrame, replies_df: pd.DataFrame | None,
                name: str, n: int = 100) -> Investigation:
    idcol = next((c for c in feats_df.columns if "id" in c.lower()), None)
    ids = feats_df[idcol].astype(str) if idcol else pd.Series([f"row-{i + 1}" for i in range(len(feats_df))])
    feats = feats_df.set_axis(ids.values)[scorer.features]
    feats = feats[~feats.index.duplicated()]
    p = pd.Series(scorer.score(feats), index=feats.index)
    top = p.nlargest(n)
    groups, e, rep = {}, pd.DataFrame(columns=["u", "v", "shared"]), None
    if replies_df is not None and len(replies_df):
        r = replies_df.rename(columns={replies_df.columns[0]: C.COL_USER, replies_df.columns[1]: C.COL_TWEET})
        r[C.COL_USER] = r[C.COL_USER].astype(str)
        r[C.COL_TYPE] = C.TARGET_TYPE
        e = build_coreply_edges(r, min_tweets=2, min_shared=UPLOAD_MIN_SHARED)
        if len(e):
            g = nx.from_pandas_edgelist(e, "u", "v", "shared")
            comms = nx.community.louvain_communities(g, weight="shared", seed=C.RANDOM_STATE)
            comms = sorted([sorted(c) for c in comms if len(c) >= GROUP_MIN],
                           key=lambda c: -p.reindex(c).mean())
            groups = {f"G{i + 1}": c for i, c in enumerate(comms)}
    gid_of = {u: g for g, m in groups.items() for u in m}
    q = pd.DataFrame({"account": [short(u) if str(u).isdigit() else str(u) for u in top.index],
                      "score": top.values}, index=top.index)
    q["group"] = [gid_of.get(u, "") for u in q.index]
    q["group_size"] = q["group"].map(lambda g: len(groups.get(g, [])))
    q["rank"] = np.arange(1, len(q) + 1)
    q["decision"], q["note"] = "", ""
    q["reasons"] = [" · ".join(reasons(feats.loc[u], ns, q.at[u, "group"], q.at[u, "group_size"])) for u in q.index]
    used = {g: m for g, m in groups.items() if g in set(q["group"])}
    return Investigation(name, "upload", q, used, feats.loc[q.index], e, rep, scorer.threshold, all_scores=p,
                         n_scored=len(p), n_above=int((p >= scorer.threshold).sum()))
