"""Co-reply network: two accounts are linked when they replied to the same target tweets.

Built from reply structure only. No labels are used, so the graph features below can be
given to the model without leaking the answer.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C


def build_coreply_edges(
    replies: pd.DataFrame,
    min_tweets: int = C.GRAPH_MIN_TWEETS,
    min_shared: int = C.GRAPH_MIN_SHARED,
    max_repliers_per_tweet: int = C.GRAPH_MAX_REPLIERS_PER_TWEET,
    tweet_type: str = C.TARGET_TYPE,
) -> pd.DataFrame:
    """Return edges (u, v, shared) between repliers that share >= min_shared tweets."""
    r = replies.loc[replies[C.COL_TYPE] == tweet_type, [C.COL_TWEET, C.COL_USER]].drop_duplicates()
    r = r[r.groupby(C.COL_TWEET)[C.COL_USER].transform("size") <= max_repliers_per_tweet]
    r = r[r.groupby(C.COL_USER)[C.COL_TWEET].transform("size") >= min_tweets]
    if r.empty:
        return pd.DataFrame(columns=["u", "v", "shared"])

    u_codes, users = pd.factorize(r[C.COL_USER])
    t_codes, _ = pd.factorize(r[C.COL_TWEET])
    b = sparse.csr_matrix(
        (np.ones(len(r), dtype=np.int32), (u_codes, t_codes)),
        shape=(len(users), t_codes.max() + 1),
    )
    a = sparse.triu(b @ b.T, k=1).tocoo()
    keep = a.data >= min_shared
    return pd.DataFrame({
        "u": users[a.row[keep]],
        "v": users[a.col[keep]],
        "shared": a.data[keep].astype(int),
    })


def to_graph(edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_weighted_edges_from(edges[["u", "v", "shared"]].itertuples(index=False, name=None))
    return g


def communities(g: nx.Graph, seed: int = C.RANDOM_STATE) -> dict:
    """Louvain communities -> {user: community_id}, ids ordered by community size."""
    comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    return {node: i for i, comm in enumerate(comms) for node in comm}


def graph_features(g: nx.Graph, membership: dict, users: pd.Series) -> pd.DataFrame:
    """Label-free per-account network features. Accounts outside the graph get zeros."""
    comm_size = pd.Series(membership).map(pd.Series(membership).value_counts())
    feats = pd.DataFrame({
        "graph_degree": pd.Series(dict(g.degree())),
        "graph_strength": pd.Series(dict(g.degree(weight="weight"))),
        "graph_community_size": comm_size,
    })
    feats.index.name = C.COL_USER
    out = pd.DataFrame({C.COL_USER: users.unique()}).merge(
        feats.reset_index(), on=C.COL_USER, how="left"
    )
    return out.fillna(0)
