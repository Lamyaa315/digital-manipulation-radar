"""Read a plain replies file, turn it into model features, and build a review queue.

Accepted columns (case-insensitive, common X/Twitter export names work too):
    account_id   author_id, user_id, account, username            required
    tweet_id     in_reply_to_tweet_id, target_tweet_id, tweet     required
    reply_time   created_at, time, timestamp                      required unless reply_id is a tweet ID
    reply_id     reply_tweet_id, id
    tweet_time
    account_created_at   user_created_at
    followers    followers_count
    following    following_count, friends_count
    reply_text   text
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.review import GROUP_MIN, UPLOAD_MIN_SHARED, Investigation
from app.core.store import C, short
from radar import simple as S
from radar.graph import build_coreply_edges

ALIASES = {
    "account": ["account_id", "author_id", "user_id", "userid", "account", "user", "username", "user_name",
                "screen_name", "handle", "author", "replier_userid", "replier_id"],
    "tweet": ["tweet_id", "in_reply_to_tweet_id", "in_reply_to_status_id", "in_reply_to_id", "target_tweet_id",
              "replied_to_id", "conversation_id", "parent_id", "tweet", "poster_tweetid", "original_tweet_id"],
    "reply_time": ["reply_time", "reply_created_at", "reply_date", "created_at", "posted_at", "datetime", "date_time",
                   "timestamp", "timestamp_utc", "time_utc", "time", "date"],
    "reply_id": ["reply_id", "reply_tweet_id", "id", "tweet_id_reply", "replier_tweetid"],
    "tweet_time": ["tweet_time", "tweet_created_at", "tweet_date", "target_time", "target_created_at",
                   "original_tweet_time", "parent_created_at"],
    "created": ["account_created_at", "user_created_at", "author_created_at", "account_created", "joined"],
    "followers": ["followers", "followers_count", "follower_count"],
    "following": ["following", "following_count", "friends_count"],
    "text": ["reply_text", "text", "full_text", "content", "body"],
}
TEMPLATE_COLS = ["account_id", "tweet_id", "reply_time", "account_created_at", "followers", "following", "reply_text"]


def _norm(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def _pick(df: pd.DataFrame, key: str, exclude: set | None = None):
    low = {_norm(c): c for c in df.columns if not exclude or c not in exclude}
    for a in ALIASES[key]:
        if a in low:
            return df[low[a]]
    return None


def _ms(series) -> pd.Series:
    """Times in any common form -> epoch milliseconds. Numbers are read as epoch seconds or
    milliseconds by their size; text is parsed per value (mixed formats allowed), day-first
    tried when that reads more of the values."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.where(num > 1e11, num * 1000)          # seconds -> ms
    s = series.astype(str).str.strip()
    a = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    b = pd.to_datetime(s, utc=True, errors="coerce", format="mixed", dayfirst=True)
    t = b if b.notna().sum() > a.notna().sum() else a
    return pd.Series((t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() * 1000, index=series.index)


def _snow(series) -> pd.Series:
    """Real X/Twitter IDs carry their posting time (snowflake)."""
    n = pd.to_numeric(series, errors="coerce")
    ok = (n > 2 ** 40) & (n < 2 ** 63)
    out = pd.Series(np.nan, index=series.index)
    out[ok] = (n[ok].astype("int64").to_numpy() >> 22) + S.TWEPOCH_MS
    return out


def parse(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """-> (normalised replies, errors, notes)."""
    errors, notes = [], []
    if df.dropna(how="all").empty:
        return None, ["The file has column names but no rows. Add one row per reply "
                      "(download the example file to see what goes in it)."], notes
    found = ", ".join(f"`{c}`" for c in df.columns[:15]) or "none"
    acc, tw = _pick(df, "account"), _pick(df, "tweet")
    if acc is None:
        errors.append("No account column (account_id).")
    if tw is None:
        errors.append("No tweet column (tweet_id — the tweet being replied to).")
    if errors:
        return None, errors + [f"Columns found in your file: {found}"], notes
    r = pd.DataFrame({"account": acc.astype(str).str.strip(), "tweet": tw.astype(str).str.strip()})
    used = {acc.name, tw.name}
    rt, rid = _pick(df, "reply_time", used), _pick(df, "reply_id", used)
    low = {_norm(c): c for c in df.columns}
    if "date" in low and "time" in low:                    # separate date and time columns
        r["reply_ms"] = _ms(df[low["date"]].astype(str) + " " + df[low["time"]].astype(str))
    else:
        r["reply_ms"] = _ms(rt) if rt is not None else np.nan
    if rid is not None:
        r["reply_ms"] = r["reply_ms"].fillna(_snow(rid))
    tt = _pick(df, "tweet_time", used)
    r["tweet_ms"] = _ms(tt) if tt is not None else np.nan
    r["tweet_ms"] = r["tweet_ms"].fillna(_snow(tw))
    if r["reply_ms"].isna().all():
        errors.append("Could not find when each reply was posted. Add a `reply_time` column "
                      "(e.g. 2025-11-14 10:30), or a `reply_id` column with real X reply IDs.")
        errors.append(f"Columns found in your file: {found}")
        return None, errors, notes
    if r["tweet_ms"].isna().all():
        first = r.groupby("tweet")["reply_ms"].transform("min")
        r["tweet_ms"] = first
        notes.append("No posting time for the replied-to tweets (add `tweet_time`, or use real X tweet IDs). "
                     "Each tweet's first reply is used instead, so reply delays are approximate.")
    bad = r["reply_ms"].isna() | r["tweet_ms"].isna()
    if bad.any():
        notes.append(f"{int(bad.sum())} of {len(r)} rows have no readable time and were skipped.")
        r = r[~bad]
    if r.empty:
        return None, ["No row has a readable time."], notes
    cr = _pick(df, "created")
    created = _ms(cr) if cr is not None else pd.Series(np.nan, index=df.index)
    r["created_ms"] = created.reindex(r.index).fillna(_snow(r["account"]))
    for k in ("followers", "following"):
        v = _pick(df, k)
        r[k] = pd.to_numeric(v, errors="coerce").reindex(r.index) if v is not None else np.nan
    txt = _pick(df, "text")
    r["text"] = txt.reindex(r.index).fillna("").astype(str) if txt is not None else ""
    missing = [k for k in ("created_ms", "followers", "following") if r[k].isna().all()]
    if missing:
        notes.append("Without " + ", ".join({"created_ms": "account creation dates", "followers": "follower counts",
                                             "following": "following counts"}[m] for m in missing)
                     + " the model has less to go on.")
    return r.reset_index(drop=True), errors, notes


def near_copies(r: pd.DataFrame) -> pd.Series:
    """Rule (not the model): highest text similarity between an account's reply and another
    account's reply to the same tweet (TF-IDF on character n-grams, cosine)."""
    if not r["text"].str.strip().any():
        return pd.Series(dtype=float)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    best = pd.Series(0.0, index=r.index)
    for _, g in r[r["text"].str.len() > 15].groupby("tweet"):
        if g["account"].nunique() < 2 or len(g) > 3000:
            continue
        m = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(g["text"])
        s = cosine_similarity(m)
        same = g["account"].to_numpy()[:, None] == g["account"].to_numpy()[None, :]
        s[same] = 0
        best.loc[g.index] = s.max(axis=1)
    return best.groupby(r["account"]).max()


def reasons(row: pd.Series, ns: dict, group: str, size: int, sim: float | None, fast: float) -> list[str]:
    out = []
    if sim is not None and sim >= 0.8:
        out.append((5, f"Posted a near-copy of another account's reply on the same tweet (text similarity {sim:.2f})"))
    z = ((row[ns["med"].index] - ns["med"]) / ns["iqr"]).abs()
    if fast >= 0.5 and row["n_replies"] >= 2:
        out.append((4, f"{fast:.0%} of its replies came within an hour of the tweet"))
    if row["n_tweets"] >= 3 and z.get("n_tweets", 0) >= 2:
        out.append((3, f"Replied to {int(row['n_tweets'])} of the tweets under investigation"))
    if pd.notna(row["age_years"]) and row["age_years"] < 0.25:
        out.append((3, f"Account was created only {row['age_years'] * 365:.0f} days before its first reply"))
    if group:
        out.append((3.5, f"Replies to the same tweets as {size - 1} other accounts (group {group})"))
    if z.get("log_followers", 0) >= 2 or z.get("log_following", 0) >= 2:
        out.append((2, "Its follower or following count is far from typical accounts"))
    out.sort(key=lambda x: -x[0])
    return [t for _, t in out[:4]] or ["No single signal stands out; the score comes from several weaker signals"]


def build(r: pd.DataFrame, scorer, name: str, n: int = 100, labels: pd.Series | None = None) -> Investigation:
    feats = S.features(r.rename(columns={}))
    p = pd.Series(scorer.score(feats), index=feats.index)
    top = p.nlargest(n)
    sims = near_copies(r)
    r = r.assign(delay_min=(r["reply_ms"] - r["tweet_ms"]) / 60000)
    fast = (r["delay_min"] <= 60).groupby(r["account"]).mean()
    e = build_coreply_edges(pd.DataFrame({C.COL_USER: r["account"], C.COL_TWEET: r["tweet"], C.COL_TYPE: C.TARGET_TYPE}),
                            min_tweets=2, min_shared=UPLOAD_MIN_SHARED)
    groups = {}
    if len(e):
        import networkx as nx
        g = nx.from_pandas_edgelist(e, "u", "v", "shared")
        comms = [sorted(c) for c in nx.community.louvain_communities(g, weight="shared", seed=C.RANDOM_STATE)
                 if len(c) >= GROUP_MIN]
        comms.sort(key=lambda c: -p.reindex(c).mean())
        groups = {f"G{i + 1}": c for i, c in enumerate(comms)}
    gid = {u: g for g, m in groups.items() for u in m}
    q = pd.DataFrame({"account": [short(u) if u.isdigit() else u for u in top.index], "score": top.values},
                     index=top.index)
    q["group"] = [gid.get(u, "") for u in q.index]
    q["group_size"] = q["group"].map(lambda x: len(groups.get(x, [])))
    q["rank"] = np.arange(1, len(q) + 1)
    q["decision"], q["note"] = "", ""
    q["reasons"] = [" · ".join(reasons(feats.loc[u], scorer.ns, q.at[u, "group"], q.at[u, "group_size"],
                                       sims.get(u), fast.get(u, 0))) for u in q.index]
    used = {g: m for g, m in groups.items() if g in set(q["group"])}
    rep = pd.DataFrame({C.COL_USER: r["account"], C.COL_TWEET: r["tweet"],
                        "t": pd.to_datetime(r["reply_ms"], unit="ms", utc=True), "delay_min": r["delay_min"]})
    lab = labels.reindex(q.index) if labels is not None else None
    return Investigation(name, "simple", q, used, feats.loc[q.index], e, rep, scorer.threshold,
                         labels=lab if lab is not None and lab.notna().all() else None, all_scores=p,
                         n_scored=len(p), n_above=int((p >= scorer.threshold).sum()))
