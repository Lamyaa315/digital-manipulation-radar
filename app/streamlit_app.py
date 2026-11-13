"""Digital Manipulation Radar — dashboard.

Reads only from artifacts/ (written by `python -m radar.pipeline`). No training here.
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar import config as C  # noqa: E402
from radar.data import feature_group  # noqa: E402

A = C.ARTIFACTS_DIR
st.set_page_config(page_title="Digital Manipulation Radar", layout="wide")


@st.cache_resource
def load(stamp: float):  # stamp = run_info.json mtime, so a new pipeline run reloads

    info = json.loads((A / "run_info.json").read_text())
    acc = pd.read_pickle(A / "accounts.pkl")
    edges = pd.read_pickle(A / "edges.pkl")
    replies = pd.read_pickle(A / "replies.pkl")
    metrics = pd.read_csv(A / "metrics.csv")
    imp = pd.read_csv(A / "importance.csv") if (A / "importance.csv").exists() else None
    comm = pd.read_csv(A / "communities.csv")
    delays = pd.read_pickle(A / "delays.pkl") if (A / "delays.pkl").exists() else None
    return info, acc, edges, replies, metrics, imp, comm, delays


def short(uid) -> str:
    """Never show full account IDs next to a score."""
    return f"acct-{str(uid)[-6:]}"


def short_tweet(tid) -> str:
    return f"tweet-{str(tid)[-6:]}"


@st.cache_resource
def prepare(stamp: float):
    """Derived tables shared by several pages (cached per pipeline run)."""
    info, acc, edges, replies, metrics, imp, comm, delays = load(stamp)
    best, main = info["best_model"], "behavior (all released features)"
    thr = metrics.query("model == @best and feature_set == @main").set_index("held_out")["threshold"]
    acc = acc.assign(
        account=acc[C.COL_USER].map(short),
        flagged=acc["score"] >= acc["campaign_group"].map(thr),
        rank_in_campaign=acc.groupby("campaign_group")["score"].rank(ascending=False, method="min"),
        campaign_size=acc.groupby("campaign_group")["score"].transform("size"),
    )
    targets = replies[replies[C.COL_TYPE] == C.TARGET_TYPE]
    io = targets[targets[C.COL_LABEL] == 1]
    tcamp = io.groupby(C.COL_TWEET)[C.COL_CAMPAIGN].agg(lambda s: s.mode().iat[0])
    flagged_ids = set(acc.loc[acc["flagged"], C.COL_USER])
    tweets = targets.groupby(C.COL_TWEET).agg(
        replies=(C.COL_USER, "size"), repliers=(C.COL_USER, "nunique"),
        flagged=(C.COL_USER, lambda s: s.drop_duplicates().isin(flagged_ids).sum()),
    ).assign(campaign=lambda t: t.index.map(tcamp)).reset_index()
    tweets["tweet"] = tweets[C.COL_TWEET].map(short_tweet)
    text_path = C.RAW_DIR / C.TEXT_FILE
    if not info["is_sample"] and text_path.exists():
        txt = pd.read_csv(text_path, usecols=["tweetid", "english_text"], dtype={"tweetid": str})
        txt = txt.dropna().drop_duplicates("tweetid").set_index("tweetid")["english_text"]
        tweets["text"] = tweets[C.COL_TWEET].astype(str).map(txt)
    else:
        tweets["text"] = np.nan
    return info, acc, edges, replies, metrics, imp, comm, delays, tweets


def why_flagged(acc: pd.DataFrame, row: pd.Series, feats: list[str]) -> None:
    """Bar chart: how far the account is from typical normal accounts, per feature group."""
    normal = acc.loc[acc[C.COL_LABEL] == 0, feats]
    spread = (normal.quantile(0.75) - normal.quantile(0.25)).replace(0, np.nan)
    z = (row[feats].astype(float) - normal.median()) / spread
    zg = (z.rename("robust_z").to_frame().assign(group=[feature_group(f) for f in feats])
            .groupby("group")["robust_z"].apply(lambda s: s.abs().mean()).sort_values())
    st.plotly_chart(px.bar(zg, orientation="h",
                           labels={"value": "average distance from normal repliers (robust z)",
                                   "group": ""}), use_container_width=True)
    st.caption("How unusual this account is compared with typical accounts, by feature group. "
               "This describes the account; it is not the model's internal reasoning.")


def go_account(uid) -> None:
    st.session_state["page"] = "Account lookup"
    st.session_state["acct_q"] = str(uid)


def go_tweet(tid) -> None:
    st.session_state["page"] = "Attacked tweet"
    st.session_state["tweet_sel"] = int(tid)


if not (A / "run_info.json").exists():
    st.error("No artifacts yet. Run `python -m radar.pipeline` (or `--sample`) first.")
    st.stop()

info, acc, edges, replies, metrics, imp, comm, delays, tweets = prepare(
    (A / "run_info.json").stat().st_mtime)

st.sidebar.title("Digital Manipulation Radar")
page = st.sidebar.radio("Page", ["Overview", "Model evaluation", "Coordination network",
                                 "Group & account investigation", "Account lookup",
                                 "Attacked tweet"], key="page")
st.sidebar.caption(f"Data: {info['data']}\n\nRun: {info['run_at']}")
if info["is_sample"]:
    st.warning("SYNTHETIC SAMPLE DATA — these numbers test the code and mean nothing.")
st.sidebar.info("Scores mark **possible coordination for human review**. "
                "They are not proof of fake accounts or malicious intent.")

best = info["best_model"]
main_set = "behavior (all released features)"

# ------------------------------------------------------------------------------------
if page == "Overview":
    st.header("Overview")
    def compact(n: int) -> str:
        return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,}"

    c = st.columns(5)
    c[0].metric("Accounts analyzed", compact(info["n_accounts"]))
    c[1].metric("Replies analyzed", compact(info["n_replies"]))
    c[2].metric("Target tweets", compact(info["n_target_tweets"]))
    c[3].metric("Co-reply groups", compact(acc["community"].nunique()))
    c[4].metric("Known IO accounts", compact(info["n_io"]),
                f"{info['n_io'] / info['n_accounts']:.2%} of accounts", delta_color="off")

    st.subheader(f"Can we catch a campaign we never trained on? ({best}, leave-one-campaign-out)")
    m = metrics.query("model == @best and feature_set == @main_set")
    m = m.assign(lift=m["pr_auc"] / m["prevalence"])
    st.dataframe(m[["held_out", "n_test", "n_pos", "prevalence", "pr_auc", "lift", "roc_auc",
                    "precision_at_100", "precision", "recall"]].round(3),
                 hide_index=True, use_container_width=True)
    st.caption("PR-AUC is measured at the real class ratio of the held-out campaign. "
               "'lift' = PR-AUC ÷ prevalence (1.0 = random ranking).")

# ------------------------------------------------------------------------------------
elif page == "Model evaluation":
    st.header("Model evaluation")
    held = st.selectbox("Held-out campaign", sorted(metrics["held_out"].dropna().unique()))

    st.subheader("Models (all released behavioral features)")
    mm = metrics.query("feature_set == @main_set")
    fig = px.bar(mm, x="held_out", y="pr_auc", color="model", barmode="group",
                 labels={"pr_auc": "PR-AUC", "held_out": "held-out campaign"})
    for _, r in mm.drop_duplicates("held_out").iterrows():
        fig.add_shape(type="line", x0=r["held_out"], x1=r["held_out"], y0=0, y1=r["prevalence"],
                      line=dict(width=24, color="rgba(0,0,0,0.25)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey stub = prevalence, the PR-AUC of a random ranking.")

    st.subheader(f"Where does the signal come from? (ablation, {best})")
    ab = metrics.query("model == @best").groupby("feature_set")[["pr_auc", "roc_auc"]].mean()
    st.dataframe(ab.round(3), use_container_width=True)
    if "network only" in ab.index and ab.loc["network only", "pr_auc"] > 0.9:
        st.error("Network-only features score almost perfectly. Treat this as a warning sign: "
                 "IO and normal accounts were observed differently when the dataset was built "
                 "(see docs/PLAN.md, Risks). It is not evidence that the graph detects coordination.")

    rep = info["replication"]
    st.info(f"Paper-style setup ({rep['setup']}, {rep['model']}): ROC-AUC {rep['roc_auc']:.3f}, "
            f"F1 {rep['f1']:.3f}. Balanced data hides the real class ratio, so this is for "
            "comparison with the paper only.")

    row = metrics.query("held_out == @held and model == @best and feature_set == @main_set")
    if not row.empty:
        r = row.iloc[0]
        st.subheader(f"Confusion matrix — {held}")
        cm = pd.DataFrame([[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                          index=["actual normal", "actual IO"],
                          columns=["predicted normal", "predicted IO"]).astype(int)
        st.dataframe(cm)
        st.caption(f"Threshold {r['threshold']:.3f}, chosen on training campaigns only.")

    if imp is not None:
        st.subheader("Which feature groups matter (permutation, drop in PR-AUC)")
        gi = imp.groupby("group")["pr_auc_drop"].agg(["mean", "std"]).reset_index()
        st.plotly_chart(px.bar(gi.sort_values("mean"), x="mean", y="group", error_x="std",
                               orientation="h", labels={"mean": "drop in PR-AUC"}),
                        use_container_width=True)

# ------------------------------------------------------------------------------------
elif page == "Coordination network":
    st.header("Co-reply network")
    st.caption(f"Accounts are linked when they replied to ≥ {info['graph']['min_shared']} of the "
               "same target tweets. Colour = model score from a model that never saw this campaign.")
    camp = st.selectbox("Campaign", sorted(acc["campaign_group"].unique()))
    max_nodes = st.slider("Max accounts shown", 50, 500, 250, 50)
    show_truth = st.checkbox("Outline known IO accounts (evaluation only)")

    nodes = acc[(acc["campaign_group"] == camp) & acc["community"].notna()]
    nodes = nodes.nlargest(max_nodes, "graph_strength")
    ids = set(nodes[C.COL_USER])
    e = edges[edges["u"].isin(ids) & edges["v"].isin(ids)]
    if e.empty:
        st.info("No co-reply edges for this campaign with the current settings.")
    else:
        g = nx.Graph()
        g.add_weighted_edges_from(e[["u", "v", "shared"]].itertuples(index=False, name=None))
        pos = nx.spring_layout(g, seed=1, weight="weight")
        ex, ey = [], []
        for u, v in g.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        nd = nodes.set_index(C.COL_USER).loc[list(g.nodes())]
        fig = go.Figure([
            go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="#bbb"), hoverinfo="skip"),
            go.Scatter(
                x=[pos[n][0] for n in g.nodes()], y=[pos[n][1] for n in g.nodes()], mode="markers",
                marker=dict(size=np.clip(np.sqrt(nd["graph_degree"]) * 3, 5, 30), color=nd["score"],
                            colorscale="Reds", cmin=0, cmax=1, showscale=True,
                            colorbar=dict(title="score"),
                            line=dict(width=np.where(nd[C.COL_LABEL] == 1, 2, 0) if show_truth else 0,
                                      color="black")),
                text=[f"{r.account}<br>group {int(r.community)}<br>score {r.score:.2f}<br>"
                      f"co-repliers {int(r.graph_degree)}" for r in nd.itertuples()],
                hoverinfo="text"),
        ])
        fig.update_layout(showlegend=False, height=650, xaxis=dict(visible=False),
                          yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------
elif page == "Group & account investigation":
    st.header("Group & account investigation")
    show_truth = st.checkbox("Show known labels (evaluation only)")
    cols = ["community", "size", "mean_score", "main_campaign"] + (["n_io"] if show_truth else [])
    st.dataframe(comm[cols].round(3), hide_index=True, use_container_width=True)

    gid = st.selectbox("Group", comm["community"].astype(int).tolist())
    members = acc[acc["community"] == gid].sort_values("score", ascending=False)
    targets = replies[replies[C.COL_USER].isin(members[C.COL_USER])
                      & (replies[C.COL_TYPE] == C.TARGET_TYPE)]
    per_tweet = targets.groupby(C.COL_TWEET)[C.COL_USER].nunique().sort_values(ascending=False)

    c = st.columns(3)
    c[0].metric("Accounts", len(members))
    c[1].metric("Target tweets they replied to", f"{len(per_tweet):,}")
    c[2].metric("Most shared tweet: group members replying", int(per_tweet.iloc[0]) if len(per_tweet) else 0)

    show = ["account", "score", "graph_degree", "graph_strength", "n_campaigns"] + \
           ([C.COL_LABEL] if show_truth else [])
    st.dataframe(members[show].round(3), hide_index=True, use_container_width=True)

    if delays is not None and C.COL_REPLY in replies.columns:
        d = replies[replies[C.COL_USER].isin(members[C.COL_USER])].merge(delays, on=C.COL_REPLY)
        if not d.empty:
            st.subheader("How fast did they reply? (hours after the target tweet, rounded up; IO replies only)")
            st.plotly_chart(px.histogram(d, x=C.COL_DELAY, nbins=50, log_x=False),
                            use_container_width=True)

    st.subheader("Why was this account flagged?")
    acct = st.selectbox("Account", members["account"].tolist())
    row = members.loc[members["account"] == acct].iloc[0]
    why_flagged(acc, row, info["feature_columns"])
    st.button("Open this account in Account lookup", on_click=go_account,
              args=(row[C.COL_USER],))

# ------------------------------------------------------------------------------------
elif page == "Account lookup":
    st.header("Account lookup")
    st.caption("Only accounts that are in the dataset (Twitter, 2018–2021) can be looked up. "
               "A new account cannot be scored: its features need its replies to attacked "
               "tweets and the other replies to those same tweets.")
    show_truth = st.checkbox("Show known label (evaluation only)", key="acct_truth")

    def pick(df: pd.DataFrame) -> None:
        st.session_state["acct_q"] = str(df.sample(1)[C.COL_USER].iat[0])

    b = st.columns(3)
    b[0].button("Example: high score", on_click=pick, args=(acc.nlargest(200, "score"),))
    b[1].button("Example: flagged but normal (false alarm)",
                on_click=pick, args=(acc[acc["flagged"] & (acc[C.COL_LABEL] == 0)],))
    b[2].button("Example: random account", on_click=pick, args=(acc,))
    q = st.text_input("Account ID (full number) or short code (acct-123456)", key="acct_q").strip()

    if q:
        if q.lower().startswith("acct-"):
            found = acc[acc["account"] == q.lower()]
        elif q.isdigit():
            found = acc[acc[C.COL_USER].astype(str) == q]
        else:
            found = acc.iloc[0:0]
        if found.empty:
            st.warning("This account is not in the dataset, so it cannot be scored.")
        else:
            if len(found) > 1:  # short codes can collide
                idx = st.selectbox("Several accounts share this short code", found.index,
                                   format_func=lambda i: f"{found.at[i, 'account']} · "
                                                         f"{found.at[i, 'campaign']}")
                r = found.loc[idx]
            else:
                r = found.iloc[0]
            uid = r[C.COL_USER]

            c = st.columns(4)
            c[0].metric("Model score", f"{r['score']:.2f}")
            c[1].metric("Rank in its campaign",
                        f"#{int(r['rank_in_campaign']):,} of {int(r['campaign_size']):,}",
                        f"top {r['rank_in_campaign'] / r['campaign_size']:.2%}", delta_color="off")
            c[2].metric("Flagged for review", "Yes" if r["flagged"] else "No")
            c[3].metric("Co-repliers", int(r["graph_degree"]))
            st.write(f"**{r['account']}** · campaign of the tweets it replied to: "
                     f"`{r['campaign']}` · group "
                     f"{'—' if pd.isna(r['community']) else int(r['community'])}"
                     + (f" · **known label: {'IO account' if r[C.COL_LABEL] == 1 else 'normal'}**"
                        if show_truth else ""))
            st.caption("The score comes from a model that never saw this account's campaign. "
                       "'Flagged' uses the threshold chosen on the other campaigns.")

            left, right = st.columns(2)
            with left:
                st.subheader("Attacked tweets it replied to")
                mine = replies[(replies[C.COL_USER] == uid)
                               & (replies[C.COL_TYPE] == C.TARGET_TYPE)][C.COL_TWEET].unique()
                t = tweets[tweets[C.COL_TWEET].isin(mine)].sort_values("flagged", ascending=False)
                st.dataframe(t[["tweet", "campaign", "repliers", "flagged"]], hide_index=True,
                             use_container_width=True)
                if len(t):
                    tsel = st.selectbox("Tweet", t[C.COL_TWEET],
                                        format_func=lambda x: short_tweet(x))
                    st.button("Open in Attacked tweet", on_click=go_tweet, args=(tsel,))
            with right:
                st.subheader("Strongest co-reply links")
                e = edges[(edges["u"] == uid) | (edges["v"] == uid)]
                if e.empty:
                    st.info("No account shares 3 or more attacked tweets with this one.")
                else:
                    e = e.assign(other=np.where(e["u"] == uid, e["v"], e["u"])).nlargest(15, "shared")
                    nb = e.merge(acc[[C.COL_USER, "account", "score", C.COL_LABEL]],
                                 left_on="other", right_on=C.COL_USER)
                    cols = ["account", "shared", "score"] + ([C.COL_LABEL] if show_truth else [])
                    st.dataframe(nb[cols].rename(columns={"shared": "shared tweets"}).round(2),
                                 hide_index=True, use_container_width=True)

            st.subheader("Why was it scored this way?")
            why_flagged(acc, r, info["feature_columns"])

# ------------------------------------------------------------------------------------
elif page == "Attacked tweet":
    st.header("Attacked tweet: who replied, and who looks coordinated?")
    st.caption("Target tweets are tweets that received 5 or more replies from IO accounts. "
               "Tweet IDs are shortened because the authors are real people "
               "(journalists, officials, media).")
    show_truth = st.checkbox("Show known labels (evaluation only)", key="tweet_truth")
    fc = st.columns([2, 1])
    camp = fc[0].selectbox("Campaign", ["all"] + sorted(tweets["campaign"].dropna().unique()))
    with_text = fc[1].checkbox(f"Only tweets with text ({tweets['text'].notna().sum():,})")
    pool = tweets if camp == "all" else tweets[tweets["campaign"] == camp]
    if with_text:
        pool = pool[pool["text"].notna()]
    if pool.empty:
        st.info("No tweets match these filters.")
        st.stop()
    pool = pool.sort_values("repliers", ascending=False)
    options = pool[C.COL_TWEET].tolist()
    if st.session_state.get("tweet_sel") not in options:
        st.session_state.pop("tweet_sel", None)
    labels = pool.set_index(C.COL_TWEET)
    tid = st.selectbox(
        "Tweet (sorted by number of repliers)", options, key="tweet_sel",
        format_func=lambda x: f"{labels.at[x, 'tweet']} · {labels.at[x, 'campaign']} · "
                              f"{labels.at[x, 'repliers']} repliers, {labels.at[x, 'flagged']} flagged")

    text = labels.at[tid, "text"]
    if isinstance(text, str):
        with st.container(border=True):
            st.text(text)  # plain text: tweet content is never rendered as markdown
        st.caption("English translation released with the dataset (the originals were in "
                   "Serbian or Arabic). The original tweet and its author are not shown.")
    else:
        st.caption("No text for this tweet in the dataset: the authors released translations "
                   "only for part of the attacked tweets.")

    rep = replies[(replies[C.COL_TWEET] == tid)]
    rep = rep.merge(acc[[C.COL_USER, "account", "score", "flagged", "community", "graph_degree"]],
                    on=C.COL_USER, how="left")
    if delays is not None:
        rep = rep.merge(delays, on=C.COL_REPLY, how="left")
    rep = rep.sort_values("score", ascending=False)

    c = st.columns(4)
    c[0].metric("Replies", len(rep))
    c[1].metric("Repliers", rep[C.COL_USER].nunique())
    c[2].metric("Flagged for review", int(rep.drop_duplicates(C.COL_USER)["flagged"].fillna(False).sum()))
    if show_truth:
        top = rep.drop_duplicates(C.COL_USER).head(10)
        c[3].metric("Known IO in top 10", f"{int(top[C.COL_LABEL].sum())} / {len(top)}")

    if delays is not None and rep[C.COL_DELAY].notna().any():
        st.subheader("Reply time vs. model score")
        plot = rep.dropna(subset=[C.COL_DELAY, "score"]).assign(
            minutes=lambda d: d[C.COL_DELAY].clip(lower=1),
            status=lambda d: np.where(d["flagged"] == True, "flagged", "not flagged"))  # noqa: E712
        fig = px.scatter(plot, x="minutes", y="score", color="status", log_x=True,
                         hover_data={"account": True, "minutes": True, "score": ":.2f"},
                         color_discrete_map={"flagged": "#d62728", "not flagged": "#9aa0a6"},
                         symbol=np.where(plot[C.COL_LABEL] == 1, "known IO", "normal")
                         if show_truth else None,
                         labels={"minutes": "hours after the tweet, rounded up (log scale)"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Coordinated replies often arrive in a burst and score high together.")

    st.subheader("Repliers, most suspicious first")
    cols = ["account", "score", "flagged", "community", "graph_degree"] + \
           ([C.COL_DELAY] if delays is not None else []) + ([C.COL_LABEL] if show_truth else [])
    st.dataframe(rep.drop_duplicates(C.COL_USER)[cols]
                 .rename(columns={"graph_degree": "co-repliers", C.COL_DELAY: "minutes"})
                 .round(3), hide_index=True, use_container_width=True)
    pick_acc = st.selectbox("Open an account", rep.drop_duplicates(C.COL_USER).head(200)[C.COL_USER],
                            format_func=short)
    st.button("Open in Account lookup", on_click=go_account, args=(pick_acc,))
