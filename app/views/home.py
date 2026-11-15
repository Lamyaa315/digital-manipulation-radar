"""Landing page: hero with a live (real-data) network whose flagged accounts pop up their reply activity, and a ticker.
Every number on this page is read from the pipeline's outputs, not typed in."""
import pandas as pd
import streamlit as st

from app.core.store import C, MAIN_SET, PRETTY, Store, graph_payload
from app.ui import canvas
from app.ui.blocks import html
from app.views.common import goto

HERO_CAMPAIGN = "honduras_022020"


@st.cache_data(show_spinner=False)
def hero_payload(_store: Store, campaign: str, n_groups: int = 7, per: int = 48) -> dict:
    """Most connected accounts of the largest co-reply communities of one campaign (pipeline output)."""
    a = _store.acc[(_store.acc["campaign_group"] == campaign) & _store.acc["community"].notna()]
    top = a["community"].value_counts().head(n_groups).index
    groups = {}
    for c in top:
        m = a[a["community"] == c].nlargest(per, "graph_strength").index
        groups[f"C{int(c)}"] = list(m)
    ids = [u for g in groups.values() for u in g]
    e = _store.edges[_store.edges["u"].isin(ids) & _store.edges["v"].isin(ids)]
    gid = {u: g for g, us in groups.items() for u in us}
    same = e["u"].map(gid) == e["v"].map(gid)
    e = pd.concat([e[same].nlargest(1600, "shared"), e[~same].nlargest(80, "shared")])
    p = graph_payload(_store.acc, groups, e, spread=0.75)
    # real reply activity of each account, shown as pop-up "tweets" (the dataset does not publish reply text)
    r = _store.replies[_store.replies[C.COL_USER].isin(ids)]
    act = r.groupby(C.COL_USER).agg(n=("delay_min", "size"), med=("delay_min", "median"), first=("t", "min"))
    for node in p["nodes"]:
        uid = int(node["id"])
        if node["flag"] and uid in act.index:
            a_ = act.loc[uid]
            m = float(a_["med"])
            med = f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h" if m < 2880 else f"{m / 1440:.0f} days"
            node["tw"] = {"n": int(a_["n"]), "med": f"median reply {med} after the post",
                          "when": f"{a_['first']:%b %Y}"}
    p["tweets"] = True
    return p


def render(store: Store) -> None:
    info, best = store.info, store.info["best_model"]
    m = store.metrics.query("model == @best and feature_set == @MAIN_SET").set_index("held_out")
    sa = m.loc["saudi_arabia_112019"]
    mean_pr = m["pr_auc"].mean()

    left, right = st.columns([6, 5], gap="large")
    with left:
        html('<div class="hero"><p class="kick">CLUEX</p>'
             '<h1>Not every online movement is <em>organic.</em></h1>'
             '<p>State-backed operations flood chosen tweets with replies from accounts that work together. '
             'ClueX scores every replier with a machine-learning model, links the accounts that keep '
             'showing up on the same tweets, and shows the evidence behind every score.</p></div>')
        a, b = st.columns([3, 2])
        if a.button("Start an Investigation  →", type="primary", width="stretch", icon=":material/radar:"):
            goto(1)
        if b.button("How It Works", width="stretch", icon=":material/play_arrow:"):
            goto(3)
    with right:
        p = hero_payload(store, HERO_CAMPAIGN)
        canvas.network(p, key="hero-net", height=520, rotate=0.00003,
                       caption=f"Real data · {PRETTY[HERO_CAMPAIGN]} · {len(p['nodes'])} accounts · pop-ups = real reply activity · ")

    msgs = [f"<b>{info['n_accounts']:,}</b> accounts analysed", f"<b>{info['n_replies']:,}</b> replies",
            f"<b>{info['n_target_tweets']:,}</b> attacked tweets", f"<b>{info['n_io']:,}</b> accounts removed by Twitter",
            f"<b>{sa['precision_at_100']:.0%}</b> of top-100 Saudi accounts confirmed — campaign never seen in training",
            f"mean PR-AUC <b>{mean_pr:.2f}</b> on unseen campaigns", "data: OSoMe · Twitter 2018–2021 · CC BY 4.0",
            "sample data is always labelled"]
    line = "".join(f"<span>{x}</span>" for x in msgs)
    html(f'<div class="ticker"><div>{line}{line}</div></div>')
