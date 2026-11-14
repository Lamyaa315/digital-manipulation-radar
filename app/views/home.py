"""Landing page: hero with a live (real-data) network, ticker, numbered sections, pipeline diagram.
Every number on this page is read from the pipeline's outputs, not typed in."""
import pandas as pd
import streamlit as st

from app.core.store import MAIN_SET, PRETTY, Store, graph_payload
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
    return graph_payload(_store.acc, groups, e, spread=0.75)


def _pipeline_svg() -> str:
    steps = [("01", "Replies", "who · what · when"), ("02", "Features", "timing · text · profile"),
             ("03", "Model", "score 0 → 1"), ("04", "Network", "co-reply groups"),
             ("05", "Review", "a human decides"), ("06", "Report", "evidence to share")]
    w, gap, y = 170, 26, 30
    parts = []
    for i, (idx, name, sub) in enumerate(steps):
        x = 10 + i * (w + gap)
        parts.append(f'<g class="node"><title>{name}: {sub}</title><rect x="{x}" y="{y}" width="{w}" height="86"/>'
                     f'<text class="idx" x="{x + 14}" y="{y + 22}">{idx}</text>'
                     f'<text x="{x + 14}" y="{y + 50}">{name}</text>'
                     f'<text class="sub" x="{x + 14}" y="{y + 70}">{sub.upper()}</text></g>')
        if i < len(steps) - 1:
            parts.append(f'<path class="flow" d="M{x + w} {y + 43} H{x + w + gap}"/>')
    width = 10 + len(steps) * (w + gap)
    return f'<div class="pipe"><svg viewBox="0 0 {width} 150" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg></div>'


def render(store: Store) -> None:
    info, best = store.info, store.info["best_model"]
    m = store.metrics.query("model == @best and feature_set == @MAIN_SET").set_index("held_out")
    sa = m.loc["saudi_arabia_112019"]
    mean_pr = m["pr_auc"].mean()

    left, right = st.columns([5, 6], gap="large")
    with left:
        html('<div class="hero"><p class="kicker">Digital Manipulation Radar</p>'
             '<h1>Not every online movement is <em>organic.</em></h1>'
             '<p>State-backed operations flood chosen tweets with replies from accounts that work together. '
             'The Radar scores every replier with a machine-learning model, links the accounts that keep '
             'showing up on the same tweets, and walks an analyst through the evidence.</p></div>')
        a, b = st.columns([3, 2])
        if a.button("Enter Intelligence Center  →", type="primary", width="stretch"):
            goto(1)
        if b.button("How it works", width="stretch"):
            goto(4)
    with right:
        p = hero_payload(store, HERO_CAMPAIGN)
        canvas.network(p, key="hero-net", height=520, rotate=0.00003,
                       caption=f"Real data · {PRETTY[HERO_CAMPAIGN]} · {len(p['nodes'])} accounts · ")

    msgs = [f"<b>{info['n_accounts']:,}</b> accounts analysed", f"<b>{info['n_replies']:,}</b> replies",
            f"<b>{info['n_target_tweets']:,}</b> attacked tweets", f"<b>{info['n_io']:,}</b> accounts removed by Twitter",
            f"<b>{sa['precision_at_100']:.0%}</b> of top-100 Saudi accounts confirmed — campaign never seen in training",
            f"mean PR-AUC <b>{mean_pr:.2f}</b> on unseen campaigns", "data: OSoMe · Twitter 2018–2021 · CC BY 4.0",
            "sample data is always labelled"]
    line = "".join(f"<span>{x}</span>" for x in msgs)
    html(f'<div class="ticker"><div>{line}{line}</div></div>')

    html(f'''<div class="numsec reveal"><div class="n">01 / PROBLEM</div><div>
      <h2>Coordinated replies look like public opinion.</h2>
      <p>An operation picks a tweet — a journalist, an activist, a minister — and dozens of accounts reply within
      hours. To a reader it looks like a crowd. In this dataset every attacked tweet drew replies from at least
      five accounts that Twitter later removed as part of state-backed operations.</p>
      <div class="cols3"><div><div class="v">{info['n_target_tweets']:,}</div><div class="l">attacked tweets</div></div>
      <div><div class="v lime">{info['n_io']:,}</div><div class="l">operation accounts</div></div>
      <div><div class="v">{info['n_io'] / info['n_accounts']:.1%}</div><div class="l">of all repliers — the needle</div></div></div>
    </div></div>''')

    html('<div class="numsec reveal"><div class="n">02 / PIPELINE</div><div>'
         '<h2>From raw replies to a report.</h2>'
         '<p>Hover a step. The model ranks; the network groups; a person decides.</p>'
         f'{_pipeline_svg()}</div></div>')

    html(f'''<div class="numsec reveal"><div class="n">03 / EVIDENCE</div><div>
      <h2>Tested on campaigns it never saw.</h2>
      <p>Each campaign was hidden from training and used only for testing, at its real (tiny) share of
      operation accounts. Ranking works; automatic labelling does not — so the Radar is built around review.</p>
      <div class="cols3"><div><div class="v lime">{int(round(sa['precision_at_100'] * 100))}/100</div>
      <div class="l">top Saudi accounts confirmed</div></div>
      <div><div class="v">{sa['pr_auc'] / sa['prevalence']:.0f}×</div><div class="l">better than random (Saudi)</div></div>
      <div><div class="v">{mean_pr:.2f}</div><div class="l">mean PR-AUC · 6 unseen campaigns</div></div></div>
    </div></div>''')

    html('<div class="numsec reveal"><div class="n">04 / INVESTIGATE</div><div>'
         '<h2>Start with a real campaign or your own export.</h2>'
         '<p>Load a campaign from the dataset, or upload a plain replies file from X. You get a ranked queue, '
         'the reasons behind each score, the groups, and a report you can download.</p></div></div>')
    c1, c2, _ = st.columns([2, 2, 3])
    if c1.button("Enter Intelligence Center  →", type="primary", width="stretch", key="cta2"):
        goto(1)
    if c2.button("About the model", width="stretch", key="cta3"):
        goto(4)
