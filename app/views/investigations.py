"""02 — Investigations: the case list, and the case workspace."""
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.store import C, SYNC_WINDOW_MIN, Store, tier
from app.ui import canvas
from app.ui import theme as T
from app.ui.blocks import compact, evidence, figures, html, risk, section, tag
from app.views.common import (campaign, case_payload, cases, dossier, fmt_range, get_case, goto,
                              pretty, sequence, show_truth, timeline_fig)


def render(store: Store) -> None:
    cs = cases(store)
    if cs is None:
        html('<p class="kicker"><b>02</b> — INVESTIGATIONS</p>')
        st.markdown(f"## No investigation run yet for {pretty(campaign())}")
        if st.button("◉  INITIATE INVESTIGATION", type="primary"):
            bar = st.progress(0.0)
            store.detect(campaign(), progress=lambda p, m: bar.progress(p, text=m))
            st.rerun()
        return
    case = get_case(store, st.session_state.get("case_id"))
    if case is None:
        case_list(store, cs)
    else:
        workspace(store, case)


# ---------------------------------------------------------------------------------------
def case_list(store: Store, cs: pd.DataFrame) -> None:
    html(f'<p class="kicker"><b>02</b> — INVESTIGATIONS · {pretty(campaign()).upper()}</p>'
         f'<div class="hero-title" style="font-size:52px">CAMPAIGN <span>DISCOVERIES</span></div>'
         f'<p class="hero-desc">Each case is a group of accounts that kept replying to the same attacked tweets '
         f'{tag("RULE")}, ordered by the model\'s average score {tag("MODEL")}. Titles are generated from '
         f'the strongest measured signal.</p>')
    f1, f2, f3 = st.columns([2, 2, 3])
    lvl = f1.segmented_control("Priority", ["ALL", "HIGH", "ELEVATED", "LOW"], default="ALL", key="lvl")
    min_size = f2.slider("Minimum accounts", 5, 200, 5, key="minsize")
    view = cs[cs["size"] >= min_size]
    if lvl and lvl != "ALL":
        view = view[view["score"].map(tier) == lvl]
    f3.markdown(f'<p class="idline" style="margin-top:34px">{len(view)} of {len(cs)} cases shown</p>',
                unsafe_allow_html=True)
    for _, k in view.head(40).iterrows():
        t = tier(k["score"])
        span = (k["last"] - k["first"])
        a, b = st.columns([6, 1])
        a.markdown("\n".join(s.strip() for s in f"""
        <div class="case">
          <div class="cid">CASE<b>{k['case_id']}</b></div>
          <div><h3>{store.title(k)}</h3>
            <ul>
              <li><b>{k['size']}</b> associated accounts</li>
              <li><b>{k['posts']:,}</b> replies on <b>{k['targets']}</b> attacked tweets</li>
              <li><b>{k['sync_share']:.0%}</b> of replies within {SYNC_WINDOW_MIN} min of the post</li>
              <li><b>{k['shared_targets']}</b> tweets hit by ≥ 2 members</li>
              <li>reply similarity (max cosine, median) <b>{k['max_cosine']:.2f}</b></li>
              <li>active {k['first']:%b %Y} – {k['last']:%b %Y} ({span.days} days)</li>
              {f"<li>known IO accounts <b>{k['n_io']}</b> {tag('TRUTH')}</li>" if show_truth() else ""}
            </ul></div>
          {risk(t, k['score'])}
        </div>""".splitlines()), unsafe_allow_html=True)
        b.markdown("<div style='height:46px'></div>", unsafe_allow_html=True)
        if b.button("Investigate →", key=f"inv-{k['case_id']}", width="stretch"):
            goto(1, case_id=k["case_id"], acct=None)


# ---------------------------------------------------------------------------------------
def workspace(store: Store, k: pd.Series) -> None:
    if st.button("← All cases"):
        goto(1, case_id=None, acct=None)
    t = tier(k["score"])
    html(f'<p class="kicker"><b>CASE FILE</b> · {pretty(k["campaign"]).upper()} · OPENED FROM LIVE ANALYSIS</p>'
         f'<div style="display:flex;align-items:flex-end;gap:30px;flex-wrap:wrap">'
         f'<div><div class="hero-title" style="font-size:64px;margin-bottom:0">{k["case_id"]}</div>'
         f'<p class="hero-sub">{store.title(k)}</p></div>'
         f'<div style="min-width:220px;margin-left:auto">{risk(t, k["score"])}</div></div>')

    # A. overview
    section("A", "Case overview")
    figures([
        (f"{k['size']}", "associated accounts", "", "RULE"),
        (f"{k['posts']:,}", "relevant replies", "", "DATASET"),
        (f"{k['targets']}", "attacked tweets", "", "DATASET"),
        (f"{k['flagged']}", "above review threshold", "coral", "MODEL"),
        (fmt_range(k["first"], k["last"]), "activity range (UTC)", "", "DATASET"),
    ] + ([(f"{k['n_io']}", "known IO accounts", "", "TRUTH")] if show_truth() else []))

    # B + E. network and account
    section("B", "Account network", "click an account to open it in panel E")
    members = k["members"]
    if st.session_state.get("acct") not in set(members):
        st.session_state["acct"] = int(store.acc.loc[members, "score"].idxmax())
    p = case_payload(store, k["campaign"], k["case_id"])
    sel = canvas.network(p, key=f"net-{k['case_id']}", height=520, selected_group=k["case_id"], lock_group=True,
                         caption=f"{len(p['nodes'])} OF {k['size']} ACCOUNTS (MOST CONNECTED) · ")
    if sel and int(sel["node"]) != st.session_state.get("acct"):
        st.session_state["acct"] = int(sel["node"])
    deg = pd.Series({n["id"]: n["deg"] for n in p["nodes"]}).sort_values(ascending=False)
    html(f'<p class="idline">CENTRAL ACCOUNTS {tag("RULE")} &nbsp;' +
         " · ".join(f'<b>acct-{i[-6:]}</b> {d} links' for i, d in deg.head(5).items()) + "</p>")

    # C. evidence
    section("C", "Behavioral evidence", "why this group surfaced")
    m_sync, o_sync = store.sync_baseline(k)
    rep = store.case_replies(k)
    c1, c2 = st.columns(2)
    with c1:
        evidence("1 · Temporal synchronization", f"{m_sync:.0%} <span style='font-size:15px;color:#87919F'>vs {o_sync:.0%}</span>",
                 f"Share of replies posted within {SYNC_WINDOW_MIN} minutes of the attacked tweet: this group vs "
                 f"everyone else who replied to the same tweets. Times come from the tweet IDs.",
                 "RULE", "hot" if m_sync > 1.3 * o_sync else "")
        on = store.replies[store.replies[C.COL_TWEET].isin(rep[C.COL_TWEET].unique())]
        inside = on[C.COL_USER].isin(members)
        bins = np.logspace(0, np.log10(60 * 24 * 30), 30)
        fig = go.Figure()
        for lab, sel_, col in (("others on same tweets", ~inside, T.STEEL), ("this group", inside, T.CORAL)):
            h, _ = np.histogram(on.loc[sel_, "delay_min"].clip(1), bins=bins)
            fig.add_scatter(x=bins[:-1], y=h / max(h.sum(), 1), name=lab, line=dict(color=col, width=2, shape="hv"),
                            fill="tozeroy", opacity=.9)
        fig.update_layout(height=200, xaxis=dict(type="log", title="minutes after the post (log)"),
                          yaxis_title="share", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="ev-sync")
    with c2:
        base = store.baseline["max_cosine"]
        evidence("2 · Content similarity", f"{k['max_cosine']:.2f} <span style='font-size:15px;color:#87919F'>vs {base:.2f}</span>",
                 "Highest cosine similarity among each account's replies (median over the group) vs the median "
                 "account with at least 2 replies to attacked tweets. Released by the dataset authors; the reply texts themselves are not public.",
                 "FEATURE", "hot" if k["max_cosine"] > 2 * max(base, .1) else "")
        a = store.acc.loc[members]
        fig = go.Figure()
        fig.add_box(x=a["max_cosine"], name="this group", marker_color=T.CORAL, boxpoints=False)
        fig.add_box(x=store.baseline_cos, name="accounts with ≥ 2 replies",
                    marker_color=T.STEEL, boxpoints=False)
        fig.update_layout(height=200, showlegend=False, xaxis_title="max reply cosine similarity")
        st.plotly_chart(fig, width="stretch", key="ev-sim")
    c3, c4 = st.columns(2)
    with c3:
        evidence("3 · Common interaction targets", f"{k['shared_targets']} <span style='font-size:15px;color:#87919F'>of {k['targets']} tweets</span>",
                 f"Attacked tweets replied to by at least two members. Linked pairs share {k['mean_shared']:.1f} "
                 f"targets on average; {k['density']:.0%} of all possible pairs are linked.",
                 "RULE", "hot" if k["density"] > .3 else "cool")
        mat = store.case_matrix(k)
        fig = go.Figure(go.Heatmap(z=(mat.values > 0).astype(int), x=[f"…{str(c)[-4:]}" for c in mat.columns],
                                   y=[f"acct-{str(i)[-6:]}" for i in mat.index],
                                   colorscale=[[0, T.SURFACE], [1, T.CYAN]], showscale=False, xgap=2, ygap=2,
                                   hovertemplate="%{y} → tweet %{x}<extra></extra>"))
        fig.update_layout(height=320, xaxis=dict(title="attacked tweets", tickfont=dict(size=9)),
                          yaxis=dict(tickfont=dict(size=9, family="JetBrains Mono")))
        st.plotly_chart(fig, width="stretch", key="ev-mat")
    with c4:
        evidence("4 · Shared URLs", "not testable",
                 f"The dataset releases how many URLs each reply had (median for this group: {k['mean_num_url']:.2f} "
                 f"per reply, all accounts: {store.baseline['mean_num_url']:.2f}), but not the URLs. Shared-link "
                 "amplification cannot be checked with this data.", "NA")
        evidence("Model view", f"{k['score']:.2f}",
                 f"Average held-out score of the members. {k['flagged']} of {k['size']} are above the threshold "
                 "for their campaign. Text similarity drives the model most (permutation importance 0.52 of "
                 "PR-AUC), then profile and reply timing.", "MODEL", "cool")

    # D. timeline
    section("D", "Activity timeline", "real UTC times decoded from tweet IDs")
    st.plotly_chart(timeline_fig(rep), width="stretch", key="tl-case")
    busiest = rep.groupby(C.COL_TWEET).size().sort_values(ascending=False)
    opts = list(busiest.index[:25])
    tid = st.selectbox("Replay one attacked tweet", opts, key=f"seq-{k['case_id']}",
                       format_func=lambda t: f"tweet-{str(t)[-6:]} · {busiest[t]} group replies"
                       + (" · has text" if t in store.texts.index else ""))
    all_rep = store.tweet_replies(tid)
    a, b = st.columns([3, 2])
    with a:
        txt = store.texts.get(tid)
        if isinstance(txt, str):
            html(f'<div class="quote">{escape(txt)}</div><p class="idline">English translation released with the dataset {tag("DATASET")}</p>')
        z = all_rep.assign(group=np.where(all_rep[C.COL_USER].isin(members), "this group", "other repliers"))
        z = z[z["delay_min"] <= z.loc[z["group"] == "this group", "delay_min"].max() * 1.2 + 30]
        fig = go.Figure()
        for lab, col, sz in (("other repliers", T.STEEL, 5), ("this group", T.CORAL, 9)):
            q = z[z["group"] == lab]
            fig.add_scatter(x=q["t"], y=q["score"], mode="markers", name=lab, text=q["account"],
                            marker=dict(color=col, size=sz, line=dict(width=0)),
                            hovertemplate="%{text}<br>%{x|%H:%M:%S}<br>score %{y:.2f}<extra></extra>")
        fig.update_layout(height=260, yaxis_title="model score", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="tl-one")
    with b:
        html(f'<p class="kicker">SEQUENCE OF EVENTS {tag("DATASET")}</p>' + "<br>".join(
            f'<p class="idline" style="margin:6px 0">{s}</p>'
            for s in sequence(all_rep, all_rep["t_post"].iloc[0], members)))

    # E. account
    section("E", "Account investigation")
    dossier(store, st.session_state["acct"], key="ws")
