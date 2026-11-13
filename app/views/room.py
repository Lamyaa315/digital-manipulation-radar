"""01 — Radar Room: pick a campaign, run detection, see the radar and the living network."""
import streamlit as st

from app.core.store import Store, tier
from app.ui import canvas
from app.ui.blocks import compact, figures, html, section, tag
from app.views.common import (campaign, cases, goto, overview_payload, pretty, radar_payload,
                              raw_payload)


def render(store: Store) -> None:
    html("""
    <p class="kicker"><b>01</b> — THE RADAR ROOM</p>
    <div class="hero-title">DIGITAL MANIPULATION <span>RADAR</span></div>
    <p class="hero-sub">See the coordination others miss.</p>
    <p class="hero-desc">Uncover suspicious behavioral patterns and hidden relationships across social media
    activity — 2.67 million replies to 3,866 attacked tweets, from state-backed influence operations removed by Twitter.</p>
    """)

    # ---- step 1: select data -------------------------------------------------------------
    camps = store.campaigns()
    section("STEP 1", "Select data", "each campaign is scored by a model that never saw it")
    pick = st.segmented_control("Campaign", list(camps.index), format_func=pretty,
                                default=campaign(), key="camp_pick", label_visibility="collapsed")
    if pick and pick != campaign():
        st.session_state["campaign"] = pick
        st.session_state.pop("case_id", None)
        st.session_state.pop("acct", None)
        st.rerun()
    c = camps.loc[campaign()]
    figures([
        (compact(c["accounts"]), "accounts", "", "DATASET"),
        (compact(c["replies"]), "replies to targets", "", "DATASET"),
        (f"{int(c['targets']):,}", "attacked tweets", "", "DATASET"),
        (compact(c["in_graph"]), "accounts with co-reply links", "", "RULE"),
        (f"{c['first']:%Y}–{c['last']:%Y}", "reply period", "", "DATASET"),
    ])

    # ---- step 2: run -------------------------------------------------------------------
    section("STEP 2", "Run analysis")
    ran = cases(store) is not None
    left, right = st.columns([1, 2])
    with left:
        go = st.button("◉  INITIATE INVESTIGATION" if not ran else "✓  RESULTS READY", type="primary",
                       width="stretch", disabled=ran)
        html(f'<p class="idline" style="margin-top:10px">What runs now: co-reply network of this campaign → '
             f'Louvain communities → evidence per group {tag("RULE")}<br>Scores: held-out predictions saved by '
             f'<code>python -m radar.pipeline</code> {tag("MODEL")}</p>')
    with right:
        if go:
            bar = st.progress(0.0, text="Starting")
            store.detect(campaign(), progress=lambda p, msg: bar.progress(p, text=msg))
            st.session_state["run_id"] = f"{campaign()}-{st.session_state.get('run_n', 0) + 1}"
            st.session_state["run_n"] = st.session_state.get("run_n", 0) + 1
            st.rerun()
        if ran:
            cs = cases(store)
            html(f'<p class="step"><b>✓</b> ANALYSIS COMPLETE — {len(cs)} co-reply groups of ≥ 5 accounts · '
                 f'{int((cs["score"] >= 0.7).sum())} high-priority</p>')

    # ---- step 3: radar + network --------------------------------------------------------
    section("STEP 3", "Discover patterns", "radar sweep is a visual metaphor — blips are real accounts")
    cs = cases(store)
    col_r, col_l = st.columns([5, 4])
    with col_r:
        html(f'<p class="kicker">COORDINATION RADAR · distance from centre = 1 − model score {tag("MODEL")}</p>')
        if cs is None or cs.empty:
            canvas.radar([], [], "idle", key="radar", idle="AWAITING DATA — INITIATE INVESTIGATION")
        else:
            blips, groups = radar_payload(store, campaign())
            hit = canvas.radar(blips, groups, st.session_state.get("run_id", "r"), key="radar")
            if hit:
                goto(1, case_id=hit, acct=None)
    with col_l:
        html('<p class="kicker">TOP DISCOVERIES</p>')
        if cs is None or cs.empty:
            st.caption("No results yet. Groups appear here after the analysis runs.")
        else:
            for _, k in cs.head(7).iterrows():
                t = tier(k["score"])
                a, b = st.columns([4, 1])
                a.markdown(
                    f'<span class="idline"><b>{k["case_id"]}</b></span> &nbsp;{store.title(k)}<br>'
                    f'<span class="idline">{k["size"]} accounts · {k["shared_targets"]} shared targets · '
                    f'<span style="color:{"#FF626D" if t == "HIGH" else "#F4B860" if t == "ELEVATED" else "#87919F"}">'
                    f'{t} {k["score"]:.2f}</span></span>', unsafe_allow_html=True)
                if b.button("Open", key=f"open-{k['case_id']}"):
                    goto(1, case_id=k["case_id"], acct=None)

    html(f'<p class="kicker" style="margin-top:28px">THE LIVING NETWORK · each dot an account, each line ≥ 3 attacked '
         f'tweets replied to in common {tag("RULE")}</p>')
    if cs is None or cs.empty:
        p = raw_payload(store, campaign())
        canvas.network(p, key="net-raw", height=540,
                       caption=f"{len(p['nodes'])} MOST-CONNECTED ACCOUNTS · NOT YET ANALYSED")
        st.caption("Before the analysis: raw co-reply activity, no groups, no scores. Run the investigation to reveal them.")
    else:
        p = overview_payload(store, campaign())
        sel = canvas.network(p, key="net-hero", height=600, caption="TOP 12 GROUPS · ",
                             group_labels={g: g for g in p["groups"]})
        if sel:
            st.session_state["hero_pick"] = sel
        pick = st.session_state.get("hero_pick")
        if pick and pick.get("group") in set(cs["case_id"]):
            k = cs[cs["case_id"] == pick["group"]].iloc[0]
            a, b, c3 = st.columns([3, 1, 1])
            a.markdown(f'<span class="idline">SELECTED <b>acct-{pick["node"][-6:]}</b> in <b>{k["case_id"]}</b> — '
                       f'{store.title(k)}, {k["size"]} accounts</span>', unsafe_allow_html=True)
            if b.button("Investigate case →", key="hero-case", type="primary"):
                goto(1, case_id=k["case_id"], acct=int(pick["node"]))
            if c3.button("Open account →", key="hero-acct"):
                goto(2, acct=int(pick["node"]), case_id=k["case_id"])
        st.caption("Scroll to zoom · drag to pan · drag a dot to move it · click a dot to light up its links · double-click to reset.")
