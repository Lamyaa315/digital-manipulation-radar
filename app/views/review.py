"""Step 2 — review the ranked accounts one by one and record a decision."""
from html import escape

import pandas as pd
import streamlit as st

from app.core.review import DECISIONS
from app.core.store import C, Store, graph_payload
from app.ui import canvas
from app.ui import theme as T
from app.ui.blocks import html
from app.views.common import goto

STATUS = {"": "To review", **DECISIONS}
FILTERS = ["To review", "All", "Suspicious", "Looks normal", "Unsure"]


def _pill(d: str) -> str:
    return f'<span class="pill {d or "open"}">{STATUS[d]}</span>'


def _color(s: float) -> str:
    return T.CORAL if s >= 0.7 else T.AMBER if s >= 0.4 else T.STEEL


def render(store: Store) -> None:
    inv = st.session_state.get("inv")
    if inv is None:
        html('<h1 class="page-title">Review</h1>')
        st.info("No investigation yet.")
        if st.button("Start one →", type="primary"):
            goto(0)
        return
    q = inv.queue
    done, total = inv.progress()
    counts = q["decision"].value_counts()

    top_l, top_r = st.columns([3, 1])
    top_l.markdown(f'<h1 class="page-title">Review — {escape(inv.name)}</h1>', unsafe_allow_html=True)
    if top_r.button("Go to report →", width="stretch", type="primary" if done == total else "secondary"):
        goto(2)
    st.progress(done / max(total, 1), text=f"{done} of {total} reviewed · "
                f"{counts.get('suspicious', 0)} suspicious · {counts.get('normal', 0)} normal · "
                f"{counts.get('unsure', 0)} unsure")

    f1, f2 = st.columns([3, 2])
    show = f1.segmented_control("Show", FILTERS, default="To review", key="rv_filter", label_visibility="collapsed")
    groups = ["All groups"] + sorted(inv.groups, key=lambda g: int(g[1:]))
    if "_rv_group" in st.session_state:
        st.session_state["rv_group"] = st.session_state.pop("_rv_group")
        st.session_state["rv_filter"] = "All"
    gsel = f2.selectbox("Group", groups, key="rv_group", label_visibility="collapsed",
                        format_func=lambda g: g if g == "All groups" else f"{g} · {len(inv.groups[g])} accounts")
    view = q
    if show and show != "All":
        code = {"To review": "", "Suspicious": "suspicious", "Looks normal": "normal", "Unsure": "unsure"}[show]
        view = view[view["decision"] == code]
    if gsel != "All groups":
        view = view[view["group"] == gsel]

    left, right = st.columns([5, 6], gap="large")
    with left:
        table = view.assign(status=view["decision"].map(STATUS))[["rank", "account", "score", "group", "status"]]
        ev = st.dataframe(table, hide_index=True, width="stretch", height=560, on_select="rerun",
                          selection_mode="single-row", key=f"rv_tbl_{show}_{gsel}",
                          column_config={"rank": st.column_config.NumberColumn("#", width="small"),
                                         "score": st.column_config.ProgressColumn("score", min_value=0, max_value=1,
                                                                                  format="%.2f")})
        rows = ev.selection.rows if ev else []
        if rows:
            uid = view.index[rows[0]]
            if uid != st.session_state.get("_last_click"):
                st.session_state["_last_click"] = uid
                st.session_state["sel"] = uid
        if view.empty:
            st.caption("Nothing here." if show != "To review" else "All accounts in this view are reviewed. 🎉")

    with right:
        uid = st.session_state.get("sel")
        if uid not in q.index:
            uid = inv.next_open() or q.index[0]
            st.session_state["sel"] = uid
        card(store, inv, uid)


def card(store: Store, inv, uid) -> None:
    q = inv.queue
    r = q.loc[uid]
    with st.container(border=True):
        html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
             f'<div><span class="kicker">#{int(r["rank"])} in the queue</span><br>'
             f'<span style="font-size:22px;font-weight:700">{escape(str(r["account"]))}</span> '
             f'<span class="idline mono">{escape(str(uid))}</span></div>{_pill(r["decision"])}</div>'
             f'<div style="margin-top:12px" class="idline">Model score <b>{r["score"]:.2f}</b>'
             f'{" · above the review threshold" if r["score"] >= inv.threshold else ""}</div>'
             f'<div class="scorebar"><i style="width:{r["score"] * 100:.0f}%;background:{_color(r["score"])}"></i></div>')
        st.markdown("**Why it was flagged**")
        html("".join(f'<div class="reason"><span class="dot"></span><span>{escape(x)}</span></div>'
                     for x in str(r["reasons"]).split(" · ")))
        st.caption("These describe how the account differs from typical accounts; the score is the model's overall "
                   "judgement. A high score means “worth a look”, not proof.")

        note = st.text_input("Note (optional)", value=r["note"], key=f"note_{uid}",
                             placeholder="e.g. same text as acct-…, created the same week")
        b1, b2, b3 = st.columns(3)
        for col, code, label, kind in ((b1, "suspicious", "Suspicious", "primary"),
                                       (b2, "normal", "Looks normal", "secondary"),
                                       (b3, "unsure", "Unsure", "secondary")):
            if col.button(label, key=f"d_{code}_{uid}", type=kind, width="stretch"):
                inv.decide(uid, code, note)
                st.session_state["sel"] = inv.next_open(after=uid) or uid
                st.rerun()

    g = r["group"]
    if g:
        members = inv.groups[g]
        in_q = q.index.intersection(members)
        open_ = [u for u in in_q if q.at[u, "decision"] == ""]
        with st.container(border=True):
            st.markdown(f"**Group {g}** — {len(members)} accounts that keep replying to the same tweets; "
                        f"{len(in_q)} of them are in your queue.")
            c1, c2 = st.columns(2)
            if c1.button(f"Mark {len(open_)} as suspicious", key=f"grp_{g}", help="All unreviewed members of this group in your queue",
                         disabled=not open_, width="stretch"):
                inv.decide(open_, "suspicious", f"marked with group {g}")
                st.session_state["sel"] = inv.next_open(after=uid) or uid
                st.rerun()
            if c2.button(f"Show only {g} in the list", key=f"flt_{g}", width="stretch"):
                st.session_state["_rv_group"] = g
                st.rerun()
            with st.expander("See how the group is connected"):
                sub = inv.edges[inv.edges["u"].isin(members) & inv.edges["v"].isin(members)].nlargest(900, "shared")
                keep = list(pd.unique(pd.concat([sub["u"], sub["v"]])))[:160] or members[:160]
                acc = pd.DataFrame({"score": inv.all_scores.reindex(keep).fillna(0)})
                acc["flagged"] = acc["score"] >= inv.threshold
                p = graph_payload(acc, {g: keep}, sub[sub["u"].isin(keep) & sub["v"].isin(keep)])
                canvas.network(p, key=f"rvnet_{g}", height=380, selected_group=g, lock_group=True,
                               caption=f"{len(p['nodes'])} accounts · line = replied to the same tweets · ")

    if inv.replies is not None:
        rep = inv.replies[inv.replies[C.COL_USER] == uid]
        if len(rep):
            with st.expander(f"Its {len(rep)} replies to the tweets under investigation"):
                t = rep.assign(when=rep["t"].dt.strftime("%Y-%m-%d %H:%M"),
                               after=rep["delay_min"].map(lambda m: f"{m:,.0f} min" if m < 120 else f"{m / 60:,.1f} h"),
                               tweet_text=rep[C.COL_TWEET].map(store.texts).fillna("").str.slice(0, 100))
                st.dataframe(t[["when", "after", "tweet_text"]].rename(columns={"after": "after the tweet",
                                                                                 "tweet_text": "the tweet it replied to"}),
                             hide_index=True, width="stretch")
