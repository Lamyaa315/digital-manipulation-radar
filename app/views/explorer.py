"""03 — Network Explorer: browse the detected groups and look up any account."""
import streamlit as st

from app.core.store import C, Store
from app.ui import canvas
from app.ui.blocks import html, section, tag
from app.views.common import campaign, case_payload, cases, dossier, overview_payload, pretty


def render(store: Store) -> None:
    html(f'<p class="kicker">{pretty(campaign())}</p><h1 class="page-title">Network</h1>')
    q = st.text_input("Look up an account", placeholder="full account ID or acct-XXXXXX", key="acct_q")
    if q:
        q = q.strip().lower().removeprefix("acct-")
        ids = store.acc.index.astype(str)
        hit = store.acc.index[ids == q] if len(q) > 6 else store.acc.index[ids.str.endswith(q)]
        if len(hit) == 0:
            st.warning("No account matches. Only accounts in the dataset can be looked up — the app does not "
                       "fetch live data from X/Twitter.")
        else:
            if len(hit) > 1:
                st.caption(f"{len(hit)} accounts end with these digits; showing the highest-scored one.")
            st.session_state["acct"] = int(store.acc.loc[hit, "score"].idxmax())
            row = store.acc.loc[st.session_state["acct"]]
            if row["campaign_group"] != campaign():
                st.info(f"This account belongs to {pretty(row['campaign_group'])}.")

    cs = cases(store)
    section("", "Network", "pick a scope")
    scopes = ["Top 12 groups"] + ([f"{c} · {s} accounts" for c, s in zip(cs["case_id"], cs["size"])] if cs is not None else [])
    if cs is None:
        st.info("Run the investigation in the Radar Room to see the detected groups here.")
    else:
        scope = st.selectbox("Scope", scopes, key="exp_scope", label_visibility="collapsed")
        if scope.startswith("Top"):
            p = overview_payload(store, campaign())
            labels = {g: g for g in p["groups"]}
            group = None
        else:
            cid = scope.split(" · ")[0]
            p = case_payload(store, campaign(), cid)
            labels, group = {}, cid
        sel = canvas.network(p, key=f"exp-{scope}", height=620, selected_group=group, group_labels=labels,
                             caption=f"{len(p['nodes'])} accounts · {len(p['links'])} links · ")
        if sel:
            st.session_state["acct"] = int(sel["node"])
        st.caption("Click an account to load its dossier below. Scroll to zoom, drag to pan, double-click to reset.")

    if st.session_state.get("acct") in store.acc.index:
        section("", "Account")
        dossier(store, st.session_state["acct"], key="exp")
