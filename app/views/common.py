"""State, navigation and pieces shared by several pages."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.store import C, PRETTY, Store, graph_payload, short, short_tweet, tier
from app.ui import theme as T
from app.ui.blocks import compact, evidence, figures, html, section, tag

PAGES = ["1 · Upload", "2 · Review", "3 · Report", "About"]


def goto(page: int, **state) -> None:
    """Navigate after the current run (the sidebar widget is already drawn)."""
    st.session_state["_goto"] = PAGES[page]
    st.session_state.update(state)
    st.rerun()


def show_truth() -> bool:
    return bool(st.session_state.get("show_truth", False))


def campaign() -> str:
    return st.session_state.get("campaign", "honduras_022020")


def cases(store: Store) -> pd.DataFrame | None:
    c = campaign()
    return store._cases.get(c)


def get_case(store: Store, case_id: str | None):
    cs = cases(store)
    if case_id is None or cs is None:
        return None
    hit = cs[cs["case_id"] == case_id]
    return hit.iloc[0] if len(hit) else None


def pretty(c: str) -> str:
    return PRETTY.get(c, c)


def fmt_range(a, b) -> str:
    if pd.isna(a):
        return "—"
    return f"{a:%d %b %Y} → {b:%d %b %Y}" if a.date() != b.date() else f"{a:%d %b %Y %H:%M} → {b:%H:%M} UTC"


def _h(x) -> float:
    return int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---- payloads (cached) -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def overview_payload(_store: Store, campaign: str, top: int = 12, per: int = 38) -> dict:
    cs = _store._cases[campaign].head(top)
    groups = {}
    for _, c in cs.iterrows():
        e = _store.edges[_store.edges["u"].isin(c["members"]) & _store.edges["v"].isin(c["members"])]
        st_ = pd.concat([e.groupby("u")["shared"].sum(), e.groupby("v")["shared"].sum()]).groupby(level=0).sum()
        groups[c["case_id"]] = list(st_.reindex(c["members"]).fillna(0).sort_values(ascending=False).index[:per])
    ids = [u for g in groups.values() for u in g]
    e = _store.edges[_store.edges["u"].isin(ids) & _store.edges["v"].isin(ids)]
    gid = {u: g for g, us in groups.items() for u in us}
    same = e["u"].map(gid) == e["v"].map(gid)
    e = pd.concat([e[same].nlargest(2400, "shared"), e[~same].nlargest(120, "shared")])
    return graph_payload(_store.acc, groups, e, spread=1.0)


@st.cache_data(show_spinner=False)
def raw_payload(_store: Store, campaign: str, n: int = 320) -> dict:
    """Before detection: the most connected accounts of the campaign, one undivided cloud,
    scores hidden. Same data the detection will then organise."""
    ids = _store.acc.index[_store.acc["campaign_group"] == campaign]
    e = _store.edges[_store.edges["u"].isin(ids) & _store.edges["v"].isin(ids)]
    s = pd.concat([e.groupby("u")["shared"].sum(), e.groupby("v")["shared"].sum()]).groupby(level=0).sum()
    keep = list(s.nlargest(n).index)
    e = e[e["u"].isin(keep) & e["v"].isin(keep)].nlargest(1600, "shared")
    p = graph_payload(_store.acc, {"unsorted": keep}, e)
    for node in p["nodes"]:
        node["flag"], node["score"] = False, None
    return p


@st.cache_data(show_spinner=False)
def case_payload(_store: Store, campaign: str, case_id: str) -> dict:
    return _store.case_graph(get_case(_store, case_id))


@st.cache_data(show_spinner=False)
def radar_payload(_store: Store, campaign: str, top: int = 14, per: int = 70):
    cs = _store._cases[campaign].head(top)
    w = np.sqrt(cs["size"].clip(lower=9)).to_numpy()
    edges = np.concatenate([[0], np.cumsum(w / w.sum())]) * 2 * np.pi - np.pi / 2
    groups, blips = [], []
    for i, (_, c) in enumerate(cs.iterrows()):
        a0, a1 = float(edges[i]), float(edges[i + 1])
        groups.append({"id": c["case_id"], "label": c["case_id"], "a0": a0, "a1": a1, "size": int(c["size"]),
                       "score": float(c["score"]), "title": _store.title(c)})
        a = _store.acc.loc[c["members"], ["score", "flagged"]]
        a = a.loc[sorted(a.index, key=_h)[:per]]
        pad = min(0.04, (a1 - a0) * 0.15)
        for uid, row in a.iterrows():
            blips.append({"id": str(uid), "group": c["case_id"], "a": a0 + pad + _h(uid) * (a1 - a0 - 2 * pad),
                          "r": 0.06 + 0.92 * (1 - float(row["score"])) * (0.85 + 0.15 * _h(-uid)),
                          "score": float(row["score"]), "flag": bool(row["flagged"])})
    return blips, groups


# ---- account dossier ---------------------------------------------------------------------
def dossier(store: Store, uid, key: str) -> None:
    d = store.account(uid)
    row, nb, rep = d["row"], d["neighbors"], d["replies"]
    html(f'<p class="kicker">Account</p>'
         f'<h3 style="margin:0">{short(uid)}</h3>'
         f'<p class="idline">id <b>{uid}</b> · campaign <b>{pretty(row["campaign_group"])}</b></p>')
    figures([
        (f"{row['score']:.2f}", "model score", "coral" if row["flagged"] else "cyan", "MODEL"),
        (f"#{int(row['rank']):,}", f"rank of {int((store.acc['campaign_group'] == row['campaign_group']).sum()):,}", "", "MODEL"),
        (f"{len(rep)}", "replies to targets", "", "DATASET"),
        (f"{int(row['graph_degree'])}", "co-reply links", "", "RULE"),
    ])
    st.markdown(("▲ **Above the review threshold.** " if row["flagged"] else "Below the review threshold. ")
                + "A score ranks accounts for human review; it is not proof of a fake account.")
    if show_truth():
        html(f'<p class="idline">Known label {tag("TRUTH")} <b>{"IO account (suspended)" if row[C.COL_LABEL] else "normal replier"}</b></p>')

    c1, c2 = st.columns(2)
    with c1:
        html(f'<p class="kicker">Related accounts {tag("RULE")}</p>')
        if nb.empty:
            st.caption("No co-reply links (replied to fewer than 2 target tweets in common with anyone).")
        for _, n in nb.head(6).iterrows():
            if st.button(f"{n['account']}  ·  {int(n['shared'])} shared targets  ·  score {n['score']:.2f}",
                         key=f"{key}-nb-{n['uid']}", width="stretch"):
                st.session_state["acct"] = int(n["uid"])
                st.rerun()
    with c2:
        html(f'<p class="kicker">How unusual, by feature group {tag("FEATURE")}</p>')
        z = store.unusual(uid)
        if z is None:
            st.caption("Not included in the hosted lite version (kept only for accounts in the co-reply "
                       "network or above the threshold). Run the app locally with the full data to see it.")
        else:
            fig = go.Figure(go.Bar(x=z.values, y=z.index, orientation="h",
                                   marker_color=[T.CORAL if v > 2 else T.STEEL for v in z.values]))
            fig.update_layout(height=210, xaxis_title="distance from typical normal accounts (robust z)")
            st.plotly_chart(fig, width="stretch", key=f"{key}-z")
    html(f'<p class="kicker">Replies on attacked tweets {tag("DATASET")} '
         f'· shared URLs {tag("NA")} · reply text {tag("NA")}</p>')
    show = rep.assign(when=rep["t"].dt.strftime("%Y-%m-%d %H:%M"),
                      after=(rep["delay_min"]).map(lambda m: f"{m:,.0f} min" if m < 120 else f"{m / 60:,.1f} h"),
                      target_text=rep["text"].fillna("").str.slice(0, 90))
    st.dataframe(show[["when", "tweet", "after", "target_text"]].rename(columns={"after": "after post"}),
                 hide_index=True, width="stretch", height=180)
    st.caption(f"URLs per reply (median of this account's replies): {row['50%_num_url']:.0f} — the dataset "
               "releases URL counts, not the URLs, so shared links cannot be checked.")


def timeline_fig(rep: pd.DataFrame, height: int = 380) -> go.Figure:
    """Real timestamps (from tweet IDs) of replies, one row per target tweet."""
    order = rep.groupby("tweet")["t"].min().sort_values().index
    fig = px.scatter(rep, x="t", y="tweet", color="score", color_continuous_scale=[[0, T.STEEL], [.5, "#F4B860"], [1, T.CORAL]],
                     range_color=(0, 1), hover_data={"account": True, "t": "|%Y-%m-%d %H:%M:%S", "score": ":.2f", "tweet": False},
                     category_orders={"tweet": list(order)}, labels={"t": "", "tweet": "", "score": "model score"})
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(height=height, coloraxis_colorbar=dict(thickness=8, len=.6),
                      xaxis=dict(rangeslider=dict(visible=True, bgcolor=T.SURFACE, thickness=.06)))
    return fig


def sequence(rep: pd.DataFrame, posted, members=None) -> list[str]:
    """Event log for one target tweet, all times real (UTC)."""
    out = [f"<b>{posted:%H:%M:%S}</b> &nbsp;target tweet posted ({posted:%d %b %Y})"]
    r = rep.sort_values("t")
    if members is not None:
        r = r[r[C.COL_USER].isin(members)]
    if r.empty:
        return out
    first = r.iloc[0]
    out.append(f"<b>{first['t']:%H:%M:%S}</b> &nbsp;first {'group ' if members is not None else ''}reply — "
               f"{first['account']} (+{first['delay_min']:,.0f} min)")
    for w in (10, 60):
        win = r[r["delay_min"] <= w]
        if len(win) > 1:
            out.append(f"<b>+{w} min</b> &nbsp;{len(win)} replies from {win['account'].nunique()} accounts")
    gaps = r["t"].diff().dt.total_seconds()
    if len(r) > 3:
        roll = r.set_index("t")["account"].rolling("5min").count()
        peak_t, peak_n = roll.idxmax(), int(roll.max())
        out.append(f"<b>{peak_t:%H:%M:%S}</b> &nbsp;busiest 5 minutes — {peak_n} replies")
        out.append(f"<b>median gap</b> &nbsp;{np.nanmedian(gaps):,.0f} s between consecutive replies")
    last = r.iloc[-1]
    out.append(f"<b>{last['t']:%Y-%m-%d %H:%M}</b> &nbsp;last {'group ' if members is not None else ''}reply")
    return out
