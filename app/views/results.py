"""Step 2 — the model's findings, straight away. Reviewing them is optional."""
from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.store import C, graph_payload
from app.ui import canvas
from app.ui import theme as T
from app.ui.blocks import compact, html
from app.views.common import goto

MODEL_NAME = {"dataset": "a model that never saw this campaign", "upload": "the full model",
              "simple": "the simple-input model"}


def _groups(inv) -> pd.DataFrame:
    rows = []
    for g, members in inv.groups.items():
        sc = inv.all_scores.reindex(members).dropna()
        rows.append({"group": g, "accounts": len(members), "in top list": int((inv.queue["group"] == g).sum()),
                     "mean score": float(sc.mean()) if len(sc) else float("nan"),
                     "above threshold": int((sc >= inv.threshold).sum())})
    df = pd.DataFrame(rows)
    return df.sort_values("mean score", ascending=False) if len(df) else df


def _html(inv, groups: pd.DataFrame) -> str:
    q = inv.queue
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = "".join(f"<tr><td>{int(r['rank'])}</td><td>{escape(str(r['account']))}</td><td>{r['score']:.2f}</td>"
                   f"<td>{escape(r['group'] or '—')}</td><td>{escape(r['reasons'])}</td></tr>" for _, r in q.iterrows())
    grows = "".join(f"<tr><td>{r['group']}</td><td>{r['accounts']}</td><td>{r['mean score']:.2f}</td>"
                    f"<td>{r['above threshold']}</td></tr>" for _, r in groups.iterrows())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Radar results — {escape(inv.name)}</title>
<style>body{{font-family:Inter,Arial,sans-serif;color:#111;max-width:980px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#f5f5f5}}
.m{{color:#666;font-size:13px}}</style></head><body>
<h1>Model results — {escape(inv.name)}</h1>
<p class="m">Generated {now} by Digital Manipulation Radar using {MODEL_NAME[inv.source]}. Not reviewed by a person.
Scores rank accounts for review; a high score is not proof of a fake account.</p>
<p><b>{inv.n_scored:,}</b> accounts scored · <b>{inv.n_above:,}</b> above the review threshold ({inv.threshold:.2f})
· <b>{len(groups)}</b> groups among the top {len(q)}</p>
<h2>Top {len(q)} accounts</h2><table><tr><th>#</th><th>account</th><th>score</th><th>group</th><th>why</th></tr>{rows}</table>
<h2>Groups</h2><table><tr><th>group</th><th>accounts</th><th>mean score</th><th>above threshold</th></tr>{grows}</table>
</body></html>"""


ICONS = {
    "flag": '<svg viewBox="0 0 24 24"><path d="M4 21V4"/><path d="M4 4h12l-2 4 2 4H4"/></svg>',
    "clock": '<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>',
    "users": ('<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/>'
              '<circle cx="17" cy="9" r="2.4"/><path d="M15.5 14.2c2.9.3 5.5 2.6 5.5 5.8"/></svg>'),
    "net": ('<svg viewBox="0 0 24 24"><circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="12" r="2.6"/>'
            '<circle cx="5" cy="18" r="2.2"/><circle cx="19" cy="18" r="2.2"/>'
            '<path d="M7 7.3 10 10.5M17 7.3 14 10.5M7 16.7 10 13.5M17 16.7 14 13.5"/></svg>'),
    "pie": '<svg viewBox="0 0 24 24"><path d="M21 12A9 9 0 1 1 12 3v9z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15z"/></svg>',
    "time": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "list": ('<svg viewBox="0 0 24 24"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/>'
             '<circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>'),
}
TICKS = [1, 10, 30, 60, 120, 360, 720, 1440]
TICK_TXT = ["1 min", "10 min", "30 min", "1 h", "2 h", "6 h", "12 h", "24 h"]


def _dur(m: float) -> str:
    return "—" if m != m else f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h" if m < 48 * 60 else f"{m / 1440:.0f} days"


@st.cache_data(show_spinner=False)
def _campaign_delays(_store, campaign: str) -> pd.DataFrame:
    """Every reply by an account of this campaign: minutes after the tweet, and whether the account is flagged."""
    acc = _store.acc[_store.acc["campaign_group"] == campaign]
    r = _store.replies[_store.replies[C.COL_USER].isin(acc.index)]
    return pd.DataFrame({"delay_min": r["delay_min"].to_numpy(),
                         "flag": r[C.COL_USER].map(acc["flagged"]).fillna(False).to_numpy(bool)})


def _delays(inv, store) -> pd.DataFrame | None:
    if inv.source == "dataset" and inv.campaign:
        return _campaign_delays(store, inv.campaign)
    if inv.replies is None or "delay_min" not in inv.replies:
        return None
    r = inv.replies
    sc = inv.all_scores.reindex(r[C.COL_USER]).to_numpy()
    return pd.DataFrame({"delay_min": r["delay_min"].to_numpy(), "flag": sc >= inv.threshold})


def _scores(inv, store) -> pd.Series:
    if inv.source == "dataset" and inv.campaign:
        return store.acc.loc[store.acc["campaign_group"] == inv.campaign, "score"]
    return inv.all_scores


def _kpis(inv, d) -> None:
    half = float(d["delay_min"].median()) if d is not None and len(d) else float("nan")
    in_groups = sum(len(m) for m in inv.groups.values())
    kp = [("flag", "Flagged for review", f"{inv.n_above / max(inv.n_scored, 1):.1%}",
           f"of analyzed accounts ({inv.n_above:,} of {inv.n_scored:,})", "hot"),
          ("clock", "Reply speed", _dur(half),
           "half of the replies came within this time" if d is not None else "needs reply times (not in this file)", ""),
          ("users", "Accounts analyzed", f"{inv.n_scored:,}", "every account scored by the model", ""),
          ("net", "Coordinated groups", f"{len(inv.groups)}", f"{in_groups:,} accounts keep replying together", "")]
    html('<div class="kpis">' + "".join(
        f'<div class="kpi {c}">{ICONS[i]}<div><div class="t">{t}</div><div class="v">{v}</div><div class="s">{s}</div></div></div>'
        for i, t, v, s, c in kp) + '</div>')


def _classification(inv, scores: pd.Series) -> None:
    html(f'<div class="panel-h">{ICONS["pie"]}Account Classification</div>')
    hi = int((scores >= inv.threshold).sum())
    mid = int(((scores >= inv.threshold / 2) & (scores < inv.threshold)).sum())
    lo = int(len(scores) - hi - mid)
    tot = max(len(scores), 1)
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    fig = go.Figure(go.Pie(values=[hi, mid, lo], hole=.68, sort=False, direction="clockwise",
                           marker=dict(colors=[T.LIME, "#7FA31F", "#B9BBC1"], line=dict(color=T.BG, width=2)),
                           textinfo="none", hovertemplate="%{value:,} accounts<extra></extra>"))
    fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      annotations=[dict(text=f"<b>{compact(tot)}</b><br>accounts", showarrow=False,
                                        font=dict(size=18, color=T.TEXT))])
    c1.plotly_chart(fig, width="stretch", key="donut")
    c2.markdown("".join(f'<div class="legend-row"><i style="background:{col}"></i><span>{lab}</span>'
                        f'<b>{v / tot:.1%} <span style="color:#8B8D93;font-weight:400">({v:,})</span></b></div>'
                        for lab, v, col in [("Likely coordinated", hi, T.LIME), ("Borderline", mid, "#7FA31F"),
                                            ("Likely organic", lo, "#B9BBC1")]), unsafe_allow_html=True)
    st.caption(f"Likely coordinated = score ≥ {inv.threshold:.2f} (the review threshold); borderline = at least half "
               "of it. A score ranks accounts for review — it is not proof.")


def _over_time(d) -> None:
    html(f'<div class="panel-h">{ICONS["time"]}Replies Over Time</div>')
    if d is None or not len(d):
        st.info("This file has no reply times, so the timing view is not available. "
                "Upload a replies file with times to see it.")
        return
    fig = go.Figure()
    for flag, name, col, w in [(False, "other accounts", "#8B8D93", 2), (True, "flagged accounts", T.LIME, 3)]:
        x = np.sort(d.loc[d["flag"] == flag, "delay_min"].clip(lower=.5).to_numpy())
        if len(x) < 2:
            continue
        y = np.arange(1, len(x) + 1) / len(x) * 100
        keep = np.unique(np.linspace(0, len(x) - 1, 400).astype(int))
        fig.add_trace(go.Scatter(x=x[keep], y=y[keep], name=f"{name} ({len(x):,} replies)",
                                 line=dict(color=col, width=w, shape="hv"),
                                 fill="tozeroy" if flag else None, fillcolor="rgba(198,244,50,.10)",
                                 hovertemplate="%{y:.0f}% within %{x:.0f} min<extra></extra>"))
        if flag:
            m = float(np.median(x))
            fig.add_annotation(x=np.log10(m), y=50, text=f"50% within {_dur(m)}", showarrow=True, arrowcolor=T.LIME,
                               ax=60, ay=-40, bordercolor=T.LIME, borderwidth=1, bgcolor=T.SURFACE,
                               font=dict(color=T.TEXT, size=12))
    fig.update_layout(height=300, xaxis=dict(type="log", tickvals=TICKS, ticktext=TICK_TXT, range=[0, np.log10(2880)],
                                             title="time after the tweet"),
                      yaxis=dict(range=[0, 101], ticksuffix="%", title="replies so far"),
                      legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch", key="rot")


def _groups_panel(inv, groups: pd.DataFrame) -> None:
    html(f'<div class="panel-h">{ICONS["net"]}Coordinated Groups</div>')
    if not inv.groups:
        st.caption("No groups among the top accounts — they do not keep replying to the same tweets.")
        return
    gsub = {g: list(inv.groups[g])[:60] for g in list(groups["group"].head(6))}
    ids = list(dict.fromkeys(u for m in gsub.values() for u in m))
    e = inv.edges[inv.edges["u"].isin(ids) & inv.edges["v"].isin(ids)].nlargest(1500, "shared")
    acc = pd.DataFrame({"score": inv.all_scores.reindex(ids).fillna(0).to_numpy()}, index=ids)
    acc["flagged"] = acc["score"] >= inv.threshold
    p = graph_payload(acc, gsub, e)
    canvas.network(p, key="res_net", height=340, rotate=0.00002,
                   caption=f"{len(p['nodes'])} accounts in the {len(gsub)} largest groups · line = replied to the same tweets · ")
    st.dataframe(groups.head(8), hide_index=True, width="stretch", height=180,
                 column_config={"mean score": st.column_config.NumberColumn(format="%.2f")})


def _top_accounts(inv) -> None:
    html(f'<div class="panel-h">{ICONS["list"]}Highest-scored Accounts</div>')
    q = inv.queue
    html("".join(f'<div class="barrow"><span>{escape(str(r["account"]))}</span><div class="track"><div class="fill'
                 f'{"" if r["score"] >= inv.threshold else " low"}" style="width:{r["score"] * 100:.0f}%"></div></div>'
                 f'<span class="num">{r["score"]:.2f}</span></div>' for _, r in q.head(12).iterrows()))
    st.caption(f"Bar = model score (lime: above the review threshold {inv.threshold:.2f}). "
               f"Full list of {len(q)} with reasons: Review page or the CSV.")


def render(store) -> None:
    inv = st.session_state.get("inv")
    if inv is None:
        html('<div class="ptitle"><h1>Investigation Results</h1><p>No investigation yet.</p></div>')
        if st.button("Start one →", type="primary"):
            goto(1)
        return
    q = inv.queue
    groups = _groups(inv)
    d = _delays(inv, store)

    h1, h2 = st.columns([3, 2], vertical_alignment="center")
    h1.markdown(f'<div class="ptitle"><h1>Investigation Results</h1><p>Scored by {MODEL_NAME[inv.source]}. '
                'Review is optional.</p></div>', unsafe_allow_html=True)
    h2.markdown(f'<div class="chip"><b>{escape(inv.name)}</b> · {datetime.now(timezone.utc):%d %b %Y}</div>',
                unsafe_allow_html=True)
    _kpis(inv, d)

    a, b, c = st.columns(3)
    a.download_button("Download results (HTML report)", _html(inv, groups).encode(), "radar_results.html",
                      "text/html", type="primary", width="stretch", icon=":material/download:")
    b.download_button("Download ranked accounts (CSV)",
                      q.rename_axis(None).reset_index(names="account_id")[["rank", "account_id", "account", "score",
                                                                           "group", "reasons"]].to_csv(index=False).encode(),
                      "radar_ranked_accounts.csv", "text/csv", width="stretch", icon=":material/table:")
    if c.button("Review them yourself (optional) →", width="stretch", icon=":material/search:"):
        goto(3)

    if inv.focus:
        focus_section(inv)

    left, right = st.columns([2, 3], gap="medium")
    with left.container(border=True):
        _classification(inv, _scores(inv, store))
    with right.container(border=True):
        _over_time(d)

    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        _groups_panel(inv, groups)
    with right.container(border=True):
        _top_accounts(inv)

    if inv.labels is not None:
        with st.expander("How good is this list? (demo data — Twitter’s own labels)"):
            n = int(inv.labels.sum())
            st.markdown(f"Of the top **{len(q)}** accounts, **{n}** were removed by Twitter as part of the operation "
                        f"(**{n / len(q):.0%}**).")
            st.caption("Only possible on data from the dataset. For your own data there is no answer key.")


def focus_section(inv) -> None:
    """The accounts the user asked about, shown first, with a plain verdict."""
    html('<div class="section"><span class="num">00</span><span class="name">The accounts you asked about</span></div>')
    if inv.alone:
        st.info("This file holds only these accounts, so the model judged each one on its own behaviour "
                "(timing, profile). It cannot see whether they reply together with other accounts — include every "
                "reply to the same tweets for a stronger check.")
    ranks = inv.all_scores.rank(ascending=False, method="min")
    for u in inv.focus:
        sc = float(inv.all_scores[u])
        r = inv.queue.loc[u] if u in inv.queue.index else None
        above = sc >= inv.threshold
        verdict = ("Behaves like the coordinated accounts the model was trained on — worth a closer look."
                   if above else "Looks like an ordinary replier to the model.")
        group = r["group"] if r is not None else ""
        mates = [m for m in inv.groups.get(group, []) if m != u]
        with st.container(border=True):
            c1, c2 = st.columns([2, 3])
            with c1:
                html(f'<p class="kicker">account</p><div style="font-size:20px">{escape(str(r["account"]) if r is not None else str(u))}</div>'
                     f'<p class="idline">{escape(str(u))}</p>'
                     f'<div style="font-size:30px;margin-top:8px;color:{"#C6F432" if above else "#F2F2F0"}">{sc:.2f}</div>'
                     f'<p class="idline">score · rank <b>#{int(ranks[u])}</b> of {inv.n_scored:,} · threshold {inv.threshold:.2f}</p>')
            with c2:
                st.markdown(f"**{verdict}**")
                if r is not None:
                    html("".join(f'<div class="reason"><span class="dot"></span><span>{escape(x)}</span></div>'
                                 for x in str(r["reasons"]).split(" · ")))
                if group:
                    st.caption(f"Member of group {group}: keeps replying to the same tweets as {len(mates)} other "
                               f"accounts in this file.")
                elif not inv.alone:
                    st.caption("Not part of any group: it does not keep replying to the same tweets as other accounts here.")
                if inv.labels is not None and u in inv.labels.index:
                    st.caption("Demo data — Twitter's label: " + ("**removed as part of the operation**"
                               if inv.labels[u] == 1 else "**normal account**"))

