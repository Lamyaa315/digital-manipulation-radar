"""04 — Activity Timeline: how replies to one attacked tweet unfolded, in real time."""
from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.core.store import C, Store
from app.ui import theme as T
from app.ui.blocks import figures, html, section, tag
from app.views.common import campaign, goto, pretty, sequence, show_truth


def render(store: Store) -> None:
    html(f'<p class="kicker">{pretty(campaign())}</p>'
         '<h1 class="page-title">One tweet, reply by reply</h1>'
         f'<p class="lead">Every reply time here is real: Twitter tweet IDs encode their posting time to the '
         f'millisecond {tag("DATASET")}.</p>')
    tw = store.tweets(campaign())
    only_text = st.toggle("Only tweets with an English translation", value=True, key="tl_text")
    if only_text and tw["text"].notna().any():
        tw = tw[tw["text"].notna()]
    tid = st.selectbox("Attacked tweet", list(tw.index[:300]), key="tl_tweet",
                       format_func=lambda t: f"{tw.at[t, 'tweet']} · {tw.at[t, 'posted']:%d %b %Y} · "
                       f"{tw.at[t, 'replies']:,} replies · {int(tw.at[t, 'flagged']):,} above threshold")
    if tid is None:
        st.info("No attacked tweets for this campaign.")
        return
    r = store.tweet_replies(tid)
    posted = r["t_post"].iloc[0]
    txt = tw.at[tid, "text"]
    if isinstance(txt, str):
        html(f'<div class="quote" style="font-size:19px">{escape(txt)}</div>'
             f'<p class="idline">tweet <b>{tid}</b> · posted <b>{posted:%Y-%m-%d %H:%M:%S} UTC</b> · English translation {tag("DATASET")}</p>')
    else:
        html(f'<p class="idline">tweet <b>{tid}</b> · posted <b>{posted:%Y-%m-%d %H:%M:%S} UTC</b> · no text released for this tweet</p>')
    figures([
        (f"{len(r):,}", "replies", "", "DATASET"),
        (f"{int(r['flagged'].sum()):,}", "from accounts above threshold", "coral", "MODEL"),
        (f"{(r['delay_min'] <= 60).mean():.0%}", "within the first hour", "", "RULE"),
    ] + ([(f"{int(r[C.COL_USER].map(store.acc[C.COL_LABEL]).sum()):,}", "known IO replies", "", "TRUTH")] if show_truth() else []))

    section("", "Unfolding", "drag the slider under the chart to zoom into a window")
    hrs = st.select_slider("Show the first", ["1 hour", "6 hours", "24 hours", "7 days", "all"], value="24 hours", key="tl_win")
    lim = {"1 hour": 60, "6 hours": 360, "24 hours": 1440, "7 days": 10080, "all": np.inf}[hrs]
    v = r[r["delay_min"] <= lim]
    fig = go.Figure()
    for lab, m, col, sz in (("below threshold", ~v["flagged"].astype(bool), T.STEEL, 5),
                            ("above threshold", v["flagged"].astype(bool), T.CORAL, 8)):
        q = v[m]
        fig.add_scatter(x=q["t"], y=q["score"], mode="markers", name=lab, text=q["account"],
                        marker=dict(color=col, size=sz, line=dict(width=0)),
                        hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M:%S}<br>score %{y:.2f}<extra></extra>")
    fig.add_vline(x=posted, line=dict(color=T.CYAN, dash="dot"))
    fig.update_layout(height=380, yaxis_title="model score", legend=dict(orientation="h", y=1.1),
                      xaxis=dict(rangeslider=dict(visible=True, bgcolor=T.SURFACE, thickness=.06)))
    st.plotly_chart(fig, width="stretch", key="tl-main")

    # cumulative curve: are flagged replies early?
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        for lab, m, col in (("above threshold", r["flagged"].astype(bool), T.CORAL), ("below threshold", ~r["flagged"].astype(bool), T.STEEL)):
            d = np.sort(r.loc[m, "delay_min"].clip(lower=0.1).to_numpy())
            if len(d):
                fig.add_scatter(x=d, y=np.arange(1, len(d) + 1) / len(d), name=lab, line=dict(color=col, width=2))
        fig.update_layout(height=260, xaxis=dict(type="log", title="minutes after the post (log)"),
                          yaxis=dict(title="share of replies so far", tickformat=".0%"), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="tl-cdf")
    with b:
        html('<p class="kicker">What happened, in order (accounts above the threshold)</p>' + "".join(
            f'<p class="idline" style="margin:6px 0">{s}</p>'
            for s in sequence(r, posted, set(r.loc[r["flagged"].astype(bool), C.COL_USER]))))

    section("", "Who replied", "ranked by model score")
    top = r.sort_values("score", ascending=False).drop_duplicates(C.COL_USER).head(40)
    top = top.assign(when=top["t"].dt.strftime("%H:%M:%S"),
                     after=top["delay_min"].map(lambda m: f"{m:,.1f} min" if m < 120 else f"{m / 60:,.1f} h"))
    cols = ["account", "score", "when", "after"] + ([C.COL_LABEL] if show_truth() else [])
    ev = st.dataframe(top[cols], hide_index=True, width="stretch", height=300, on_select="rerun",
                      selection_mode="single-row", key="tl_tbl",
                      column_config={"score": st.column_config.ProgressColumn("model score", min_value=0, max_value=1, format="%.2f")})
    rows = ev.selection.rows if ev else []
    if rows:
        uid = int(top.iloc[rows[0]][C.COL_USER])
        if st.button(f"Open {top.iloc[rows[0]]['account']} in the Network Explorer →", type="primary"):
            goto(3, acct=uid)
