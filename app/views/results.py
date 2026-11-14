"""Step 2 — the model's findings, straight away. Reviewing them is optional."""
from datetime import datetime, timezone
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import theme as T
from app.ui.blocks import figures, html
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


def render(store) -> None:
    inv = st.session_state.get("inv")
    if inv is None:
        html('<h1 class="page-title">Results</h1>')
        st.info("No investigation yet.")
        if st.button("Start one →", type="primary"):
            goto(1)
        return
    q = inv.queue
    groups = _groups(inv)
    html(f'<p class="kicker">Results · {escape(inv.name)}</p><h1 class="page-title">What the model found</h1>'
         f'<p class="lead">Scored by {MODEL_NAME[inv.source]}. These are the model\'s results as they are — '
         'you can download them now, or review the accounts yourself first (optional).</p>')
    figures([(f"{inv.n_scored:,}", "accounts scored", "", ""),
             (f"{inv.n_above:,}", f"above review threshold ({inv.threshold:.2f})", "coral", ""),
             (f"{len(groups)}", f"groups among top {len(q)}", "", ""),
             (f"{q['score'].iloc[0]:.2f}", "highest score", "", "")])

    a, b, c = st.columns(3)
    a.download_button("Download results (HTML report)", _html(inv, groups).encode(), "radar_results.html",
                      "text/html", type="primary", width="stretch")
    b.download_button("Download ranked accounts (CSV)",
                      q.rename_axis(None).reset_index(names="account_id")[["rank", "account_id", "account", "score",
                                                                           "group", "reasons"]].to_csv(index=False).encode(),
                      "radar_ranked_accounts.csv", "text/csv", width="stretch")
    if c.button("Review them yourself (optional) →", width="stretch"):
        goto(3)

    left, right = st.columns([3, 2], gap="large")
    with left:
        html('<div class="section"><span class="num">01</span><span class="name">Highest-scored accounts</span></div>')
        top = q.head(15).assign(why=q["reasons"].str.split(" · ").str[0])
        st.dataframe(top[["rank", "account", "score", "group", "why"]], hide_index=True, width="stretch", height=420,
                     column_config={"rank": st.column_config.NumberColumn("#", width="small"),
                                    "score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.2f"),
                                    "why": st.column_config.TextColumn("main reason", width="large")})
        st.caption(f"Showing 15 of the top {len(q)}. The full list is in the downloads and on the Review page.")
    with right:
        html('<div class="section"><span class="num">02</span><span class="name">Groups</span>'
             '<span class="note">accounts that keep replying together</span></div>')
        if groups.empty:
            st.caption("No groups found among the top accounts (they do not share enough tweets).")
        else:
            st.dataframe(groups, hide_index=True, width="stretch", height=230,
                         column_config={"mean score": st.column_config.NumberColumn(format="%.2f")})
        html('<div class="section"><span class="num">03</span><span class="name">Score distribution</span></div>')
        sc = (store.acc.loc[store.acc["campaign_group"] == inv.campaign, "score"] if inv.campaign
              else inv.all_scores)
        fig = go.Figure(go.Histogram(x=sc, nbinsx=25, marker_color=T.STEEL))
        fig.add_vline(x=inv.threshold, line=dict(color=T.LIME, dash="dot"))
        fig.update_layout(height=200, xaxis_title="score", yaxis=dict(title="accounts", type="log"), bargap=.05)
        st.plotly_chart(fig, width="stretch")

    if inv.labels is not None:
        with st.expander("How good is this list? (demo data — Twitter's own labels)"):
            n = int(inv.labels.sum())
            st.markdown(f"Of the top **{len(q)}** accounts, **{n}** were removed by Twitter as part of the operation "
                        f"(**{n / len(q):.0%}**).")
            st.caption("Only possible on data from the dataset. For your own data there is no answer key.")
