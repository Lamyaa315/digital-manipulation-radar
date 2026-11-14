"""Step 3 — the report: what was reviewed, what was found, and files to share."""
from datetime import datetime, timezone
from html import escape

import streamlit as st

from app.core.store import Store
from app.ui.blocks import figures, html
from app.views.common import goto


def _html_report(inv, susp, groups) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done, total = inv.progress()
    rows = "".join(f"<tr><td>{int(r['rank'])}</td><td>{escape(str(r['account']))}</td><td>{r['score']:.2f}</td>"
                   f"<td>{escape(r['group'] or '—')}</td><td>{escape(r['reasons'])}</td><td>{escape(r['note'])}</td></tr>"
                   for _, r in susp.iterrows())
    grows = "".join(f"<tr><td>{g}</td><td>{n}</td><td>{k}</td></tr>" for g, n, k in groups)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Radar report — {escape(inv.name)}</title>
<style>body{{font-family:Inter,Arial,sans-serif;color:#1D2433;max-width:960px;margin:40px auto;padding:0 20px}}
h1{{font-size:26px}} table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 24px}}
td,th{{border-bottom:1px solid #E3E6EB;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#F6F7F9}}
.muted{{color:#667085;font-size:13px}}</style></head><body>
<h1>Coordinated-account review — {escape(inv.name)}</h1>
<p class="muted">Generated {now} with Digital Manipulation Radar. Scores rank accounts for human review; a
“suspicious” decision is the reviewer's judgement, not proof of a fake account.</p>
<p><b>{done}</b> of {total} accounts reviewed · <b>{len(susp)}</b> marked suspicious ·
{int((inv.queue['decision'] == 'normal').sum())} normal · {int((inv.queue['decision'] == 'unsure').sum())} unsure</p>
<h2>Accounts marked suspicious</h2><table><tr><th>#</th><th>account</th><th>score</th><th>group</th>
<th>why it was flagged</th><th>note</th></tr>{rows}</table>
<h2>Groups</h2><table><tr><th>group</th><th>accounts in queue</th><th>marked suspicious</th></tr>{grows}</table>
</body></html>"""


def render(store: Store) -> None:
    inv = st.session_state.get("inv")
    html('<h1 class="page-title">Report</h1>')
    if inv is None:
        st.info("No investigation yet.")
        if st.button("Start one →", type="primary"):
            goto(0)
        return
    q = inv.queue
    done, total = inv.progress()
    susp = q[q["decision"] == "suspicious"].sort_values("rank")
    html(f'<p class="lead">{escape(inv.name)} · scores by '
         f'{"a model that never saw this campaign" if inv.source == "dataset" else "the deployed model"}</p>')
    figures([(f"{done}/{total}", "reviewed", "", ""),
             (f"{len(susp)}", "suspicious", "coral", ""),
             (f"{int((q['decision'] == 'normal').sum())}", "looks normal", "green", ""),
             (f"{int((q['decision'] == 'unsure').sum())}", "unsure", "amber", ""),
             (f"{q.loc[susp.index, 'group'].replace('', None).nunique()}", "groups with suspicious accounts", "", "")])
    if done == 0:
        st.info("Nothing reviewed yet. Go to Review and mark a few accounts.")
        if st.button("Go to review →", type="primary"):
            goto(1)
        return

    groups = []
    for g in sorted(inv.groups, key=lambda x: int(x[1:])):
        in_q = q[q["group"] == g]
        groups.append((g, len(in_q), int((in_q["decision"] == "suspicious").sum())))

    st.markdown("#### Accounts marked suspicious")
    if susp.empty:
        st.caption("None yet.")
    else:
        st.dataframe(susp[["rank", "account", "score", "group", "reasons", "note"]], hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn(format="%.2f"),
                                    "reasons": st.column_config.TextColumn("why it was flagged", width="large")})
    if groups:
        st.markdown("#### Groups")
        st.dataframe([{"group": g, "accounts in queue": n, "marked suspicious": k} for g, n, k in groups],
                     hide_index=True, width="stretch")

    st.markdown("#### Share")
    a, b = st.columns(2)
    a.download_button("Download report (HTML — open and print to PDF)", _html_report(inv, susp, groups).encode(),
                      "radar_report.html", "text/html", type="primary", width="stretch")
    b.download_button("Download all decisions (CSV)", inv.decisions_table().to_csv(index=False).encode(),
                      "radar_decisions.csv", "text/csv", width="stretch")

    if inv.labels is not None:
        with st.expander("Check your decisions against Twitter's own labels (demo data only)"):
            lab = inv.labels
            rev = q[q["decision"] != ""]
            s_ = rev[rev["decision"] == "suspicious"].index
            n_ = rev[rev["decision"] == "normal"].index
            st.markdown(
                f"- You marked **{len(s_)}** suspicious: **{int(lab.loc[s_].sum())}** were really removed by Twitter "
                f"as part of the operation.\n"
                f"- You marked **{len(n_)}** as normal: **{int(lab.loc[n_].sum())}** of them were in fact part of it.\n"
                f"- In the whole queue of {total}, **{int(lab.sum())}** were operation accounts "
                f"(the model's ranking alone).")
            st.caption("With real, new data there is no answer key — this is only possible because the dataset "
                       "records which accounts Twitter removed.")
