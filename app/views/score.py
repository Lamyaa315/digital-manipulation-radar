"""Score accounts: the deployed model, used live on data the user provides."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.store import C, ROOT, Store, short
from app.ui import theme as T
from app.ui.blocks import figures, html, section

MODELS = ROOT / "models"


@st.cache_resource(show_spinner=False)
def scorer():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from radar.deploy import Scorer
    return Scorer() if Scorer.available() else None


def render(store: Store) -> None:
    html('<p class="kicker">Score accounts</p>'
         '<h1 class="page-title">Run the model on accounts</h1>'
         '<p class="lead">Give the model a list of accounts and it ranks them from most to least '
         'likely to belong to a coordinated campaign. The score is a priority for human review, '
         'not a verdict.</p>')
    sc = scorer()
    if sc is None:
        st.error("No saved model. Run `python -m radar.deploy` first.")
        return

    src = st.radio("Input", ["Try 300 sample accounts", "Upload a CSV", "One account from the dataset"],
                   horizontal=True, label_visibility="collapsed", key="score_src")
    df, note = None, ""
    if src == "Try 300 sample accounts":
        df = pd.read_csv(MODELS / "sample_accounts.csv")
        note = ("These 300 real accounts (30 known IO, 270 normal) were part of the training data, so "
                "the ranking looks better than it would on new data. Honest results on unseen campaigns "
                "are in the Data Lab.")
    elif src == "Upload a CSV":
        st.download_button("Download the template (column names)", (MODELS / "template.csv").read_bytes(),
                           "radar_template.csv", "text/csv")
        st.caption(f"One row per account, with the {len(sc.features)} feature columns of the released dataset "
                   "(RQ3_replier_classifier_features.csv). An ID column is optional. Missing values are allowed.")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001 - show any parse error to the user
                st.error(f"Could not read the file: {e}")
                return
            missing, bad = sc.check(df)
            if missing:
                st.error(f"{len(missing)} required columns are missing, e.g. {', '.join(missing[:5])}. "
                         "Use the template.")
                return
            if bad:
                st.error(f"These columns must be numbers: {', '.join(bad[:5])}")
                return
    else:
        q = st.text_input("Account ID or acct-XXXXXX", key="score_q",
                          placeholder="e.g. a full ID from the dataset")
        if q:
            q = q.strip().lower().removeprefix("acct-")
            ids = store.fx.index.astype(str)
            hit = store.fx.index[ids == q] if len(q) > 6 else store.fx.index[ids.str.endswith(q)]
            if len(hit) == 0:
                st.warning("Not found. In the hosted version, feature values are kept only for accounts in "
                           "the co-reply network or above the threshold.")
                return
            df = store.fx.loc[[hit[0]]].reset_index().rename(columns={"index": C.COL_USER})
            df = df.rename(columns={df.columns[0]: C.COL_USER})
            note = ("This account is in the training data of the deployed model. Its held-out score "
                    f"(from a model that never saw its campaign) is {store.acc.at[hit[0], 'score']:.2f}.")

    if df is None or df.empty:
        return
    with st.spinner("Scoring…"):
        p = sc.score(df)
    idcol = next((c for c in df.columns if "id" in c.lower()), None)
    out = pd.DataFrame({"account": df[idcol].map(short) if idcol else [f"row {i + 1}" for i in range(len(df))],
                        "score": p, "above threshold": p >= sc.threshold})
    if idcol:
        out.insert(0, "id", df[idcol].astype(str))
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out.index += 1

    section("", "Result")
    figures([(f"{len(out):,}", "accounts scored", "", ""),
             (f"{int(out['above threshold'].sum()):,}", f"above threshold ({sc.threshold:.2f})", "coral", ""),
             (f"{out['score'].iloc[0]:.2f}", "highest score", "", "")])
    if note:
        st.caption(note)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(out, width="stretch", height=380,
                     column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1,
                                                                             format="%.2f")})
        st.download_button("Download results (CSV)", out.to_csv(index=True, index_label="rank").encode(),
                           "radar_scores.csv", "text/csv")
    with right:
        fig = go.Figure(go.Histogram(x=p, nbinsx=20, marker_color=T.STEEL))
        fig.add_vline(x=sc.threshold, line=dict(color=T.CORAL, dash="dot"))
        fig.update_layout(height=260, xaxis_title="score", yaxis_title="accounts", bargap=.05)
        st.plotly_chart(fig, width="stretch")
        st.caption("Most real accounts score near 0. The dotted line is the review threshold.")

    with st.expander("About this model"):
        m = sc.meta
        st.markdown(
            f"- **Model:** {m['model']} on {len(sc.features)} behavioural features per account "
            f"(reply similarity, timing, profile, hashtags/mentions/links, engagement).\n"
            f"- **Trained on:** all campaigns — {m['n_train']:,} accounts ({m['n_pos']:,} known IO, "
            f"{m['neg_per_pos']} normal per IO account), {m['trained_at'][:10]}.\n"
            f"- **Threshold:** {m['threshold']:.2f}. {m['note']}\n"
            "- **How good is it?** Tested on campaigns it never saw: 87 of its top 100 Saudi accounts were "
            "real IO accounts (see Data Lab).\n"
            "- **Re-train:** `python -m radar.deploy`.")
