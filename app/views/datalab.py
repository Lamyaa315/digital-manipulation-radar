"""05 — Data Lab: dataset, what it can and cannot show, model evaluation, settings."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core import store as S
from app.core.store import Store
from app.ui import theme as T
from app.ui.blocks import compact, figures, html, legend, section, tag


def render(store: Store) -> None:
    info = store.info
    html('<h1 class="page-title">Data and model</h1>')
    if info["is_sample"]:
        st.error("SYNTHETIC SAMPLE DATA — the pipeline ran on generated data. Every number is meaningless.")

    section("", "Dataset")
    html(f'<p>OSoMe <i>Coordinated Reply Attacks in Influence Operations</i> (Pote et al., ICWSM 2025), '
         f'Zenodo 10.5281/zenodo.13896308, CC BY 4.0. Twitter, 2018–2021. Labels mark accounts Twitter '
         f'suspended as part of state-backed operations.</p>')
    figures([
        (compact(info["n_accounts"]), "accounts", "", "DATASET"),
        (compact(info["n_replies"]), "replies", "", "DATASET"),
        (f"{info['n_target_tweets']:,}", "attacked tweets", "", "DATASET"),
        (compact(info["n_io"]), "known IO accounts", "", "TRUTH"),
        (compact(info["graph"]["edges"]), "co-reply links", "", "RULE"),
    ])
    html('<p class="kicker" style="margin-top:18px">What this data can and cannot show</p>')
    st.dataframe(pd.DataFrame([
        ("Reply timing", "yes", "exact UTC times decoded from tweet IDs"),
        ("Common targets", "yes", "who replied to which attacked tweet"),
        ("Content similarity", "summary only", "released cosine statistics; reply texts are not public"),
        ("Text of attacked tweets", "partly", f"English translation for {int(store.tweets()['text'].notna().sum()):,} of {len(store.tweets()):,} attacked tweets"),
        ("Shared URLs", "no", "only URL counts per reply"),
        ("Live accounts on X today", "no", "the app scores only accounts in the dataset"),
    ], columns=["signal", "available", "detail"]), hide_index=True, width="stretch")

    section("", "Model", f"best model: {info['best_model']} · leave-one-campaign-out, real class ratio")
    m = store.metrics.query("model == @info['best_model'] and feature_set == @S.MAIN_SET").copy()
    m["lift"] = m["pr_auc"] / m["prevalence"]
    m["prevalence"] = m["prevalence"] * 100
    m["campaign"] = m["held_out"].map(lambda c: S.PRETTY.get(c, c))
    st.dataframe(m[["campaign", "n_test", "n_pos", "prevalence", "pr_auc", "lift", "roc_auc", "precision_at_100",
                    "threshold", "precision", "recall"]], hide_index=True, width="stretch",
                 column_config={"prevalence": st.column_config.NumberColumn(format="%.2f%%"),
                                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
                                "lift": st.column_config.NumberColumn(format="%.0f×"),
                                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                                "precision_at_100": st.column_config.NumberColumn("precision@100", format="%.2f")})
    st.caption(f"Replication of the paper (balanced 1:1, 10-fold, random forest): ROC-AUC "
               f"{info['replication']['roc_auc']:.3f}, F1 {info['replication']['f1']:.3f}. At the automatic threshold "
               "most flagged accounts are false alarms on the large campaigns — use the score as a review queue.")
    c1, c2 = st.columns(2)
    with c1:
        ab = (store.metrics.query("model == @info['best_model']").groupby("feature_set")["pr_auc"].mean().sort_values())
        fig = go.Figure(go.Bar(x=ab.values, y=ab.index, orientation="h",
                               marker_color=[T.CYAN if "network" in i and "+" in i else T.STEEL for i in ab.index]))
        fig.update_layout(height=260, title="Mean PR-AUC by feature set", xaxis_range=[0, 0.7])
        st.plotly_chart(fig, width="stretch")
    with c2:
        if store.imp is not None:
            imp = store.imp.groupby("group")["pr_auc_drop"].mean().sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=T.LIME))
            fig.update_layout(height=260, title="Permutation importance (drop in PR-AUC)")
            st.plotly_chart(fig, width="stretch")

    section("", "Analysis settings")
    html(f'<p class="idline">co-reply link: ≥ <b>3</b> shared attacked tweets · case = Louvain community '
         f'(resolution <b>{S.LOUVAIN_RESOLUTION}</b>) with ≥ <b>{S.CASE_MIN_SIZE}</b> accounts · '
         f'"synchronized" = within <b>{S.SYNC_WINDOW_MIN}</b> min of the post · priority HIGH ≥ 0.70, '
         f'ELEVATED ≥ 0.40 (mean model score) · review threshold per campaign chosen on training campaigns</p>')
    st.toggle("Show known labels (evaluation only)", key="show_truth",
              help="Reveals which accounts Twitter actually suspended. Off by default so the investigation "
                   "is driven by the model and the measured signals, not by the answer key.")

    section("", "How to read every number")
    legend()
    html('<p class="idline" style="margin-top:12px">A high score means "worth a human look", never "this account '
         'is fake". Labels are noisy, campaigns of normal accounts are inferred, and the network uses unlabelled '
         'replies of the campaign being tested.</p>')
