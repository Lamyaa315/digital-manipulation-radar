"""End-to-end run. Writes everything the dashboard shows into artifacts/.

    python -m radar.pipeline            # real data in data/raw/
    python -m radar.pipeline --sample   # synthetic data in data/sample/ (code test only)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from . import config as C
from . import data, graph, model


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def run(sample: bool, models: list[str], best: str, out=C.ARTIFACTS_DIR) -> None:
    t0 = time.time()
    d = data.data_dir(sample)
    out.mkdir(parents=True, exist_ok=True)

    feats = data.load_features(d)
    cols = data.feature_columns(feats)
    replies = data.load_replies(d)
    log(f"features: {feats.shape}, replies: {replies.shape}, {len(cols)} feature columns", t0)

    camps = data.account_campaigns(replies)
    df = feats.merge(camps, on=C.COL_USER, how="inner")
    dropped = len(feats) - len(df)
    log(f"accounts with a campaign: {len(df)} (dropped {dropped} without one)", t0)

    # --- network (label-free) ---
    edges = graph.build_coreply_edges(replies)
    g = graph.to_graph(edges)
    membership = graph.communities(g) if g.number_of_edges() else {}
    gfeats = graph.graph_features(g, membership, df[C.COL_USER])
    df = df.merge(gfeats, on=C.COL_USER, how="left")
    df["community"] = df[C.COL_USER].map(membership)
    gcols = [c for c in gfeats.columns if c != C.COL_USER]
    log(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
        f"{len(set(membership.values()))} communities", t0)

    # --- experiments ---
    profile = data.profile_columns(cols)
    reply = [c for c in cols if c not in profile]
    feature_sets = {"behavior (all released features)": cols}
    ablation = {"profile only": profile, "reply features only": reply,
                "behavior + network": cols + gcols, "network only": gcols}

    metrics, oof_best, importance = [], None, None
    for name in models:
        m, oof, imp = model.loco(df, cols, name, importance=(name == best))
        metrics.append(m.assign(feature_set="behavior (all released features)"))
        log(f"LOCO {name}: mean PR-AUC {m['pr_auc'].mean():.3f}", t0)
        if name == best:
            oof_best, importance = oof, imp
    for set_name, set_cols in ablation.items():
        if not set_cols:
            log(f"ablation '{set_name}': no columns, skipped", t0)
            continue
        m, oof, _ = model.loco(df, set_cols, best)
        metrics.append(m.assign(feature_set=set_name))
        log(f"ablation '{set_name}' ({best}): mean PR-AUC {m['pr_auc'].mean():.3f}", t0)
    metrics = pd.concat(metrics, ignore_index=True)

    replication = model.replicate_balanced_cv(df, cols, "rf")
    log(f"paper-style balanced CV (rf): AUC {replication['roc_auc']:.3f}", t0)

    # --- artifacts ---
    df["score"] = oof_best  # each account scored by a model that never saw its campaign
    keep = [C.COL_USER, C.COL_LABEL, C.COL_CAMPAIGN, "campaign_group", "n_campaigns",
            "community", "score", *gcols, *cols]
    df[keep].to_pickle(out / "accounts.pkl")
    metrics.to_csv(out / "metrics.csv", index=False)
    if importance is not None:
        importance.to_csv(out / "importance.csv", index=False)
    edges.to_pickle(out / "edges.pkl")
    replies.to_pickle(out / "replies.pkl")
    delays = data.load_delays(d)
    if delays is not None:
        delays.to_pickle(out / "delays.pkl")

    comm = (df.dropna(subset=["community"]).groupby("community")
              .agg(size=(C.COL_USER, "size"), mean_score=("score", "mean"),
                   main_campaign=(C.COL_CAMPAIGN, lambda s: s.mode().iat[0]),
                   n_io=(C.COL_LABEL, "sum"))
              .reset_index().sort_values("mean_score", ascending=False))
    comm.to_csv(out / "communities.csv", index=False)

    info = {
        "data": "SYNTHETIC SAMPLE — not real results" if sample else "OSoMe coordinated replies (Zenodo 13896308)",
        "is_sample": sample,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_accounts": int(len(df)), "n_io": int(df[C.COL_LABEL].sum()),
        "n_replies": int(len(replies)), "n_target_tweets": int(
            replies.loc[replies[C.COL_TYPE] == C.TARGET_TYPE, C.COL_TWEET].nunique()),
        "dropped_without_campaign": int(dropped),
        "feature_columns": cols, "profile_columns": profile, "graph_columns": gcols,
        "best_model": best, "neg_per_pos_train": C.NEG_PER_POS_TRAIN,
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                  "min_tweets": C.GRAPH_MIN_TWEETS, "min_shared": C.GRAPH_MIN_SHARED},
        "replication": replication,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    log(f"done -> {out}", t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use synthetic data in data/sample/")
    ap.add_argument("--models", nargs="+", default=list(model.MODEL_NAMES))
    ap.add_argument("--best", default="hgb", help="model used for scores, ablation, importance")
    a = ap.parse_args()
    if a.best not in a.models:
        a.models.append(a.best)
    run(a.sample, a.models, a.best)


if __name__ == "__main__":
    main()
