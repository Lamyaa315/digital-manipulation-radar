"""Builds notebooks/01_eda.ipynb from code + the written findings in scripts/eda_text.json.

    python scripts/build_eda_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TEXT_FILE = ROOT / "scripts" / "eda_text.json"
TEXT = json.loads(TEXT_FILE.read_text(encoding="utf-8")) if TEXT_FILE.exists() else {}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


def note(k):
    md(TEXT.get(k, f"**Finding:** _(to be written after running: {k})_"))


md("""
# 01 · Exploratory Data Analysis — Digital Manipulation Radar

**Question.** Can we tell accounts that belong to a state-backed influence operation (IO) apart from ordinary
repliers, using only how they reply (timing, text, profile, who they reply with)?

**Data.** OSoMe *Coordinated reply attacks* dataset (Zenodo 10.5281/zenodo.13896308, CC BY 4.0): replies on
Twitter 2018–2021 to tweets attacked by accounts that Twitter later removed as part of state-backed operations.
The labels come from Twitter's own takedown lists, not from us.

Each section ends with a **finding** and the **decision** it led to in preprocessing, modelling or evaluation.

> Run from the repo root after `python scripts/download_data.py` and `python -m radar.pipeline`
> (the pipeline writes `artifacts/accounts.pkl`: the released features joined with each account's campaign
> and its co-reply network features).
""")

code('''
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from radar import config as C
warnings.filterwarnings("ignore")

RAW, ART = ROOT / "data" / "raw", ROOT / "artifacts"
NORMAL, IO = "#2a78d6", "#eb6834"          # one colour per class, used in every chart
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": .6, "axes.axisbelow": True,
                     "axes.edgecolor": "#8a8984", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False})
pd.set_option("display.float_format", "{:,.3f}".format)

def by_class(ax, col, data, bins, log=False, xlabel=None):
    # two overlaid histograms (each class sums to 100 %): normal vs IO
    for lab, name, c in [(0, "normal", NORMAL), (1, "IO (operation)", IO)]:
        x = data.loc[data[C.COL_LABEL] == lab, col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), histtype="step", linewidth=2, color=c,
                label=f"{name}  (median {x.median():,.2f})")
    if log: ax.set_xscale("log")
    ax.set_xlabel(xlabel or col); ax.set_ylabel("% of the class's accounts"); ax.legend(fontsize=8)
''')

md("## 1 · The files and their size")
code('''
replies = pd.read_pickle(ART / "replies.pkl")            # = poster_tweetid_campaign_type.csv
acc = pd.read_pickle(ART / "accounts.pkl")               # released features + campaign + network features
delays = pd.read_csv(RAW / "RQ1_time_difference_of_reply.csv")
pd.DataFrame({
    "rows": [len(replies), len(acc), len(delays)],
    "one row is": ["a reply to a target or control tweet", "a replying account (76 features)",
                   "one reply and its delay"],
}, index=["poster_tweetid_campaign_type.csv", "RQ3_replier_classifier_features.csv",
          "RQ1_time_difference_of_reply.csv"])
''')
code('''
print(f"replies: {len(replies):,} | accounts: {acc[C.COL_USER].nunique():,} | tweets replied to: {replies[C.COL_TWEET].nunique():,}")
print("tweet type:", replies[C.COL_TYPE].value_counts().to_dict())
print("duplicate accounts in the feature file:", acc[C.COL_USER].duplicated().sum())
''')
note("files")

md("## 2 · Class imbalance — the needle in the haystack")
code('''
counts = acc[C.COL_LABEL].value_counts().rename({0: "normal", 1: "IO (operation)"})
share = counts / counts.sum()
fig, ax = plt.subplots(figsize=(7, 2.2))
ax.barh(counts.index, counts.values, color=[NORMAL, IO], height=.55)
for i, (n, s) in enumerate(zip(counts.values, share.values)):
    ax.text(n, i, f"  {n:,}  ({s:.2%})", va="center")
ax.set_xscale("log"); ax.set_xlim(1e3, 1e7); ax.set_xlabel("accounts (log scale)")
ax.set_title("Accounts by label"); ax.grid(axis="y", visible=False)
plt.show()
pd.DataFrame({"accounts": counts, "share": share})
''')
note("imbalance")

md("## 3 · Campaigns — six very different test sets")
code('''
camp = (acc.groupby("campaign_group")
          .agg(accounts=(C.COL_LABEL, "size"), io=(C.COL_LABEL, "sum"))
          .assign(io_share=lambda d: d.io / d.accounts)
          .sort_values("accounts", ascending=False))
tweets = (replies[replies[C.COL_TYPE] == "target"].merge(acc[[C.COL_USER, "campaign_group"]], on=C.COL_USER)
          .groupby("campaign_group")[C.COL_TWEET].nunique().rename("target tweets"))
camp = camp.join(tweets)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].barh(camp.index, camp.accounts, color=NORMAL, height=.6); axes[0].set_xscale("log")
axes[0].set_title("Accounts per campaign"); axes[0].set_xlabel("accounts (log)"); axes[0].invert_yaxis()
axes[1].barh(camp.index, camp.io_share * 100, color=IO, height=.6); axes[1].invert_yaxis()
for i, v in enumerate(camp.io_share * 100): axes[1].text(v, i, f" {v:.2f}%", va="center", fontsize=9)
axes[1].set_title("Share of IO accounts in each campaign"); axes[1].set_xlabel("% IO")
for a in axes: a.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()
camp
''')
note("campaigns")

md("## 4 · Replies per tweet — target vs control")
code('''
per_tweet = replies.groupby([C.COL_TWEET, C.COL_TYPE]).size().rename("replies").reset_index()
io_per_tweet = replies[replies[C.COL_LABEL] == 1].groupby(C.COL_TWEET).size()
per_tweet["io_replies"] = per_tweet[C.COL_TWEET].map(io_per_tweet).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3))
for t, c in [("control", NORMAL), ("target", IO)]:
    x = per_tweet.loc[per_tweet[C.COL_TYPE] == t, "replies"]
    ax.hist(x, bins=np.logspace(0, 4.5, 40), histtype="step", linewidth=2, color=c, weights=np.full(len(x), 100 / len(x)),
            label=f"{t} tweets (median {x.median():.0f})")
ax.set_xscale("log"); ax.set_xlabel("replies per tweet (log)"); ax.set_ylabel("% of tweets"); ax.legend()
ax.set_title("How many replies each tweet got")
plt.show()
per_tweet.groupby(C.COL_TYPE)[["replies", "io_replies"]].agg(["count", "mean", "median", "max"])
''')
note("per_tweet")

md("""
## 5 · Reply delay — and a unit surprise

The datasheet calls the column `diff_min` (minutes). A tweet ID is a *Snowflake* ID: its top bits are the time
it was posted (`(id >> 22) + 1288834974657` ms). So we can recompute each delay ourselves and check the unit.
""")
code('''
TW = 1288834974657
r = (replies[[C.COL_TWEET, C.COL_REPLY]].merge(delays.drop_duplicates(C.COL_REPLY), on=C.COL_REPLY)
       .sample(50_000, random_state=0))
ms = lambda ids: (ids.to_numpy(dtype="int64") >> 22) + TW
gap_ms = ms(r[C.COL_REPLY]) - ms(r[C.COL_TWEET])
pd.DataFrame({
    "unit tested": ["minutes, rounded up", "hours, rounded up"],
    "rows where it matches diff_min": [(np.ceil(gap_ms / 6e4) == r["diff_min"].to_numpy()).mean(),
                                       (np.ceil(gap_ms / 3.6e6) == r["diff_min"].to_numpy()).mean()]
}).style.format({"rows where it matches diff_min": "{:.1%}"}).hide(axis="index")
''')
note("unit")
code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
by_class(axes[0], "50%_diff_min", acc, bins=np.logspace(0, 4.2, 45), log=True,
         xlabel="median reply delay per account (hours, log)")
axes[0].set_xlim(.8, 300); axes[0].set_title("How fast accounts reply")
by_class(axes[1], "entropy_diff_min", acc, bins=40, xlabel="entropy of the account's delays (bits)")
axes[1].set_title("How varied their timing is")
plt.tight_layout(); plt.show()
acc.groupby(C.COL_LABEL)[["mean_diff_min", "50%_diff_min", "min_diff_min", "entropy_diff_min"]].median().rename(index={0: "normal", 1: "IO"})
''')
note("delay")

md("""
## 6 · Text similarity

`*_cosine` = how similar an account's replies are to the other replies on the same tweet (LaBSE sentence
embeddings, computed by the dataset authors).
""")
code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
by_class(axes[0], "mean_cosine", acc, bins=50, xlabel="mean cosine similarity to other replies")
axes[0].set_title("How alike their replies are")
by_class(axes[1], "max_cosine", acc, bins=50, xlabel="max cosine similarity (closest copy)")
axes[1].set_title("Their closest near-copy")
plt.tight_layout(); plt.show()
acc.groupby(C.COL_LABEL)[["mean_cosine", "max_cosine"]].agg(["count", "mean", "median"]).T.rename(columns={0: "normal", 1: "IO"})
''')
code('''
n_rep = replies.groupby(C.COL_USER).size()
(acc.assign(cosine_is_zero=acc["mean_cosine"] == 0, one_reply=acc[C.COL_USER].map(n_rep) == 1)
    .groupby(C.COL_LABEL)[["cosine_is_zero", "one_reply"]].mean().rename(index={0: "normal", 1: "IO"})
    .style.format("{:.1%}"))
''')
note("cosine")

md("""
### 6b · How much does the model depend on text similarity?

Because the text features are so strong — and partly placeholders — we re-ran the leave-one-campaign-out
evaluation of the chosen model (Gradient Boosting) **without the nine cosine features** (takes a few minutes).
""")
code('''
from radar import model as M
cols = [c for c in acc.columns if c in set(pd.read_json(ART / "run_info.json", typ="series")["feature_columns"])]
no_text, _, _ = M.loco(acc, [c for c in cols if "cosine" not in c], "hgb")
with_text = pd.read_csv(ART / "metrics.csv").query("model == 'hgb' and feature_set == 'behavior (all released features)'")
cmp = (with_text.set_index("held_out")[["prevalence", "pr_auc", "precision_at_100"]]
       .join(no_text.set_index("held_out")[["pr_auc", "precision_at_100"]], rsuffix=" (no text)"))
cmp.loc["mean"] = cmp.mean()
cmp
''')
note("cosine_ablation")
md("## 7 · Profile — age, followers, activity")
code('''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
by_class(axes[0], "age", acc, bins=40, xlabel="account age (years)")
by_class(axes[1], "followers_ratio", acc, bins=np.logspace(-.5, 4, 45), log=True, xlabel="followers ratio (log)")
by_class(axes[2], "activity_rate", acc, bins=np.logspace(-.5, 5, 45), log=True, xlabel="activity rate (log)")
for a, t in zip(axes, ["Account age", "Followers ratio", "Activity"]): a.set_title(t)
plt.tight_layout(); plt.show()
acc.groupby(C.COL_LABEL)[["age", "followers_ratio", "following_ratio", "activity_rate"]].median().rename(index={0: "normal", 1: "IO"})
''')
note("profile")

md("""
## 8 · Who replies together — the co-reply network

Two accounts are linked when they replied to at least 3 of the same target tweets (built by `radar.graph`).
""")
code('''
in_graph = acc["graph_degree"].fillna(0) > 0
t = (acc.assign(in_graph=in_graph).groupby(C.COL_LABEL)
       .agg(accounts=("in_graph", "size"), in_network=("in_graph", "mean"),
            median_links_if_in=("graph_degree", lambda s: s[s > 0].median()))
       .rename(index={0: "normal", 1: "IO"}))
fig, ax = plt.subplots(figsize=(7, 3))
by_class(ax, "graph_degree", acc[in_graph], bins=np.logspace(0, 4, 45), log=True,
         xlabel="co-reply links (accounts in the network, log)")
ax.set_title("How many accounts each one keeps replying with"); plt.show()
t.style.format({"in_network": "{:.1%}", "accounts": "{:,}", "median_links_if_in": "{:.0f}"})
''')
note("network")

md("## 9 · Which single features separate the classes best?")
code('''
feat = [c for c in acc.columns if c not in C.NON_FEATURE_COLS and c not in
        {"campaign", "n_campaigns", "community", "score"} and pd.api.types.is_numeric_dtype(acc[c])]
y = acc[C.COL_LABEL].to_numpy()
rows = []
for c in feat:
    x = acc[c].replace([np.inf, -np.inf], np.nan)
    ok = x.notna().to_numpy()
    auc = roc_auc_score(y[ok], x[ok]) if ok.sum() and y[ok].min() != y[ok].max() else np.nan
    rows.append({"feature": c, "auc": auc, "separation": abs(auc - .5) * 2, "missing": 1 - ok.mean()})
uni = pd.DataFrame(rows).sort_values("separation", ascending=False)
top = uni.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(top.feature, top.auc.sub(.5).abs() + .5, color=NORMAL, height=.6)
ax.set_xlim(.5, 1); ax.set_xlabel("ROC-AUC of the feature on its own (0.5 = useless; direction ignored)")
ax.set_title("Top 15 single features"); ax.grid(axis="y", visible=False)
plt.show()
uni.head(15)
''')
note("univariate")

md("## 10 · Data quality — missing and infinite values")
code('''
num = acc[feat]
quality = pd.DataFrame({"missing": num.isna().mean(), "infinite": np.isinf(num).mean()})
quality = quality[(quality.missing > 0) | (quality.infinite > 0)].sort_values("missing", ascending=False)
print(f"{len(quality)} of {len(feat)} features have missing or infinite values")
quality.head(15).style.format("{:.1%}")
''')
code('''
(acc.assign(no_cosine=acc["mean_cosine"].isna(), no_network=acc["graph_degree"].fillna(0) == 0)
    .groupby(C.COL_LABEL)[["no_cosine", "no_network"]].mean().rename(index={0: "normal", 1: "IO"})
    .style.format("{:.1%}"))
''')
note("quality")

md("## 11 · Do campaigns look alike? (the reason for leave-one-campaign-out)")
code('''
main = acc[acc.campaign_group != "other"]
io = main[main[C.COL_LABEL] == 1]
groups = sorted(io.campaign_group.unique())
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, col, xl in [(axes[0], "50%_diff_min", "median delay (hours)"), (axes[1], "mean_cosine", "mean cosine")]:
    data = [io.loc[io.campaign_group == g, col].dropna() for g in groups]
    ax.boxplot(data, vert=False, showfliers=False, widths=.55, patch_artist=True,
               boxprops=dict(facecolor="#fbd9cb", edgecolor=IO), medianprops=dict(color=IO, linewidth=2),
               whiskerprops=dict(color="#8a8984"), capprops=dict(color="#8a8984"))
    ax.set_yticks(range(1, len(groups) + 1), groups); ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
    if col == "50%_diff_min": ax.set_xscale("log")
axes[0].set_title("IO accounts: delay by campaign"); axes[1].set_title("IO accounts: text similarity by campaign")
plt.tight_layout(); plt.show()
io.groupby("campaign_group")[["50%_diff_min", "mean_cosine", "age", "graph_degree"]].median()
''')
note("shift")

md("## 12 · Redundant features")
code('''
corr = acc[feat].sample(100_000, random_state=0).corr(method="spearman").abs()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(ascending=False)
print(f"feature pairs with |Spearman| > 0.95: {(pairs > .95).sum()} of {len(pairs):,}")
pairs.head(10).to_frame("|spearman|")
''')
note("redundancy")

md(TEXT.get("summary", "## Summary\n_(to be written)_"))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                               "language": "python"}})
out = ROOT / "notebooks" / "01_eda.ipynb"
nbf.write(nb, out)
print("wrote", out)
