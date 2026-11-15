"""Builds notebooks/radar_notebook.ipynb — the project notebook, in the order the brief asks for:
problem → data → EDA → preprocessing → models → evaluation → model selection → conclusion.

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/radar_notebook.ipynb

Short takeaways under the outputs come from scripts/notebook_text.json.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TF = ROOT / "scripts" / "notebook_text.json"
T = json.loads(TF.read_text(encoding="utf-8")) if TF.exists() else {}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


def note(k):
    md(T.get(k, f"> **Takeaway:** _(written after running — {k})_"))


# ---------------------------------------------------------------- 0
md("""
# Digital Manipulation Radar
**Machine Learning Unit 4 & 5 — final project** · Lamyaa Alnahdi · Raghad Alqahtani · Rawan Alsubaie

| Step | Section |
|---|---|
| 1. Problem definition | §1 |
| 2. Exploratory data analysis | §2–3 |
| 3. Preprocessing | §4 |
| 4. Modeling | §5 |
| 5. Evaluation | §6–7 |
| 6. Model selection | §8 |
| 7. Deployment | §9 (the live app) |
""")

md("""
## 1 · Problem
State-backed influence operations flood chosen tweets with replies from accounts that work together.
To a reader it looks like public opinion.

**Goal:** given how an account replied (timing, text similarity, profile, who it replied with), predict whether
it belongs to an operation. **Target:** `replier_label` — 1 = removed by Twitter as part of an operation, 0 = normal.

**Data:** OSoMe *Coordinated Reply Attacks* (Indiana University, ICWSM 2025), Zenodo 10.5281/zenodo.13896308, CC BY 4.0.
""")

code('''
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.dummy import DummyClassifier

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from radar import config as C, model as M
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:,.3f}".format)

NORMAL, IO = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titleweight": "bold"})
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "hgb": "Gradient Boosting"}
''')

# ---------------------------------------------------------------- 2
md("## 2 · The data")
code('''
df = pd.read_pickle(ROOT / "artifacts" / "accounts.pkl")   # one row per account (built by radar.pipeline)
features = [c for c in pd.read_json(ROOT / "artifacts" / "run_info.json", typ="series")["feature_columns"]]
print(f"{len(df):,} accounts × {len(features)} features")
df[[C.COL_USER, "campaign_group", "age", "mean_diff_min", "mean_cosine", "followers_ratio", C.COL_LABEL]].head()
''')
code('''
quality = pd.Series({"duplicate accounts": df[C.COL_USER].duplicated().sum(),
                     "missing values": int(df[features].isna().sum().sum()),
                     "infinite values": int(np.isinf(df[features]).sum().sum())})
quality.to_frame("count")
''')
note("data")

# ---------------------------------------------------------------- 3
md("## 3 · Exploratory data analysis")
md("### 3.1 How many operation accounts?")
code('''
counts = df[C.COL_LABEL].value_counts().rename({0: "normal", 1: "operation"})
fig, ax = plt.subplots(figsize=(7, 2))
ax.barh(counts.index, counts.values, color=[NORMAL, IO], height=.5)
for i, n in enumerate(counts.values):
    ax.text(n, i, f"  {n:,} ({n / counts.sum():.2%})", va="center")
ax.set_xscale("log"); ax.set_xlim(1e3, 1e7); ax.grid(axis="y", visible=False); ax.set_title("Accounts by label")
plt.show()
''')
note("imbalance")

md("### 3.2 How operation accounts differ")
code('''
def compare(ax, col, title, bins, log=False, xlabel=""):
    for lab, name, c in [(0, "normal", NORMAL), (1, "operation", IO)]:
        x = df.loc[df[C.COL_LABEL] == lab, col].dropna()
        ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), histtype="step", lw=2, color=c,
                label=f"{name} (median {x.median():.2f})")
    if log: ax.set_xscale("log")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("% of accounts"); ax.legend(fontsize=8)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
compare(ax[0], "mean_cosine", "Text similarity to other replies", 40, xlabel="mean cosine similarity")
compare(ax[1], "age", "Account age", 40, xlabel="years")
compare(ax[2], "50%_diff_min", "Reply delay", np.logspace(0, 3, 40), log=True, xlabel="median delay (hours, log)")
plt.tight_layout(); plt.show()
''')
note("differ")

md("### 3.3 Do they reply in groups?")
code('''
grp = (df.assign(in_group=df["graph_degree"].fillna(0) > 0)
         .groupby(C.COL_LABEL)["in_group"].mean().rename({0: "normal", 1: "operation"}) * 100)
fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(grp.index, grp.values, color=[NORMAL, IO], height=.5)
for i, v in enumerate(grp.values): ax.text(v, i, f"  {v:.1f}%", va="center")
ax.set_xlabel("% of accounts in a co-reply group"); ax.grid(axis="y", visible=False); ax.set_xlim(0, 40)
plt.show()
''')
note("groups")

md("### 3.4 Is every campaign the same?")
code('''
camp = (df.groupby("campaign_group").agg(accounts=(C.COL_LABEL, "size"), operation=(C.COL_LABEL, "sum"))
          .assign(share_pct=lambda d: d.operation / d.accounts * 100))
camp["median delay of operation accounts (h)"] = (df[df[C.COL_LABEL] == 1].groupby("campaign_group")["50%_diff_min"].median())
camp.sort_values("accounts", ascending=False)
''')
note("campaigns")

# ---------------------------------------------------------------- 4
md("""
## 4 · Preprocessing
| Step | Why |
|---|---|
| Keep only the 76 numeric behaviour features; drop IDs and the label | IDs would leak the answer |
| Replace ±∞ with missing | uploaded files can contain them; trees handle missing values |
| Split **before** any sampling (stratified, 80 % train / 20 % test) | the test set keeps the real 1 % share |
| Training set only: keep all operation accounts + 10 normal per operation account | 99 % normal would teach the model to say "normal" |
| Logistic Regression only: median imputation + scaling inside a pipeline | a linear model needs comparable scales |
""")
code('''
X = df[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
y = df[C.COL_LABEL].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rng = np.random.default_rng(42)
keep = M.downsample(np.arange(len(y_tr)), y_tr, 10, rng)
X_bal, y_bal = X_tr[keep], y_tr[keep]
pd.DataFrame({"rows": [len(y_tr), len(y_bal), len(y_te)],
              "operation accounts": [y_tr.sum(), y_bal.sum(), y_te.sum()],
              "operation share": [y_tr.mean(), y_bal.mean(), y_te.mean()]},
             index=["train (all)", "train (after downsampling)", "test (untouched)"])
''')

# ---------------------------------------------------------------- 5
md("""
## 5 · Models
| Model | Why we tried it |
|---|---|
| Logistic Regression | simple, interpretable baseline |
| Random Forest | the model used in the original research paper |
| Gradient Boosting (HistGradientBoosting) | usually strongest on tabular data; fast; handles missing values |

Each model also learns a **decision threshold** on a validation part of the training data (the cut-off that gives
the best F1), because the training data is 10:1 while real data is about 100:1.
""")
code('''
fitted = {}
for name in ["logreg", "rf", "hgb"]:
    fitted[name] = M.fit_with_threshold(name, X_bal, y_bal)      # (model, threshold)
    print(f"{NAMES[name]:<20} threshold = {fitted[name][1]:.2f}")
''')

# ---------------------------------------------------------------- 6
md("## 6 · Evaluation on the test set (accuracy · precision · recall · F1)")
code('''
def scores(y_true, p, thr):
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred), "pr_auc": average_precision_score(y_true, p)}

rows = {"Always 'normal' (baseline)": scores(y_te, np.zeros(len(y_te)), .5)}
probs = {}
for name, (m, thr) in fitted.items():
    probs[name] = m.predict_proba(X_te)[:, 1]
    rows[NAMES[name]] = scores(y_te, probs[name], thr)
test_table = pd.DataFrame(rows).T
test_table
''')
note("test_table")
code('''
fig, ax = plt.subplots(figsize=(9, 3.4))
show = test_table.drop(index="Always 'normal' (baseline)")[["accuracy", "precision", "recall", "f1"]]
xs = np.arange(len(show.columns)); w = .26
for i, (name, row) in enumerate(show.iterrows()):
    ax.bar(xs + (i - 1) * w, row.values, w * .92, label=name, color=["#9a9a94", NORMAL, IO][i])
ax.set_xticks(xs, show.columns); ax.set_ylim(0, 1.05); ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
ax.set_title("Test set — the three models"); ax.grid(axis="x", visible=False)
plt.show()
''')
code('''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (name, (m, thr)) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te, (probs[name] >= thr).astype(int)),
                           display_labels=["normal", "operation"]).plot(ax=ax, colorbar=False, values_format=",", cmap="Blues")
    ax.set_title(NAMES[name]); ax.grid(False)
plt.tight_layout(); plt.show()
''')
note("confusion")

# ---------------------------------------------------------------- 7
md("""
## 7 · Evaluation on campaigns the model never saw
A random split mixes every campaign into training and test, so the model has already seen each operation's style.
The real use is spotting a **new** operation, so we also hold out **one whole campaign at a time**: train on the
others, test on it, repeat for all six (leave-one-campaign-out).
""")
code('''
loco = {}
for name in ["logreg", "rf", "hgb"]:
    loco[name], _, _ = M.loco(df, features, name)
    loco[name]["accuracy"] = (loco[name]["tp"] + loco[name]["tn"]) / loco[name]["n_test"]
loco_table = pd.DataFrame({NAMES[n]: t[["accuracy", "precision", "recall", "f1", "pr_auc"]].mean() for n, t in loco.items()}).T
loco_table
''')
code('''
best = loco["hgb"].set_index("held_out")
best[["n_test", "n_pos", "accuracy", "precision", "recall", "f1", "pr_auc", "precision_at_100"]]
''')
note("loco")

md("### 7.1 What drives the model?")
code('''
no_text, _, _ = M.loco(df, [c for c in features if "cosine" not in c], "hgb")
pd.DataFrame({"all features": loco["hgb"][["f1", "pr_auc"]].mean(),
              "without text similarity": no_text[["f1", "pr_auc"]].mean()}).T
''')
note("no_text")

# ---------------------------------------------------------------- 8
md(T.get("selection", "## 8 · Model selection\n_(to be written)_"))

# ---------------------------------------------------------------- 9
md(T.get("conclusion", "## 9 · Deployment and conclusion\n_(to be written)_"))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "radar_notebook.ipynb"
nbf.write(nb, out)
print("wrote", out)
