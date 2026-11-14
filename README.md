# Digital Manipulation Radar

Detects accounts that take part in **coordinated reply attacks** (influence-operation accounts
replying together to the same target tweets), and shows the co-reply groups they form.
Results mark **possible coordination for human review**, not proof of fake accounts or intent.

Data: OSoMe, *Coordinated Reply Attacks in Influence Operations* (Pote et al., ICWSM 2025),
Zenodo [10.5281/zenodo.13896308](https://doi.org/10.5281/zenodo.13896308), **CC BY 4.0**.
The plan, verified dataset facts, risks and team schedule are in [docs/PLAN.md](docs/PLAN.md).

## Layout

```
src/radar/config.py     paths, file + column names, all tunable constants (edit here only)
src/radar/data.py       loading, validation, campaign assignment
src/radar/graph.py      co-reply graph, Louvain communities, label-free graph features
src/radar/model.py      LR / RF / HistGradientBoosting, leave-one-campaign-out, metrics, importance
src/radar/pipeline.py   runs everything, writes artifacts/
app/streamlit_app.py    dashboard (reads artifacts/ only)
scripts/download_data.py   fetch the dataset (skips the 3.8 GB pickle)
scripts/inspect_data.py    Day-1 schema check
scripts/make_sample_data.py  SYNTHETIC data for testing code only
```

## Run

```bash
pip install -r requirements.txt
```

Test the code on synthetic data (results are meaningless):

```bash
python scripts/make_sample_data.py
```

```bash
PYTHONPATH=src python -m radar.pipeline --sample
```

Real data (about 880 MB download; the pipeline takes about 5 minutes; the app then uses about 1.3 GB of RAM):

```bash
python scripts/download_data.py
```

```bash
python scripts/inspect_data.py > docs/DATA_INSPECTION.txt
```

```bash
PYTHONPATH=src python -m radar.pipeline
```

Dashboard:

```bash
streamlit run app/streamlit_app.py
```

On Windows PowerShell set the path with `$env:PYTHONPATH = "src"` instead of the `PYTHONPATH=src` prefix.

## The review tool (main app)

```
streamlit run app/main.py
```

A tool for someone investigating a reply attack, in three steps along the top bar:

1. **Upload** — pick a campaign from the dataset (demo) or upload your own accounts file
   (`models/template.csv` format) plus an optional replies file (account id, tweet id).
2. **Review** — accounts ranked by the model, each with plain-language reasons and its group of
   accounts that reply to the same tweets. Mark each one Suspicious / Looks normal / Unsure, add a
   note, or mark a whole group at once.
3. **Report** — summary, list of suspicious accounts and groups, downloadable HTML report and CSV.
   On dataset campaigns you can also check your decisions against the accounts Twitter removed.

Two upload formats:
* **Simple replies file (recommended)** — one row per reply: `account_id`, `tweet_id`, `reply_time`, and
  optionally `account_created_at`, `followers`, `following`, `reply_text` (X export names such as
  `author_id`, `in_reply_to_tweet_id`, `created_at`, `text` also work). Template: `models/simple_template.csv`;
  sample: `models/simple_sample.csv` (every reply to 30 tweets of the Honduras operation).
  Scored by the **simple-input model** (`src/radar/simple.py`, `models/radar_simple.joblib`), built only from
  features that can be rebuilt exactly from such a file: reply-delay statistics (checked: identical to the
  released ones on 99.99 % of accounts), account age (within about half a day), followers, following, reply
  counts and the co-reply network. Leave-one-campaign-out mean PR-AUC **0.188** — weaker than the full model
  (0.536), because text similarity (LaBSE embeddings of every reply) and the activity ratios cannot be
  rebuilt from a plain file. Reply text is used by a separate, transparent rule that flags near-copies of
  another account's reply to the same tweet (TF-IDF character n-grams). Re-train: `python -m radar.simple`.
* **Dataset format** — the 76 released features per account (`models/template.csv`), scored by the full model.

**About** explains both models and shows their evaluation on campaigns they never saw.

* `app/core/store.py` — data (full artifacts, or the lite copy when hosted). No UI.
* `app/core/review.py` — queue, reasons, decisions. No UI.
* `app/views/` — upload, review, report, about (datalab.py).
* Reply times are real: tweet IDs encode their posting time. The dataset's `diff_min` column
  is the reply delay in **hours, rounded up** (not minutes) — checked against these times.
* Not possible with this data: shared-URL detection (only URL counts are released), reply
  text, and scoring live X accounts.

The older analyst dashboard is still available: `streamlit run app/streamlit_app.py`.

## Hosted lite version

`artifacts_lite/` (about 50 MB, committed) is a small copy of the artifacts for free hosting,
built with `python scripts/make_lite.py`. It keeps every account's score, every co-reply link,
every reply time and the tweet texts; the 76 features are kept only for accounts in the
co-reply network or above the threshold (used by the "how unusual" chart). Community detection
is run offline with the same code and seed, so the groups are identical to a local run
(checked: same 44 groups for Honduras, Saudi Arabia and Turkey). Peak memory: 0.7 GB vs 1.3 GB.

The app uses the full `artifacts/` when present and falls back to `artifacts_lite/`.
Force the lite copy locally with `RADAR_LITE=1`.

## The deployed model

`models/radar_hgb.joblib` is the model the app uses for uploaded files: the same
HistGradientBoosting setup as the evaluation, trained on all campaigns (re-train with
`python -m radar.deploy`). `models/sample_accounts.csv` and `models/sample_replies.csv` are
the sample files behind "Try the sample files". The leave-one-campaign-out run is how it was evaluated; this saved
model is how it is used. Keep scikit-learn at the pinned version or re-train after upgrading.
