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

## The investigation app (new interface)

```
streamlit run app/main.py
```

Five pages: Radar Room (pick a campaign → INITIATE INVESTIGATION runs co-reply community
detection live, then the radar and the living network show the results), Investigations
(case list + case workspace: overview, network, evidence, timeline, account dossier),
Network Explorer, Activity Timeline (one attacked tweet replayed in real time), Data Lab.

* `app/core/store.py` — data and analysis, no UI. Swap in a new model by re-running
  `python -m radar.pipeline`; the app reads only `artifacts/`.
* `app/ui/` — theme, HTML blocks, and two canvas components (`st.components.v2`, no build
  step, no CDN): the network (zoom, pan, drag, hover, click → Python) and the radar.
* `app/views/` — one module per page.
* Every number carries a tag: MODEL (held-out prediction), FEATURE (released by the dataset),
  RULE (computed here), DATASET, N/A (not in the data), TRUTH (known label, hidden by default).
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
