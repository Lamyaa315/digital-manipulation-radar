# ClueX

Finds accounts that take part in **coordinated reply attacks** — state-backed operations that flood a
chosen tweet with replies from accounts working together — and shows the groups they form.
Results mark **possible coordination for a person to check**, not proof of fake accounts or intent.

**Team:** Lamyaa Alnahdi · Raghad Alqahtani · Rawan Alsubaie — Machine Learning Unit 4 & 5, final project.

| | |
|---|---|
| **Live app** | https://cluex-tool.streamlit.app |
| **Data** | OSoMe, *Coordinated Reply Attacks in Influence Operations* (Pote et al., ICWSM 2025) — [Zenodo 10.5281/zenodo.13896308](https://doi.org/10.5281/zenodo.13896308), CC BY 4.0. Not stored in this repo (≈ 880 MB); `python scripts/download_data.py` fetches it. |
| **Notebook** | [notebooks/radar_notebook.ipynb](notebooks/radar_notebook.ipynb) — EDA, the three models, accuracy / precision / recall / F1, model selection |
| **Report** | [docs/REPORT.md](docs/REPORT.md) — the seven project steps and the reason behind every choice |
| **Results** | [docs/RESULTS.md](docs/RESULTS.md) — full evaluation tables |

## Results in one table

Gradient Boosting, the chosen model:

| Test | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random 20 % test set | 0.99 | 0.45 | 0.83 | 0.58 |
| **Campaigns the model never saw** (what we report) | 0.95 | 0.35 | 0.67 | **0.43** |

A model that calls every account “normal” also reaches 0.95–0.99 accuracy and finds nothing, so models are
compared on F1 and PR-AUC. Details and the other two models: the notebook.

## Repository

```
app/            the Streamlit app (main.py; views: home, upload, results, about)
src/radar/      pipeline: loading, co-reply network, models, evaluation, deployed models
models/         the two trained models + example files used by the app
artifacts_lite/ small copy of the pipeline outputs, used by the hosted app
notebooks/      the project notebook
docs/           REPORT.md, RESULTS.md
scripts/        download the data, build the lite copy, rebuild the notebook
```

## Run it yourself

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_data.py
```

```bash
PYTHONPATH=src python -m radar.pipeline
```

```bash
streamlit run app/main.py
```

The pipeline takes about 5 minutes and writes `artifacts/`; the app then needs about 1.3 GB of RAM.
Without the downloaded data the app runs on `artifacts_lite/` (identical scores and groups).
On Windows PowerShell use `$env:PYTHONPATH = "src"` instead of the `PYTHONPATH=src` prefix.

Re-train the deployed models: `python -m radar.deploy` (full model) and `python -m radar.simple`
(simple-input model for plain replies files). Rebuild the notebook:
`jupyter nbconvert --to notebook --execute --inplace notebooks/radar_notebook.ipynb`.
