# Results — first full run on the real data (2026-11-11)

Source of every number: `artifacts/metrics.csv`, `artifacts/importance.csv`,
`artifacts/run_info.json` (run log: `artifacts_run.log`). Re-run with `python -m radar.pipeline`.
Settings: training negatives downsampled to 10 per positive, test folds untouched,
threshold chosen on training campaigns, co-reply edges need ≥ 3 shared target tweets.

## 1. Replication of the paper (balanced 1:1, 10-fold CV, Random Forest)
ROC-AUC **0.975**, F1 **0.927** (paper: 0.97 / 0.92). The pipeline reproduces the published result.

## 2. Unseen campaign, real class ratio (HistGradientBoosting, all 76 released features)

| Held-out campaign | Accounts | IO | Prevalence | PR-AUC | Lift | ROC-AUC | Precision@100 | Precision@500 |
|---|---|---|---|---|---|---|---|---|
| saudi_arabia_112019 | 482,243 | 1,454 | 0.30 % | **0.417** | 138× | 0.964 | **0.87** | 0.66 |
| turkey_052020 | 145,585 | 999 | 0.69 % | 0.491 | 72× | 0.962 | 0.91 | 0.68 |
| sa_eg_ae_022020 | 76,808 | 735 | 0.96 % | 0.501 | 52× | 0.938 | 0.95 | 0.64 |
| other (pooled) | 70,738 | 2,455 | 3.5 % | 0.664 | 19× | 0.935 | 0.99 | 0.97 |
| serbia_022020 | 13,011 | 1,152 | 8.9 % | 0.448 | 5× | 0.850 | 0.90 | 0.60 |
| honduras_022020 | 6,260 | 875 | 14 % | 0.696 | 5× | 0.917 | 0.98 | 0.74 |

Mean PR-AUC by model: Logistic Regression 0.359, Random Forest 0.481, **HistGradientBoosting 0.536**.

**Reading it:** ROC-AUC stays near the paper's 0.97, but that hides the class imbalance.
On the Saudi campaign, 87 of the top 100 accounts the model ranks are real IO accounts
(random ranking: 0.3 of 100). At the automatic threshold, precision drops to only **13.5 %**
(recall 78 %): most flagged accounts would be false alarms. **A ranked review queue works;
automatic labelling does not.**

## 3. Where the signal comes from (mean PR-AUC across the 6 held-out campaigns)

| Feature set | Mean PR-AUC | Saudi PR-AUC |
|---|---|---|
| profile only (age, follower/following ratio, activity) | 0.112 | 0.034 |
| network only (co-reply degree, strength, community size) | 0.148 | 0.077 |
| reply features only (72) | 0.503 | 0.131 |
| all released features (76) | 0.536 | 0.417 |
| **all + network** | **0.579** | **0.477** (Precision@100 0.94) |

Permutation importance (drop in PR-AUC when shuffled): **text similarity 0.52**, profile 0.13,
reply delay 0.08, engagement 0.02, entities 0.02.

**Reading it:**
* The model is **not** just spotting "new, low-activity accounts": profile features alone are weak.
* **Similarity between replies is the main signal**, as in the paper.
* The **co-reply network adds a modest but consistent gain** (+0.04 mean, +0.06 Saudi). On its
  own it is weak, so the selection-bias worry raised on synthetic data did not show up here.
* On Saudi, reply features alone (0.131) are much weaker than reply + profile (0.417):
  the two kinds of features complement each other on this campaign.

## 4. Added after the EDA (2026-11-15)

**Without text similarity** (HGB, LOCO, all released features except the nine `*_cosine` ones):
mean PR-AUC **0.144** (vs 0.536), precision@100 0.08–0.49 by campaign. Reproduced in
`notebooks/radar_notebook.ipynb` §7.1. The released similarity is exactly 0 for 80 % of normal accounts
and 9 % of IO accounts, so part of this signal may reflect how the feature was computed.

**Simple-input model** (17 features rebuildable from a plain replies file; `radar.simple`):
mean PR-AUC **0.188**; precision@100 from 0.28 (Saudi/Egypt/UAE) to 0.78 (Serbia).

**Deployed models** are retrained on all campaigns with the 1,948 accounts used in the app's
examples held out, so the examples are unseen.

Every choice and its reason: [REPORT.md](REPORT.md).

## 5. Caveats
* The campaign of a normal account is inferred from the IO accounts that replied to the same tweets.
* The decision threshold is miscalibrated across campaigns, because prevalence varies 50× between folds.
* The network uses the unlabelled reply structure of the held-out campaign (transductive).
* Labels are "suspended as part of an IO" and are noisy. The data covers Twitter, 2018–2021.
