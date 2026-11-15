# Digital Manipulation Radar: project report

This report follows the seven steps the project brief asks for. Each step says **what we did** and **why**.
Every number comes from a file in the repo:

- `artifacts/metrics.csv` and `artifacts/importance.csv`, written by `python -m radar.pipeline`;
- `models/*.json`, written when the deployed models are trained;
- `notebooks/01_eda.ipynb`, the exploratory analysis.

| Step | Where to look |
|---|---|
| 1 · Problem definition | this file §1, [PLAN.md §1](PLAN.md) |
| 2 · EDA | [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) |
| 3 · Preprocessing | `src/radar/data.py`, `src/radar/graph.py`, `src/radar/model.py` |
| 4 · Modeling | `src/radar/model.py`, `src/radar/simple.py` |
| 5 · Evaluation | [RESULTS.md](RESULTS.md), `artifacts/metrics.csv` |
| 6 · Model selection | this file §6 |
| 7 · Deployment | https://digital-manipulation-radar.streamlit.app, `app/`, `src/radar/deploy.py` |

---

## 1 · Problem definition

**The problem.** In a *coordinated reply attack*, a state-backed influence operation (IO) picks a tweet
(a journalist, an activist, a minister) and floods it with replies from accounts that work together. To a reader
this looks like public opinion. Platforms find these networks late, and analysts check accounts one by one.

**The ML task.** Binary classification at the **account** level: given how an account replied (timing, text
similarity, profile, and which other accounts it replied alongside), how likely is it that the account belongs
to an operation? The output is a **score used to rank accounts for a human to review**, not a verdict.

**The data.** OSoMe *Coordinated Reply Attacks in Influence Operations* (Pote et al., ICWSM 2025),
Zenodo 10.5281/zenodo.13896308, CC BY 4.0. It contains:

- 2.67 M replies from 794,645 accounts to 3,866 attacked tweets and 3,866 control tweets (Twitter, 2018–2021);
- 76 per-account features precomputed by the authors;
- labels from Twitter's takedown lists: 7,670 accounts were suspended as part of an IO.

**Why this problem, and why this framing:**

- **Real labels.** The labels come from the platform, not from us, so the model can be tested honestly.
- **Narrowed scope.** We kept to what the data supports: coordinated *replies*. We dropped the original
  proposal's URL sharing and time-window clustering because the dataset has no URLs or raw timestamps
  ([PLAN.md §1](PLAN.md)).
- **Ranking, not labelling.** With 1 IO account per 100 (see EDA), a yes/no label would mostly produce false
  accusations. A ranked list that a person reviews is both more useful and more responsible.

**Our own contribution beyond the paper:**

- **Testing on unseen campaigns.** We evaluate on campaigns the model never saw, at their real (tiny)
  share of IO accounts. The paper's headline uses a balanced random split.
- **A co-reply network.** It adds features and groups for review.
- **A deployed tool.** It scores new data, including a plain replies file a user can export or type in.

---

## 2 · Exploratory data analysis

Full analysis with charts: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). The findings that drove later
decisions:

| Finding | Number | What it changed |
|---|---|---|
| Extreme class imbalance | 7,670 IO of 794,645 accounts = **0.97 %** | Metrics (§5), training sampling (§3), "rank for review" product (§7) |
| Campaigns differ in prevalence | **0.30 %** IO (Saudi Arabia) to **14 %** (Honduras) | Per-campaign reporting; PR-AUC read against prevalence |
| Campaigns differ in style | IO median delay **1 h** (Honduras) vs **13.5 h** (Saudi/Egypt/UAE); IO account age 0.12 vs 4.7 years | Leave-one-campaign-out evaluation (§5) |
| `diff_min` is in **hours**, not minutes | recomputed from tweet IDs: 96.5 % match in hours, 0.7 % in minutes | Hours everywhere; the app rebuilds delays in hours |
| Text similarity is the strongest single signal | `mean_cosine` alone: ROC-AUC **0.90** | Keep it; check how much the model depends on it (§5) |
| …but zeros act as placeholders | similarity is exactly 0 for **80 %** of normal vs **9 %** of IO accounts | Stated as a caveat; checked with a no-text run |
| IO accounts are young | median age **0.48** vs **2.6** years | Profile features kept; age used in the simple model |
| IO accounts reply in groups | **29 %** of IO vs **2.3 %** of normal accounts are in the co-reply network | Network features + groups in the app |
| No non-text feature separates well alone | all ≤ 0.69 ROC-AUC | Non-linear tree models (§4) |
| No missing or infinite values in the release | 0 of 79 columns | No imputation needed for trees; defensive cleaning for uploads |
| Many near-duplicate features | 60 pairs with \|Spearman\| > 0.95 | Kept (trees tolerate it); importance measured per feature *group* |

---

## 3 · Preprocessing

| What we do | Why |
|---|---|
| **One row per account**, checked for duplicates (`data.load_features`) | The account is what we classify. A duplicate would leak one account into both train and test. |
| **±∞ → missing** in every numeric column | The release has none, but uploaded files can (ratios divided by 0). Models must not crash on a user's file. |
| **Identifier and label columns are never features** (`feature_columns` refuses anything named like an ID or a label) | Prevents leakage: an ID or a label among the inputs would give a perfect, meaningless score. |
| **Each account gets a campaign** = the campaign of the attacked tweets it replied to most (`account_campaigns`). Campaigns outside the top 5 are pooled into `other`. | Normal accounts have no campaign in the file (they are all `remain`). We need one for every account to hold whole campaigns out for testing. Small campaigns are pooled because they are too small to test on alone, as in the paper. |
| **Co-reply network** (`graph.py`): accounts with ≥ 2 attacked tweets; a link needs ≥ 3 shared tweets; tweets with > 2,000 repliers skipped; Louvain communities | ≥ 2 shared tweets gave 4.1 M mostly random links, while ≥ 3 gives 831 k. Viral tweets would join thousands of unrelated accounts. The network uses **no labels**, so it is safe to build on the test campaign too. |
| **Negatives downsampled to 10 per positive, in training folds only** (`model.downsample`) | 99 % normal accounts make training slow and push the model towards "always normal". Test folds keep the real ratio, so the measured performance is the real-world one. 10:1 (not the paper's 1:1) keeps more normal variety. |
| **Median imputation + standard scaling for Logistic Regression only**, inside an sklearn `Pipeline` | A linear model needs comparable scales and no gaps. Doing it inside the pipeline fits it on training data only. Tree models need neither. |
| **Simple-input model:** delays computed as `ceil((reply time − tweet time) / 1 h)` from tweet IDs or timestamps; followers and following log-transformed | The same unit and rounding as the released data (EDA §5). Otherwise a user's file would be on a different scale from training. Log because follower counts span 6 orders of magnitude. |
| **Example accounts shown in the app are removed from training** (1,948 accounts) | So that the examples a grader tries in the app were never seen by the model. |

---

## 4 · Modeling

We trained three model families on the same features and the same splits:

| Model | Settings | Why we tried it |
|---|---|---|
| **Logistic Regression** | balanced class weights, imputation + scaling | Simple, interpretable **baseline**. If it were as good, we would pick it. |
| **Random Forest** | 200 trees, `min_samples_leaf=2` | The paper's model, so we can **replicate** their result. Handles non-linear interactions. |
| **HistGradientBoosting (HGB)** | sklearn defaults | Gradient-boosted trees: usually strongest on tabular data, fast on 800 k rows, handles missing values natively. It is the same family as XGBoost with no extra dependency. |

**Feature sets (ablation).** Run with the best model to learn *where the signal is*:

- profile only;
- network only;
- reply features only;
- all 76 released features;
- all + network;
- all without text similarity.

**Threshold.** Inside each training fold, 20 % of the training rows are held out. We pick the threshold that
maximises F1 there and then refit on all training rows. The test campaign is never used to choose anything.

**Two deployed models:**

1. **Full model:** HGB on the 76 released features, trained on all campaigns (`radar/deploy.py`). It is for data
   in the dataset's format.
2. **Simple-input model:** HGB on 17 features that can be rebuilt from a plain replies file (`radar/simple.py`):
   - 9 delay statistics;
   - account age, followers and following;
   - number of replies and tweets;
   - 3 network features.

   Why a second model: real users do not have the authors' 76 precomputed features (the text similarity needs
   their LaBSE pipeline over all replies). Without it the tool would only work on the dataset itself.

---

## 5 · Evaluation

### Design: leave-one-campaign-out (LOCO)

We train on five campaign groups and test on the sixth, then repeat for each group. **Why:** each operation
has its own style (EDA §11). A random split puts the same campaign in train and test, so the model learns that
campaign's habits and looks better than it is. The real use is spotting a *new* operation, and LOCO simulates
exactly that.

### Metrics

| Metric | Why |
|---|---|
| **PR-AUC (average precision)**: the headline metric | It measures how well the rare IO accounts are ranked at the top, at the real class ratio. Random ranking scores the prevalence (0.003 on Saudi). |
| **Precision@100 / @500** | It answers the analyst's question: "if I review the top 100, how many are real?" |
| ROC-AUC | Reported for comparison with the paper, but not used to choose: it stays high (≈ 0.93) even when most flagged accounts are false alarms. |
| Precision / recall / F1 at the threshold | They show what automatic labelling would do. |
| **Accuracy: not used** | Saying "normal" for everyone is 99 % accurate. |

### Results (mean over the 6 held-out campaigns)

| Model | PR-AUC | ROC-AUC | Precision@100 | Precision@500 | F1 at threshold |
|---|---|---|---|---|---|
| Logistic Regression | 0.359 | 0.892 | 0.66 | 0.51 | 0.31 |
| Random Forest | 0.481 | 0.925 | 0.88 | 0.64 | 0.43 |
| **HistGradientBoosting** | **0.536** | **0.928** | **0.93** | **0.72** | **0.43** |

Per campaign (HGB):

| Campaign | Prevalence | PR-AUC | Precision@100 |
|---|---|---|---|
| Saudi Arabia | 0.30 % | 0.417 | 0.87 |
| Turkey | 0.69 % | 0.491 | 0.91 |
| Saudi/Egypt/UAE | 0.96 % | 0.501 | 0.95 |
| other | 3.5 % | 0.664 | 0.99 |
| Serbia | 8.9 % | 0.448 | 0.90 |
| Honduras | 14 % | 0.696 | 0.98 |

The full table is in [RESULTS.md](RESULTS.md).

**Sanity check: replication.** In the paper's own setup (balanced 1:1, random 10-fold, Random Forest) we get
ROC-AUC 0.975 and F1 0.927 (paper: 0.97 / 0.92). The pipeline is correct; the lower LOCO numbers are the honest
picture, not a bug.

**Where the signal comes from** (mean PR-AUC, HGB):

| Feature set | Mean PR-AUC |
|---|---|
| profile only | 0.112 |
| network only | 0.148 |
| all *without* text similarity | 0.144 |
| reply features only | 0.503 |
| all 76 released features | 0.536 |
| all + network | **0.579** |

Permutation importance agrees: shuffling the text-similarity group costs **0.52** PR-AUC, profile 0.13,
delay 0.08, engagement and entities ≈ 0.02.

**What the evaluation tells us:**

1. **Ranking works.** On Saudi Arabia (1 IO account in 330), 87 of the top 100 are real IO accounts:
   **138× better than random**.
2. **Automatic labelling does not.** At the threshold, Saudi precision is only **13.5 %** (7,276 false alarms
   for 1,139 hits), because the threshold was learned where IO accounts were far more common. So the tool shows a
   ranked queue and asks a person to decide.
3. **The model is not just finding "new accounts".** Profile alone reaches only 0.11.
4. **Text similarity carries most of the signal.** Without it, PR-AUC falls to 0.14. Because 80 % of normal
   accounts have a placeholder similarity of 0, part of that signal may come from how the authors computed the
   feature. Performance on freshly collected data could be lower, and we say so in the app.
5. **The simple-input model is honestly weaker:** mean PR-AUC **0.188**, precision@100 from 0.28 to 0.78 by
   campaign. It is still 2–22× better than random (PR-AUC divided by prevalence). In the app it is labelled as the weaker model.

### Limitations

- **Label noise.** The labels mean "suspended as part of an IO", not "coordinating": the normal class surely
  contains undetected operation accounts.
- **Selection bias.** An attacked tweet is *defined* as one with ≥ 5 IO replies, so IO accounts co-reply partly
  by construction.
- **Old data.** Twitter 2018–2021; behaviour on today's X may differ.
- **Transductive network.** The co-reply network for a held-out campaign uses that campaign's (unlabelled)
  replies.

---

## 6 · Model selection

**Chosen: HistGradientBoosting.**

| Criterion | LogReg | RF | **HGB** |
|---|---|---|---|
| Mean PR-AUC (headline) | 0.359 | 0.481 | **0.536** |
| Best on how many of 6 campaigns (PR-AUC) | 1 | 1 | **4** |
| Mean precision@100 | 0.66 | 0.88 | **0.93** |
| Handles missing values in uploads without imputation | no | no | **yes** |
| Model file size / scoring speed | small / fast | large / slower | **small / fast** |

**Why not the others:**

- **Logistic Regression** cannot combine signals non-linearly (e.g. *young* **and** *fast* **and** *similar
  text*). The EDA showed no single feature is enough on its own.
- **Random Forest** is lower on every campaign except Serbia (0.534 vs 0.448). Its model files are larger, and
  it needs imputation for incomplete uploads.
- Logistic Regression edges out HGB on Honduras only (0.704 vs 0.696), the campaign where IO accounts are
  most common. It falls far behind where they are rare (Saudi Arabia: 0.069 vs 0.417), and rare is the real-world case.

**Why not "all + network" for the deployed full model:**

- It is better in evaluation (0.579), and the app *does* use network features and groups.
- But network features need all replies to the same tweets. A file in the dataset format holds one row per
  account, with no reply structure to build them from.
- So the deployed full model uses the 76 features any such file has. The simple-input model, which receives
  raw replies, *does* include the network features.

**Why a threshold at all:** it marks where the review queue starts. The app always shows the score and the
rank, and states that most accounts above the threshold are false alarms on real data.

---

## 7 · Deployment

**Live app:** https://digital-manipulation-radar.streamlit.app (Streamlit Community Cloud, free tier). We chose
Streamlit because:

- the brief lists it as an acceptable tool;
- it gives a working interface in Python, the same language as the models;
- it is free to host for anyone with the link.

**It is a usable tool, not only a dashboard.** The flow is **Upload → Results → (optional) Review → Report**:

| Input | Model used |
|---|---|
| A campaign from the dataset | Scores from the LOCO run: each account scored by a model that never saw its campaign |
| **Type it in:** a table in the app (tweet link, reply link, account, time, followers, …) | simple-input model |
| **Replies file:** a plain CSV/Excel export, many column names understood, X links accepted | simple-input model |
| **Dataset format:** a file with the 76 released features | full model |

**The output:**

- the accounts ranked by score, with the main reasons for each (which features are unusual compared with
  normal accounts);
- the groups that keep replying together;
- a "the accounts you asked about" verdict when the user names one or two accounts;
- downloads: an HTML report and a CSV.

**The optional review step:** an analyst can mark accounts as suspicious, normal or unsure, and download a
signed-off report.

**Design choices and why:**

| Choice | Why |
|---|---|
| Scores + reasons + "for review" wording, never "fake" | A score is not proof (precision at the threshold is low on real data, §5). Accusing real people from a model score would be harmful. |
| Plain-file and type-in inputs | Real users do not have 76 precomputed features. A tool that only runs on the dataset is not usable. |
| Lite data copy for hosting | The full artifacts need ≈ 1.3 GB RAM. The hosted app uses a smaller copy, `scripts/make_lite.py`, which gives identical results for the dataset campaigns. |
| No live lookup of X accounts | The X API is paid per post read and collecting people's data raises consent issues. Users bring their own export or type in what they see. |
| Example accounts held out of training | So the demo examples are a fair test. |

**Reproduce everything:** `README.md` → *Run*. The steps are:

1. download the data;
2. `python -m radar.pipeline` (≈ 5 min);
3. `python -m radar.deploy` and `python -m radar.simple` to train the deployed models;
4. `streamlit run app/main.py`.
