# Digital Manipulation Radar — Technical Plan (5 days, 4 people)

Status: Day 0. Everything under "Verified dataset facts" was checked against the
Zenodo record, the official datasheet PDF inside it, the paper (arXiv 2410.19272,
ICWSM 2025) and the authors' notebooks. Every item marked **VERIFY D1** must be checked
against the real files on Day 1 before anyone builds on it.

---

## 1. Verdict on the proposal

**The problem is real and worth doing. The proposal as written cannot be built on this dataset.**

The proposal promises content similarity, shared URLs, synchronized posting times, and
"groups that post similar text and share the same link." The OSoMe dataset has **none of
the raw material for that**: it contains no reply text, no URLs, and no per-post timestamps.
What it does have is better for a 5-day ML project: **real labels** (accounts that Twitter
suspended as part of state-backed influence operations) plus **per-account features
already computed from text similarity, reply delay and engagement**, and the
**who-replied-to-which-tweet** structure needed for a real network.

So the project needs to be **narrowed and renamed** in scope:

> Detecting accounts that take part in **coordinated reply attacks**, meaning many
> influence-operation accounts replying to the same target tweet, and showing the
> co-reply groups they form.

This is a narrower claim, but it is defensible. The general claim ("detects digital
manipulation") is not.

### Things in the proposal to drop or change

| Proposal item | Problem | Change |
|---|---|---|
| TF-IDF / sentence embeddings on posts | No reply text in the release (only 36,885 translated *target* tweets for Serbia/Egypt) | Use the released cosine-similarity **summary features**; no text model |
| "Repeatedly sharing the same URLs" | No URLs, only a per-reply URL count summarized into features | Drop as a signal; mention as future work |
| "Posts within the same time window" | No timestamps, only reply delay relative to the target tweet | Use delay features; no time-window clustering |
| DBSCAN / clustering accounts as the main ML | Labels exist, so unsupervised-only throws away the most valuable thing in the dataset | Supervised classifier as the core; graph communities as a second, unsupervised view |
| XGBoost + SHAP + Sentence Transformers | Added complexity, no evidence they are needed | sklearn `HistGradientBoostingClassifier` (same family as XGBoost, no extra dependency); permutation importance; SHAP only if time allows |
| "Coordination score / risk level" | A classifier probability is not a calibrated risk | Show a **ranked review queue** with the model score and the evidence; avoid the word "risk" |
| Random train/test split | Leaks campaign style into the test set | **Leave-one-campaign-out** (Section 4) |

### The honest problem: this has been published already

The authors already trained a Random Forest on exactly these features (replier AUC 0.97,
tweet AUC 0.88) and ran a cross-campaign test. If you only re-train RF on their CSV, your ML
contribution is a **replication**. That is fine as a starting point, but the project needs
something of its own. Three things you can defend as yours, all doable in 5 days:

1. **Evaluate at real-world prevalence.** The paper reports precision 0.93 on a
   **downsampled 1:1** dataset. In the real data the ratio is about **1 IO replier to 114 normal
   repliers** (7,670 vs 874,248). Precision at 1:1 says little about how a tool performs on
   a real feed. You report **PR-AUC, precision@k and recall at fixed precision on the
   untouched held-out campaign**. This will probably be a much lower and much more honest
   number, which is a finding in itself.
2. **Leave-one-campaign-out with Saudi Arabia as a held-out campaign.** The dataset includes
   `saudi_arabia_112019` and `egypt_022020`. This is the closest you can get to the Saudi
   context without collecting new data. The question becomes: *"If we had never seen this
   campaign, would we still have caught its accounts?"*
3. **Do network features add signal?** Build a co-reply graph (accounts linked when they
   reply to the same tweets), compute **label-free** graph features (degree, weighted degree,
   number of distinct co-repliers, community size) and measure the **ablation**:
   behavioral features alone vs. behavioral + graph. Whichever way it comes out, it is a
   result.

---

## 2. Verified dataset facts

### Checked against the downloaded files (Day 0, full output in `docs/DATA_INSPECTION.txt`)

* Download from **record 13896309** (13896308 is the concept DOI; its file URLs return 404).
* `RQ3_replier_classifier_features.csv`: **794,645 repliers** (not 881,918 as in the datasheet):
  **7,670 IO / 786,975 normal → prevalence 0.97 %**. Has `replier_userid` and `replier_label`.
  76 features: `age`, `followers_ratio`, `following_ratio`, `activity_rate` + 8 attributes
  (`cosine`, `diff_min`, `like_count`, `retweet_count`, `reply_count`, `mention_count`,
  `hashtags_count`, `num_url`) × 9 statistics (`mean`, `min`, `25%`, `50%`, `75%`, `max`,
  `range`, `iqr`, `entropy`).
* `poster_tweetid_campaign_type.csv`: **`campaign` is the replier's own campaign; every
  normal replier is `remain`.** It cannot be used directly to split normal accounts by
  campaign. Fix (in `data.tweet_campaigns`): a target tweet's campaign = campaign of the IO
  accounts that replied to it; a normal account gets the campaign of the tweets it replied to most.
* Resulting LOCO groups (accounts / IO / prevalence): saudi_arabia_112019 482,243 / 1,454 / 0.30 %;
  turkey_052020 145,585 / 999 / 0.69 %; sa_eg_ae_022020 76,808 / 735 / 0.96 %;
  other 70,738 / 2,455 / 3.5 %; serbia_022020 13,011 / 1,152 / 8.9 %; honduras_022020 6,260 / 875 / 14 %.
  **Prevalence differs by 50× between folds. Compare each fold's PR-AUC with its own prevalence.**
* `RQ1_time_difference_of_reply.csv` has 2,147,936 rows (datasheet says 148,450).
* Co-reply graph (target tweets, repliers with ≥ 2 tweets): ≥ 2 shared tweets → 4.1 M edges;
  **≥ 3 → 831 k edges / 20 k accounts (used)**; ≥ 5 → 144 k.

### From the datasheet and paper

Source: Zenodo DOI 10.5281/zenodo.13896308 (record 13896309 holds the same files).
**License: CC BY 4.0** (open access; attribution required). Code: github.com/osome-iu/io-coordinated-replies.
Data covers **Twitter influence operations from 43 state-backed datasets, 2018–2021**;
normal replies and control tweets were collected in spring 2023.

| File | Size | One row = | Columns (from the datasheet) | Use it? |
|---|---|---|---|---|
| `RQ3_replier_classifier_features.csv` | 328 MB | one replier | 76 features (4 profile + 8 reply attributes × 9 statistics) + `replier_label` (1 = IO, 0 = normal). 881,918 rows. **VERIFY D1: exact column names, whether `replier_userid` is present** | **Core: model training** |
| `RQ3_replier_info.csv` | 286 MB | one replier | `replier_userid`, `activity_count`, `replier_label`, `following_count`, `followers_count`, `age` | Dashboard profiles |
| `poster_tweetid_campaign_type.csv` | 190 MB | (target tweet, reply) pair | `poster_tweetid`, `campaign`, `replier_userid`, `replier_label`, `replier_tweetid`, `type` (target/control). 2,673,091 rows | **Core: campaign labels + co-reply graph** |
| `RQ2_tweet_classifier_features.csv` | 7.7 MB | one target or control tweet | 99 features + `tweet_label`. 7,732 rows | Optional second model |
| `RQ2_engagement.csv` | 0.3 MB | one tweet | `tweetid`, retweet/like/quote count, `type` | Dashboard |
| `RQ1_time_difference_of_reply.csv` | 48 MB | one IO reply | `replier_tweetid`, `diff_min` (minutes after target tweet). 148,450 rows | Timeline chart |
| `RQ1_num_targeted_tweet_by_IO.csv` | 4.7 MB | (target user, IO replier) | `poster_userid`, `replier_userid`, `count` | Dashboard |
| `RQ1_target_non_target_tweets.csv` | 13 MB | one target tweet (Serbia/Egypt), English translation | `tweetid`, `english_text`, `type`, `campaign` | Demo: "what was attacked" |
| `RQ1_target_annotation.csv` | 55 KB | one annotated target (Serbia/Egypt) | `userid`, profession, country, `campaign` | Demo |
| `pos_cosine_with_replier_info.pkl.gz` | **3.8 GB** | a *pair* of replies | 1.26 **billion** rows | **Do not download.** Too large for laptops |

What the eight reply attributes are (paper, Table 1): engagement (like / retweet / reply /
quote counts of the reply), entities (hashtag / mention / URL counts), **delay** (time from
the target tweet to the reply), and **cosine** similarity between replies to the same tweet
(multilingual sentence embeddings). Each is summarized per replier by 9 statistics (range,
quartiles, IQR, mean, std, skew, kurtosis, entropy).

Paper results to compare against (Random Forest, 10-fold CV, **balanced data**):
replier classifier P 0.93 / R 0.92 / F1 0.92 / AUC 0.97; tweet classifier F1 0.80 / AUC 0.88.
The most important replier feature is the **cosine similarity** set. Cross-campaign F1 stays
high except for models **trained on Serbia** (F1 about 0.48–0.76 on other campaigns).

Campaigns used in the paper's classification sets: `saudi_arabia_112019`, `serbia_022020`,
`turkey_052020`, `egypt_022020`, `honduras_022020`, and `remain` (all others pooled).
**VERIFY D1** that the `campaign` column uses these names.

### Limitations you must state in the presentation

* **Labels are "suspended by Twitter as part of an IO," not "coordinated."** The "normal" class
  surely contains some undetected coordinated accounts (label noise), and some IO accounts
  in the data were not coordinating on a given tweet.
* **Selection bias by construction.** A "targeted tweet" is *defined* as one that got at
  least 5 IO replies. So IO accounts co-reply on these tweets **because of how the dataset was
  built**. A graph that "finds" IO clusters on target tweets is partly rediscovering the
  selection rule. Always compare with the **control** tweets and never present community
  purity as proof the graph detects coordination.
* **The features were precomputed by the authors.** You cannot change how cosine or delay
  were computed, and you cannot recompute them on new data without raw tweets.
* **Old platform data (2018–2021 Twitter).** Behaviour on today's X may differ.
* **Account age and activity may reflect suspension artifacts** (IO accounts are often new
  and less active). If profile features alone score very high, say so: the model may be
  learning "new low-activity account" rather than "coordination." The ablation in 4.4 checks this.
* **IDs are real Twitter IDs, not hashed.** Never display them next to accusations. The
  dashboard shows shortened IDs and the phrase "flagged for review."

---

## 3. What counts as ML here (and what doesn't)

| Component | Type | Why |
|---|---|---|
| Replier classifier (LR baseline → RF → HistGradientBoosting), LOCO evaluation, threshold chosen on training data only | **Supervised ML (the core contribution)** | Learned from labels, evaluated on unseen campaigns |
| Ablation: profile / reply features / + graph features | **ML experiment** | Measures where the signal comes from |
| Louvain community detection on the co-reply graph | **Unsupervised graph algorithm** | No labels used; you can check it against labels afterwards |
| Co-reply edge construction (shared target tweets ≥ k) | Rule-based analysis | A fixed rule, not learned |
| "Why flagged" panel (feature z-scores vs. normal accounts, permutation importance) | Explainability | Describes the model; does not make decisions |
| PyVis network, timelines, tables | Visualization | — |

In the presentation, say plainly: *"The detection is the classifier. The graph shows how
flagged accounts relate to each other and gives the model extra features. The rules only
decide what counts as a graph edge."*

---

## 4. ML pipeline

### 4.1 Unit of prediction
**One row per replier account** (`RQ3_replier_classifier_features.csv`). Label `replier_label`.
The tweet classifier (RQ2) is a stretch goal only; do not split the team's effort across both.

### 4.2 Campaign per account
Join `replier_userid` → `campaign` from `poster_tweetid_campaign_type.csv`. A normal replier can
reply to tweets from several campaigns: assign the **most frequent** one and record
`n_campaigns`. Group small campaigns into `other` (only the top 5 campaigns stay separate, as in the paper).

### 4.3 Splits (anti-leakage)
* **Primary: leave-one-campaign-out (LOCO).** Train on 5 campaign groups, test on the 6th, rotate.
  This is the headline result. The Saudi fold is the demo fold.
* **Replication: stratified 10-fold on a balanced sample.** Only to show you can reproduce
  about 0.97 AUC. Never present it as the real-world number.
* **Downsample negatives in the training folds only** (e.g. 10:1). The **test fold stays at its
  natural ratio.**
* **Decision threshold** is chosen inside the training folds (max F1 on an inner validation
  split), never on the test fold.
* **Graph features** are computed from the reply structure only (no labels), so computing
  them on the full graph does not leak labels. Write in the report that this is
  transductive (the test campaign's *unlabeled* structure is visible).
* Never use any feature derived from labels (e.g. "% of IO accounts in my community").
  The code refuses columns whose names contain `label`.

### 4.4 Models and experiments
1. `LogisticRegression` (standardized, class_weight balanced) — baseline.
2. `RandomForestClassifier(n_estimators=200)` — paper's model.
3. `HistGradientBoostingClassifier` — strongest tabular baseline, no extra dependency.

Ablation on the best model: profile-only (4 features) / reply-features-only / all /
all + graph. **If profile-only reaches almost the same score as everything, report it.**

### 4.5 Metrics (report all, at natural prevalence)
* **PR-AUC (average precision)** — primary; compare with the prevalence baseline (≈ 0.009).
* ROC-AUC — for comparison with the paper.
* Precision / recall / F1 at the training-chosen threshold.
* **Precision@100 and @500** — "if an analyst reviews the top 100 accounts, how many are real IO?"
* Confusion matrix for the Saudi fold.

### 4.6 Network
* Bipartite graph replier ↔ target tweet (target tweets only; control tweets used for comparison).
* Keep repliers that replied to ≥ 2 tweets (unlabeled rule; most normal accounts reply once).
* Project to replier–replier with sparse matrix multiplication; keep edges with ≥ 2 shared tweets.
* Louvain communities (`networkx.community.louvain_communities`, fixed seed).
* Per community: size, mean model score, shared targets, median delay, label purity
  (**purity shown only on the evaluation page, with the selection-bias warning**).

---

## 5. MVP (must be finished by end of Day 4)

1. `scripts/download_data.py` + `scripts/inspect_data.py` run on the real files.
2. `python -m radar.pipeline` produces `artifacts/`: LOCO metrics table, per-account scores
   (out-of-fold), graph communities, permutation importance.
3. Streamlit app with 4 pages: **Overview, Model evaluation, Coordination network, Group &
   account investigation**, reading only from `artifacts/` (no training inside the app).
4. A 5-minute demo on the Saudi held-out fold.

Stretch (only if MVP is done): tweet classifier; SHAP; Serbia/Egypt target-tweet text in the
investigation page; Arabic UI labels.

Out of scope: live data collection, raw text models, URL analysis, real-time anything.

---

## 6. Team breakdown

| | **M1 — Data** | **M2 — ML** | **M3 — Network** | **M4 — App** |
|---|---|---|---|---|
| **D1** | Download (skip the 3.8 GB pickle), run `inspect_data.py`, fix column names in `config.py`, write `docs/DATA.md` (rows, columns, nulls, label counts per campaign) | Read paper RQ3 + notebook 9; run the pipeline on the synthetic sample; prepare LOCO code against real columns | Read paper; check `poster_tweetid_campaign_type` (repliers per tweet distribution) and decide `min_tweets` / `min_shared` | Streamlit skeleton runs on sample artifacts; page layout agreed |
| **D2** | Campaign join + `n_campaigns`; sanity checks (no replier in two label classes, duplicates) | LR + RF + HGB under LOCO; first metrics table | Build co-reply graph per campaign; Louvain; timing on full data | Overview + Evaluation pages on real artifacts |
| **D3** | Data limitations section; check age/activity artifact (profile-only ablation together with M2) | Ablation (profile / reply / all / +graph); permutation importance; threshold at natural prevalence; precision@k | Graph features → hand to M2; community summary table; control-vs-target comparison | Network page (PyVis, ≤ 300 nodes, campaign filter) |
| **D4** | Integration; freeze `artifacts/`; README reproduction steps | Final model on all data for the dashboard scores; error analysis (top false positives) | Group investigation data (shared targets, delays from `RQ1_time_difference_of_reply`) | Group & Account pages; "why flagged" panel; wording review |
| **D5** | Slides: data + limitations | Slides: method + results | Slides: network | Demo rehearsal ×3; screen recording as backup |

Rule: **nobody hard-codes a number into the slides or app.** Every figure is read from `artifacts/`.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Column names differ from the datasheet | High | Day-1 `inspect_data.py`; all names live in `config.py` |
| `replier_userid` missing from the features file | Medium | Do **not** join by row order unless you prove it matches `RQ3_replier_info.csv` (same row count and identical label sequence). Without a join there is no campaign split: report it and fall back to stratified CV with that caveat |
| Projected graph too large (popular tweets with thousands of repliers) | Medium | `min_tweets ≥ 2`, `max_repliers_per_tweet` cap, per-campaign graphs |
| Results look too good (AUC ≈ 0.99) | Medium | Check the profile-only ablation and the per-campaign breakdown before celebrating |
| Results look bad at natural prevalence | High | That is a valid finding: present precision@k as "review queue quality" |
| **Network features look near-perfect** (seen on the synthetic data: "network only" PR-AUC ≈ 0.98). **Real data: did not happen** (network only = 0.15 mean PR-AUC, see RESULTS.md) | Was high, now low | Most likely an **observation artifact**, not coordination: IO replies come from Twitter's full IO dump, while normal accounts are only seen on the sampled tweets, so their co-reply degree is low by construction. Report "network only" as a **warning sign**, not a win. Check it: compare degree distributions on control tweets vs target tweets, and retest with graph features computed only from tweets that both groups replied to |
| Laptop RAM (≈ 900 MB CSVs) | Low–medium | Read with explicit `usecols`/dtypes; cache as Parquet after Day 1 |
| Over-claiming in the demo | Medium | Fixed wording: "flagged for review", "possible coordination", never "bot" or "fake" |

---

## 8. Demo scenario (5 minutes)

1. **The question (30 s):** "Suppose a new state-backed campaign starts and we have never seen it.
   Can a model trained on *other* campaigns find its accounts?"
2. **Hold out Saudi Arabia (1 min):** Evaluation page → Saudi fold. Show PR-AUC vs. the 0.9%
   prevalence baseline and precision@100: "of the 100 accounts we would send to a reviewer,
   N were really part of the operation." Numbers come from `artifacts/`.
3. **One target tweet (1.5 min):** Pick a targeted tweet from the Saudi fold → Network page:
   flagged accounts light up and sit in the same community; replies arrived within X minutes
   (delay chart).
4. **Why flagged (1 min):** Account page: reply similarity much higher than normal repliers,
   short delays, etc. Show the feature-group importance chart.
5. **Honesty slide (1 min):** selection bias, label noise, the 1:1 vs real-prevalence gap,
   "coordination ≠ malicious intent," and what Arabic/Saudi data would be needed to go further.
