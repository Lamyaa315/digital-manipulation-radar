"""Generate a SMALL SYNTHETIC dataset with the same file names and columns as the real one.

It exists only so the pipeline and the dashboard can be developed and tested before the
real data is downloaded. Numbers produced from it mean nothing. Never put them in slides.
    python scripts/make_sample_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar import config as C  # noqa: E402

rng = np.random.default_rng(0)
CAMPAIGNS = ["saudi_arabia_112019", "serbia_022020", "turkey_052020", "egypt_022020",
             "honduras_022020", "cuba_082020", "venezuela_062021"]
IO_PER_CAMP, NORMAL, TWEETS_PER_CAMP = 40, 4000, 30
ATTRS = ["cosine", "delay", "like_count", "retweet_count", "quote_count",
         "hashtag_count", "mention_count", "url_count"]
STATS = ["range", "q25", "q50", "q75", "iqr", "std", "skew", "kurtosis", "entropy"]

uid = iter(range(10**15, 10**16))
tid = iter(range(10**17, 10**18))

# tweets
tweets = []
for c in CAMPAIGNS:
    for kind in ("target", "control"):
        tweets += [(next(tid), c, kind) for _ in range(TWEETS_PER_CAMP)]
tweets = pd.DataFrame(tweets, columns=[C.COL_TWEET, C.COL_CAMPAIGN, C.COL_TYPE])

# replies
rows, users = [], []
for c in CAMPAIGNS:
    targets = tweets.query(f"{C.COL_CAMPAIGN} == @c and {C.COL_TYPE} == 'target'")[C.COL_TWEET].to_numpy()
    for _ in range(IO_PER_CAMP):
        u = next(uid)
        users.append((u, 1))
        for t in rng.choice(targets, rng.integers(3, 9), replace=False):
            rows.append((t, c, u, 1, next(tid), "target"))
for _ in range(NORMAL):
    u = next(uid)
    users.append((u, 0))
    for t in tweets.sample(rng.integers(1, 3), random_state=int(rng.integers(1e9))).itertuples():
        rows.append((getattr(t, C.COL_TWEET), getattr(t, C.COL_CAMPAIGN), u, 0, next(tid), getattr(t, C.COL_TYPE)))
replies = pd.DataFrame(rows, columns=[C.COL_TWEET, C.COL_CAMPAIGN, C.COL_USER, C.COL_LABEL,
                                      C.COL_REPLY, C.COL_TYPE])

# features: IO accounts are shifted on cosine/delay, weakly on profile; heavy overlap on purpose
users = pd.DataFrame(users, columns=[C.COL_USER, C.COL_LABEL])
n, io = len(users), users[C.COL_LABEL].to_numpy() == 1
feats = {C.COL_USER: users[C.COL_USER], C.COL_LABEL: users[C.COL_LABEL]}
shift = {"cosine": 1.2, "delay": -0.8, "like_count": -0.3, "retweet_count": 0.2}
for a in ATTRS:
    base = rng.normal(0, 1, n) + io * shift.get(a, 0.0)
    for s in STATS:
        feats[f"{a}_{s}"] = base + rng.normal(0, 0.7, n)
feats["age"] = rng.gamma(3, 600, n) * np.where(io, 0.6, 1.0)
feats["follower_rate"] = rng.lognormal(0, 1, n)
feats["following_rate"] = rng.lognormal(0, 1, n) * np.where(io, 1.3, 1.0)
feats["activity_rate"] = rng.lognormal(1, 1, n) * np.where(io, 0.5, 1.0)
feats = pd.DataFrame(feats)

info = users.assign(
    activity_count=rng.integers(10, 20000, n), following_count=rng.integers(0, 3000, n),
    followers_count=rng.integers(0, 5000, n), age=feats["age"].round().astype(int))
delays = replies.loc[replies[C.COL_LABEL] == 1, [C.COL_REPLY]].assign(
    **{C.COL_DELAY: rng.lognormal(4.5, 1.5, int((replies[C.COL_LABEL] == 1).sum())).round(1)})

C.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
feats.to_csv(C.SAMPLE_DIR / C.FEATURES_FILE, index=False)
replies.to_csv(C.SAMPLE_DIR / C.REPLIES_FILE, index=False)
info.to_csv(C.SAMPLE_DIR / C.REPLIER_INFO_FILE, index=False)
delays.to_csv(C.SAMPLE_DIR / C.DELAY_FILE, index=False)
(C.SAMPLE_DIR / "README.txt").write_text("SYNTHETIC DATA for code testing only. Not real accounts, not real results.\n")
print(f"wrote synthetic sample to {C.SAMPLE_DIR}: {len(feats)} accounts ({io.sum()} IO), {len(replies)} replies")
