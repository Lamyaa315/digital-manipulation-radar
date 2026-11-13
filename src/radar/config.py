"""All paths, file names, column names and tunable constants live here.

Column names come from the dataset's official datasheet. Confirm them on Day 1 with
`python scripts/inspect_data.py` and fix them here if they differ. Nothing else in the code
should hard-code a column name.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"  # synthetic, for testing the code only
ARTIFACTS_DIR = ROOT / "artifacts"

# --- Source: Zenodo 10.5281/zenodo.13896308, CC BY 4.0 -------------------------------
ZENODO_RECORD = "13896309"  # version record; 13896308 is the concept DOI (download URLs 404)
ZENODO_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"
# name -> approximate size in MB (from the Zenodo page)
FILES = {
    "coordinated_reply_datasheet.pdf": 0.15,
    "RQ3_replier_classifier_features.csv": 328.4,
    "RQ3_replier_info.csv": 286.1,
    "poster_tweetid_campaign_type.csv": 189.5,
    "RQ1_time_difference_of_reply.csv": 47.7,
    "RQ1_target_non_target_tweets.csv": 13.2,
    "RQ2_tweet_classifier_features.csv": 7.7,
    "RQ1_num_targeted_tweet_by_IO.csv": 4.7,
    "RQ1_number_of_reply_per_tweet.csv": 1.2,
    "RQ2_engagement.csv": 0.3,
    "RQ1_target_follower_following_count.csv": 0.13,
    "RQ1_target_annotation.csv": 0.06,
    # 1.26 billion reply pairs. Not needed: its information is already summarized
    # in the cosine features of RQ3_replier_classifier_features.csv.
    "pos_cosine_with_replier_info.pkl.gz": 3800.0,
}
SKIP_BY_DEFAULT = {"pos_cosine_with_replier_info.pkl.gz"}

FEATURES_FILE = "RQ3_replier_classifier_features.csv"
REPLIER_INFO_FILE = "RQ3_replier_info.csv"
REPLIES_FILE = "poster_tweetid_campaign_type.csv"
DELAY_FILE = "RQ1_time_difference_of_reply.csv"
TEXT_FILE = "RQ1_target_non_target_tweets.csv"  # English translations (from Serbian/Arabic)

# --- Columns (datasheet names; VERIFY on Day 1) ------------------------------------------
COL_USER = "replier_userid"
COL_LABEL = "replier_label"      # 1 = IO replier, 0 = normal replier
COL_TWEET = "poster_tweetid"     # the targeted / control tweet
COL_REPLY = "replier_tweetid"
COL_CAMPAIGN = "campaign"
COL_TYPE = "type"                # "target" or "control"
TARGET_TYPE = "target"
COL_DELAY = "diff_min"

# Columns that are identifiers or labels, never model inputs.
NON_FEATURE_COLS = {COL_USER, COL_LABEL, COL_TWEET, COL_CAMPAIGN, "tweet_label", "campaign_group"}

# Profile features are recognized by name tokens (split on "_"). VERIFY on Day 1.
PROFILE_TOKENS = {"age", "follower", "followers", "following", "activity"}

# Feature-group mapping for explanations: first matching token wins.
FEATURE_GROUP_TOKENS = [
    ("network", {"graph"}),
    ("text similarity", {"cosine"}),
    ("reply delay", {"delay", "diff"}),
    ("entities", {"hashtag", "hashtags", "mention", "url"}),
    ("engagement", {"like", "retweet", "reply", "quote"}),
    ("profile", PROFILE_TOKENS),
]

# --- Experiment settings --------------------------------------------------------------
RANDOM_STATE = 42
TOP_CAMPAIGNS = 5          # the rest are pooled into "other" (as in the paper)
NEG_PER_POS_TRAIN = 10     # negatives kept per positive in TRAINING folds only
PRECISION_AT_K = (100, 500)
IMPORTANCE_MAX_ROWS = 50_000

# --- Co-reply graph -------------------------------------------------------------------
GRAPH_MIN_TWEETS = 2              # a replier must reply to >= this many target tweets
GRAPH_MIN_SHARED = 3              # an edge needs >= this many shared target tweets
                                  # (real data: 2 -> 4.1M edges, 3 -> 831k, 5 -> 144k)
GRAPH_MAX_REPLIERS_PER_TWEET = 2000  # skip viral tweets that would create huge cliques
