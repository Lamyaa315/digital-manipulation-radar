"""Day-1 check: print the real schema of every downloaded CSV so config.py can be fixed.

    python scripts/inspect_data.py > docs/DATA_INSPECTION.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar import config as C  # noqa: E402

d = C.SAMPLE_DIR if "--sample" in sys.argv else C.RAW_DIR
for path in sorted(d.glob("*.csv")):
    head = pd.read_csv(path, nrows=5000)
    n_rows = sum(1 for _ in open(path, "rb")) - 1
    print(f"\n=== {path.name}  ({n_rows:,} rows, {len(head.columns)} columns)")
    for col in head.columns:
        s = head[col]
        print(f"  {col:45s} {str(s.dtype):8s} nulls(first 5k)={s.isna().sum():5d}  e.g. {s.dropna().iloc[0] if s.notna().any() else '-'}")
    for col in (C.COL_LABEL, C.COL_TYPE, C.COL_CAMPAIGN, "tweet_label"):
        if col in head.columns:
            full = pd.read_csv(path, usecols=[col])[col]
            print(f"  value counts of {col}:\n" + full.value_counts().head(50).to_string())

expected = {C.FEATURES_FILE: [C.COL_USER, C.COL_LABEL],
            C.REPLIES_FILE: [C.COL_TWEET, C.COL_USER, C.COL_CAMPAIGN, C.COL_TYPE, C.COL_LABEL]}
print("\n=== config check")
for name, cols in expected.items():
    p = d / name
    if not p.exists():
        print(f"  MISSING FILE {name}")
        continue
    have = set(pd.read_csv(p, nrows=1).columns)
    for c in cols:
        print(f"  {'ok     ' if c in have else 'MISSING'} {name}: {c}")
