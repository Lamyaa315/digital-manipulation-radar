"""Build artifacts_lite/: a small copy of the pipeline's artifacts for free hosting.

    python scripts/make_lite.py

Needs the full artifacts (python -m radar.pipeline) and data/raw (for the tweet texts).
Keeps every account's score, every co-reply link and every reply time; drops the 76
feature columns except for accounts in the co-reply network or above the threshold.
Community detection is run here with the same code and seed, so the hosted app shows
the same groups as a local run.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.store import LITE, Store  # noqa: E402

t0 = time.time()
store = Store._load_full()
print(f"loaded full artifacts in {time.time() - t0:.0f}s")
store.save_lite(LITE)
total = 0
for f in sorted(LITE.iterdir()):
    mb = f.stat().st_size / 1e6
    total += mb
    print(f"  {f.name:24s} {mb:8.1f} MB")
print(f"total {total:.1f} MB in {LITE} ({time.time() - t0:.0f}s)")
