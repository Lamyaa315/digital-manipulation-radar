"""Download the OSoMe coordinated-replies dataset (Zenodo 13896308, CC BY 4.0) to data/raw/.

Skips the 3.8 GB pairwise-cosine pickle unless --all is given. Existing files are kept.
    python scripts/download_data.py
    python scripts/download_data.py --only RQ2_engagement.csv RQ3_replier_info.csv
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar import config as C  # noqa: E402


def fetch(name: str) -> None:
    dest = C.RAW_DIR / name
    if dest.exists():
        print(f"exists  {name}")
        return
    url = C.ZENODO_URL.format(record=C.ZENODO_RECORD, name=name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get     {name} (~{C.FILES[name]:.1f} MB)")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r        {done / total:6.1%}", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also download the 3.8 GB pickle")
    ap.add_argument("--only", nargs="+", choices=sorted(C.FILES))
    a = ap.parse_args()
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = a.only or [n for n in C.FILES if a.all or n not in C.SKIP_BY_DEFAULT]
    print(f"{len(names)} files, ~{sum(C.FILES[n] for n in names):.0f} MB -> {C.RAW_DIR}")
    for n in names:
        fetch(n)


if __name__ == "__main__":
    main()
