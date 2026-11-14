"""Digital Manipulation Radar — a review tool for coordinated reply accounts.

    streamlit run app/main.py

Flow: 1 Upload (a demo campaign or your own files) → 2 Review (ranked accounts, reasons,
decisions) → 3 Report (summary + downloads). About explains the model and its evaluation.
    app/core/store.py    data (full artifacts, or the lite copy when hosted)
    app/core/review.py   queue, reasons, decisions (no UI)
    app/views/           one module per step
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="Digital Manipulation Radar", page_icon="◎", layout="wide",
                   initial_sidebar_state="collapsed")

from app.core.store import Store  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.blocks import html  # noqa: E402
from app.views import datalab, home, report, review, upload  # noqa: E402
from app.views.common import PAGES  # noqa: E402

theme.apply()

SRC = Store.source()
if SRC is None:
    st.error("No artifacts yet. Run `python -m radar.pipeline` first "
             "(or `python scripts/make_lite.py` to build the small hosted copy).")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_store(stamp: float) -> Store:
    return Store.load()


with st.spinner("Loading the data…"):
    store = get_store((SRC / "run_info.json").stat().st_mtime)

if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
if not st.session_state.get("nav"):          # clicking the active tab again clears it
    st.session_state["nav"] = st.session_state.get("_page", PAGES[0])

brand, nav = st.columns([5, 4], vertical_alignment="center")
with brand:
    html('<div class="brand"><svg width="22" height="22" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" '
         'fill="none" stroke="#F2F2F0" stroke-opacity=".5" stroke-width="1"/><circle cx="12" cy="12" r="6" fill="none" '
         'stroke="#F2F2F0" stroke-opacity=".3" stroke-width="1"/><path d="M12 12 L12 1" stroke="#C6F432" stroke-width="1.2"/>'
         '<circle cx="16.5" cy="7.5" r="2" fill="#C6F432"/></svg>DIGITAL MANIPULATION RADAR<small>intelligence center</small></div>')
with nav:
    page = st.segmented_control("Navigate", PAGES, key="nav", label_visibility="collapsed")
page = page or st.session_state.get("_page", PAGES[0])
st.session_state["_page"] = page
st.divider()

{PAGES[0]: home, PAGES[1]: upload, PAGES[2]: review, PAGES[3]: report, PAGES[4]: datalab}[page].render(store)

foot = "Scores rank accounts for human review. They are not proof of fake accounts or intent."
if store.lite:
    foot += " Hosted lite version: same scores and groups as the full run."
st.caption(foot)
