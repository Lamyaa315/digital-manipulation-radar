"""Digital Manipulation Radar — the investigation app.

    streamlit run app/main.py

Reads the artifacts written by `python -m radar.pipeline`. Layout:
    app/core/store.py   data + analysis (no UI)
    app/ui/             theme, HTML blocks, canvas components (network, radar)
    app/views/          one module per page
The older analyst dashboard is still at app/streamlit_app.py.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="Digital Manipulation Radar", page_icon="◉", layout="wide",
                   initial_sidebar_state="expanded")

from app.core.store import A, Store  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.blocks import html  # noqa: E402
from app.views import datalab, explorer, investigations, room, timeline  # noqa: E402
from app.views.common import PAGES, campaign, pretty  # noqa: E402

theme.apply()

if not (A / "run_info.json").exists():
    st.error("No artifacts yet. Run `python -m radar.pipeline` first.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_store(stamp: float) -> Store:
    return Store.load()


with st.spinner("Loading 794k accounts and 2.6M replies…"):
    store = get_store((A / "run_info.json").stat().st_mtime)

if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")

with st.sidebar:
    html('<div style="display:flex;align-items:center;gap:10px;margin:4px 0 26px">'
         '<svg width="26" height="26" viewBox="0 0 26 26"><circle cx="13" cy="13" r="12" fill="none" stroke="#43D9E8" stroke-opacity=".5"/>'
         '<circle cx="13" cy="13" r="7" fill="none" stroke="#43D9E8" stroke-opacity=".3"/>'
         '<path d="M13 13 L13 1" stroke="#43D9E8" stroke-width="1.5"/><circle cx="18" cy="8" r="2" fill="#FF626D"/></svg>'
         '<div><div style="font-family:Space Grotesk;font-weight:700;letter-spacing:.02em;line-height:1">DM RADAR</div>'
         '<div class="kicker" style="margin:2px 0 0;font-size:9px">coordination intelligence</div></div></div>')
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    ran = campaign() in store._cases
    html(f'<div style="margin-top:30px" class="kicker">ACTIVE DATASET</div>'
         f'<div style="font-size:14px">{pretty(campaign())}</div>'
         f'<div class="idline" style="margin-top:4px">{"● analysed" if ran else "○ not analysed yet"}'
         f'{" · case " + st.session_state["case_id"] if st.session_state.get("case_id") else ""}</div>')
    if store.info["is_sample"]:
        st.error("SYNTHETIC DATA")
    html('<div class="notice" style="margin-top:30px">Scores rank accounts for human review. '
         'They are not proof of fake accounts or intent.</div>')

{PAGES[0]: room, PAGES[1]: investigations, PAGES[2]: explorer,
 PAGES[3]: timeline, PAGES[4]: datalab}[page].render(store)
