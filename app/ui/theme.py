"""Visual system: palette, fonts, global CSS and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, LINE = "#090B10", "#111620", "#1C2330"
TEXT, MUTED = "#E8EDF3", "#87919F"
CYAN, LIME, CORAL, STEEL = "#43D9E8", "#C8F169", "#FF626D", "#4A5668"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

:root {{ --bg:{BG}; --surface:{SURFACE}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
        --cyan:{CYAN}; --lime:{LIME}; --coral:{CORAL}; --steel:{STEEL}; }}

/* calm base: no background effects, no entrance animations */
html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--bg) !important; color:var(--text); }}
.stApp {{ font-family:'Inter',system-ui,sans-serif; font-size:15px; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stToolbar"] {{ opacity:.35; }}
.block-container {{ padding-top:2.4rem; max-width:1240px; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; color:var(--text); font-weight:600 !important; }}
code, .mono {{ font-family:'JetBrains Mono',monospace !important; font-size:.9em; }}
a {{ color:var(--cyan) !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:#0C0F15; border-right:1px solid var(--line); min-width:240px; max-width:240px; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:9px 12px; border-radius:6px; width:100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:rgba(255,255,255,.03); }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{ background:rgba(67,217,232,.08); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-size:14.5px; color:var(--muted); white-space:nowrap; }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {{ color:var(--text); font-weight:500; }}

/* widgets */
.stButton > button, .stDownloadButton > button {{ background:transparent; color:var(--text); border:1px solid var(--line);
  border-radius:6px; font-size:14px; }}
.stButton > button:hover {{ border-color:var(--cyan); color:var(--cyan); }}
.stButton > button[kind="primary"] {{ background:var(--cyan); color:#05070A; border-color:var(--cyan); font-weight:600; }}
.stButton > button[kind="primary"]:hover {{ background:#6FE6F1; color:#05070A; }}
[data-baseweb="select"] > div, .stTextInput input {{ background:var(--surface) !important; border-color:var(--line) !important; border-radius:6px !important; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:8px; background:transparent; }}
.stProgress > div > div > div > div {{ background:var(--cyan); }}
hr {{ border-color:var(--line) !important; }}

/* text */
.page-title {{ font-family:'Space Grotesk',sans-serif; font-size:34px !important; font-weight:600; margin:0 0 6px !important; padding:0 !important; }}
.lead {{ color:var(--muted); font-size:16px; max-width:720px; margin:0 0 8px; line-height:1.55; }}
.kicker {{ font-size:13px !important; color:var(--muted); margin:0 0 6px; }}
.step, .step * {{ font-size:14px !important; color:var(--muted); }}
.idline, .idline * {{ font-size:13px !important; }}
.idline {{ color:var(--muted); }} .idline b {{ color:var(--text); font-weight:500; }}
.notice {{ font-size:12.5px; color:var(--muted); border:1px solid var(--line); border-radius:6px; padding:8px 10px; line-height:1.45; }}
.section {{ display:flex; align-items:baseline; gap:10px; margin:34px 0 12px; }}
.section .num {{ color:var(--cyan); font-size:15px; font-weight:600; }}
.section .name {{ font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:600; }}
.section .note {{ color:var(--muted); font-size:13px; margin-left:auto; }}

/* figures row */
.figs {{ display:flex; flex-wrap:wrap; gap:28px; margin:6px 0 10px; }}
.fig {{ min-width:110px; }}
.fig .v {{ font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:600; line-height:1.1; }}
.fig .l {{ font-size:13px; color:var(--muted); margin-top:2px; }}
.fig .tag {{ display:none; }}
.fig.coral .v {{ color:var(--coral); }} .fig.cyan .v {{ color:var(--cyan); }} .fig.lime .v {{ color:var(--text); }}

/* source tags: quiet, one style */
.tag {{ display:inline-block; font-size:10.5px; padding:0 6px; border:1px solid var(--line); border-radius:4px;
  color:var(--muted); vertical-align:middle; margin-left:6px; font-weight:400; }}
.tag.NA {{ color:var(--coral); border-color:rgba(255,98,109,.35); }}

/* cases */
.case {{ display:grid; grid-template-columns:90px 1fr 170px; gap:20px; padding:16px 0; border-bottom:1px solid var(--line); }}
.case .cid {{ font-size:12px; color:var(--muted); }}
.case .cid b {{ display:block; font-size:17px; color:var(--text); margin-top:2px; font-family:'Space Grotesk',sans-serif; }}
.case h3 {{ margin:0 0 6px !important; font-size:18px !important; padding:0 !important; }}
.case ul {{ list-style:none; padding:0; margin:0; color:var(--muted); font-size:14px; columns:2; }}
.case li {{ margin:2px 0; }} .case li b {{ color:var(--text); font-weight:500; }}
.risk {{ font-size:13px; }}
.risk .bar {{ height:4px; background:var(--line); margin:6px 0; position:relative; border-radius:2px; }}
.risk .bar i {{ position:absolute; left:0; top:0; bottom:0; border-radius:2px; }}
.risk.HIGH {{ color:var(--coral); }} .risk.HIGH i {{ background:var(--coral); }}
.risk.ELEVATED {{ color:#E0B060; }} .risk.ELEVATED i {{ background:#E0B060; }}
.risk.LOW {{ color:var(--muted); }} .risk.LOW i {{ background:var(--steel); }}
.risk .tag {{ display:none; }}

/* evidence */
.evidence {{ padding:2px 0 2px 14px; margin:6px 0 14px; border-left:2px solid var(--line); }}
.evidence.hot {{ border-left-color:var(--coral); }} .evidence.cool {{ border-left-color:var(--line); }}
.evidence h4 {{ margin:0 0 2px !important; font-size:16px !important; padding:0 !important; }}
.evidence p {{ color:var(--muted); margin:0; font-size:14px; line-height:1.5; }}
.evidence .big {{ font-family:'Space Grotesk',sans-serif; font-size:24px; color:var(--text); }}
.quote {{ border-left:2px solid var(--cyan); padding:10px 16px; background:var(--surface); border-radius:0 6px 6px 0; font-size:16px; margin:6px 0 12px; }}
"""


def apply() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    tpl = go.layout.Template(pio.templates["plotly_dark"])
    tpl.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[CYAN, LIME, CORAL, "#AFC0D6", "#F4B860"],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=CYAN, font=dict(family="JetBrains Mono", color=TEXT)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    pio.templates["radar"] = tpl
    pio.templates.default = "radar"
