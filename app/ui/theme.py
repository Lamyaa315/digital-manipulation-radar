"""Visual system: light palette, fonts, global CSS and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, LINE = "#F6F7F9", "#FFFFFF", "#E3E6EB"
TEXT, MUTED = "#1D2433", "#667085"
ACCENT, CORAL, AMBER, GREEN, STEEL = "#0E7C86", "#D64550", "#C98A1B", "#2E8B57", "#A3ACBA"
CYAN = LIME = ACCENT  # older code refers to these names

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');

:root {{ --bg:{BG}; --surface:{SURFACE}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
        --accent:{ACCENT}; --coral:{CORAL}; --amber:{AMBER}; --green:{GREEN}; }}

html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--bg) !important; color:var(--text); }}
.stApp {{ font-family:'Inter',system-ui,sans-serif; font-size:15px; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display:none !important; }}
.block-container {{ padding-top:1.6rem; max-width:1200px; }}
h1,h2,h3,h4 {{ font-family:'Inter',sans-serif !important; color:var(--text); font-weight:700 !important; letter-spacing:-.015em; }}
code, .mono {{ font-family:'JetBrains Mono',monospace !important; font-size:.9em; }}
a {{ color:var(--accent) !important; }}

/* top bar */
.brand {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:17px; }}
.brand small {{ font-weight:400; color:var(--muted); font-size:13px; margin-left:6px; }}

/* widgets */
.stButton > button, .stDownloadButton > button {{ background:var(--surface); color:var(--text); border:1px solid var(--line);
  border-radius:8px; font-size:14px; font-weight:500; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--accent); color:var(--accent); }}
.stButton > button[kind="primary"] {{ background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }}
.stButton > button[kind="primary"]:hover {{ background:#0B6A73; color:#fff; }}
[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea {{ background:var(--surface) !important; border-radius:8px !important; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
.stProgress > div > div > div > div {{ background:var(--accent); }}

/* text */
.page-title {{ font-size:30px !important; font-weight:700; margin:0 0 4px !important; padding:0 !important; }}
.lead {{ color:var(--muted); font-size:16px; max-width:760px; margin:0 0 10px; line-height:1.55; }}
.kicker {{ font-size:13px !important; color:var(--muted); margin:0 0 4px; font-weight:500; }}
.step, .step * {{ font-size:14px !important; color:var(--muted); }}
.idline, .idline * {{ font-size:13px !important; }}
.idline {{ color:var(--muted); }} .idline b {{ color:var(--text); font-weight:600; }}
.notice {{ font-size:13px; color:var(--muted); background:#EEF2F5; border-radius:8px; padding:10px 12px; line-height:1.5; }}
.section {{ display:flex; align-items:baseline; gap:10px; margin:26px 0 10px; }}
.section .num {{ color:var(--accent); font-weight:700; }}
.section .name {{ font-size:19px; font-weight:700; }}
.section .note {{ color:var(--muted); font-size:13px; margin-left:auto; }}

/* figures */
.figs {{ display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 10px; }}
.fig {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:130px; }}
.fig .v {{ font-size:24px; font-weight:700; line-height:1.15; }}
.fig .l {{ font-size:13px; color:var(--muted); margin-top:2px; }}
.fig .tag {{ display:none; }}
.fig.coral .v {{ color:var(--coral); }} .fig.cyan .v, .fig.accent .v {{ color:var(--accent); }}
.fig.green .v {{ color:var(--green); }} .fig.amber .v {{ color:var(--amber); }}

.tag {{ display:inline-block; font-size:11px; padding:0 6px; border:1px solid var(--line); border-radius:4px;
  color:var(--muted); vertical-align:middle; margin-left:6px; font-weight:500; }}
.tag.NA {{ color:var(--coral); }}

/* review */
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12.5px; font-weight:600; }}
.pill.suspicious {{ background:#FBE9EB; color:var(--coral); }}
.pill.normal {{ background:#E7F4EC; color:var(--green); }}
.pill.unsure {{ background:#FBF2E1; color:var(--amber); }}
.pill.open {{ background:#EEF0F3; color:var(--muted); }}
.scorebar {{ height:8px; background:#EEF0F3; border-radius:4px; position:relative; margin:6px 0 2px; }}
.scorebar i {{ position:absolute; left:0; top:0; bottom:0; border-radius:4px; }}
.reason {{ display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--line); font-size:14.5px; }}
.reason:last-child {{ border-bottom:none; }}
.reason .dot {{ width:8px; height:8px; border-radius:50%; background:var(--coral); margin-top:7px; flex:none; }}

/* evidence & misc kept for shared helpers */
.evidence {{ padding:2px 0 2px 14px; margin:6px 0 14px; border-left:3px solid var(--line); }}
.evidence.hot {{ border-left-color:var(--coral); }}
.evidence h4 {{ margin:0 0 2px !important; font-size:16px !important; padding:0 !important; }}
.evidence p {{ color:var(--muted); margin:0; font-size:14px; line-height:1.5; }}
.evidence .big {{ font-size:22px; font-weight:700; }}
.quote {{ border-left:3px solid var(--accent); padding:10px 16px; background:var(--surface); border-radius:0 8px 8px 0; font-size:15px; margin:6px 0 12px; }}
.risk .tag {{ display:none; }}
"""


def apply() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    tpl = go.layout.Template(pio.templates["plotly_white"])
    tpl.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[ACCENT, CORAL, AMBER, STEEL, GREEN],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=ACCENT, font=dict(family="Inter", color=TEXT)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    pio.templates["radar"] = tpl
    pio.templates.default = "radar"
