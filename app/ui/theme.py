"""Visual system: palette, fonts, global CSS and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, LINE = "#090B10", "#111620", "#1C2330"
TEXT, MUTED = "#E8EDF3", "#87919F"
CYAN, LIME, CORAL, STEEL = "#43D9E8", "#C8F169", "#FF626D", "#4A5668"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{ --bg:{BG}; --surface:{SURFACE}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
        --cyan:{CYAN}; --lime:{LIME}; --coral:{CORAL}; --steel:{STEEL}; }}

html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--bg) !important; color:var(--text); }}
.stApp {{ font-family:'Inter',system-ui,sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stToolbar"] {{ opacity:.35; }}
.block-container {{ padding-top:2.2rem; max-width:1400px; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; color:var(--text); }}
code, .mono {{ font-family:'JetBrains Mono',monospace !important; }}
p, li, label, .stMarkdown {{ color:var(--text); }}
a {{ color:var(--cyan) !important; }}

/* background radar grid, very faint */
.stApp::before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:
   radial-gradient(circle at 78% 18%, rgba(67,217,232,.07), transparent 38%),
   repeating-radial-gradient(circle at 78% 18%, rgba(135,145,159,.05) 0 1px, transparent 1px 90px);
}}

/* ---------- sidebar = narrow index ---------- */
[data-testid="stSidebar"] {{ background:#07090D; border-right:1px solid var(--line); min-width:250px; max-width:250px; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:10px 10px; border-left:2px solid transparent;
  border-radius:0; transition:all .18s ease; width:100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:rgba(67,217,232,.05); }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{ border-left-color:var(--cyan); background:rgba(67,217,232,.07); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-family:'JetBrains Mono',monospace; font-size:11.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted); white-space:nowrap; }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {{ color:var(--text); }}

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button {{ background:transparent; color:var(--text); border:1px solid var(--line);
  border-radius:2px; font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.1em; text-transform:uppercase;
  transition:all .18s ease; }}
.stButton > button:hover {{ border-color:var(--cyan); color:var(--cyan); background:rgba(67,217,232,.05); }}
.stButton > button[kind="primary"] {{ background:var(--cyan); color:#05070A; border-color:var(--cyan); font-weight:600; }}
.stButton > button[kind="primary"]:hover {{ background:#6FE6F1; color:#05070A; box-shadow:0 0 0 4px rgba(67,217,232,.15); }}
[data-baseweb="select"] > div, .stTextInput input {{ background:var(--surface) !important; border-color:var(--line) !important;
  border-radius:2px !important; font-family:'JetBrains Mono',monospace; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:2px; background:transparent; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
[data-testid="stTabs"] button p {{ font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.1em; text-transform:uppercase; }}
.stProgress > div > div > div > div {{ background:var(--cyan); }}
hr {{ border-color:var(--line) !important; }}

/* ---------- editorial pieces ---------- */
.kicker {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.22em; text-transform:uppercase; color:var(--muted); margin:0 0 6px; }}
.kicker b {{ color:var(--cyan); font-weight:500; }}
.hero-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:clamp(40px,5.6vw,78px); line-height:.95;
  letter-spacing:-.03em; margin:4px 0 14px; }}
.hero-title span {{ color:var(--cyan); }}
.hero-sub {{ font-size:19px; color:var(--text); margin:0; }}
.hero-desc {{ color:var(--muted); max-width:620px; margin:6px 0 0; }}
.section {{ display:flex; align-items:baseline; gap:14px; margin:38px 0 14px; border-top:1px solid var(--line); padding-top:14px; }}
.section .num {{ font-family:'JetBrains Mono',monospace; color:var(--cyan); font-size:12px; letter-spacing:.15em; }}
.section .name {{ font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:600; }}
.section .note {{ color:var(--muted); font-size:13px; margin-left:auto; }}

.figs {{ display:flex; flex-wrap:wrap; gap:0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:8px 0 6px; }}
.fig {{ padding:14px 22px 12px 0; margin-right:22px; border-right:1px solid var(--line); min-width:120px; }}
.fig:last-child {{ border-right:none; }}
.fig .v {{ font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:600; line-height:1; }}
.fig .l {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-top:6px; }}
.fig.coral .v {{ color:var(--coral); }} .fig.cyan .v {{ color:var(--cyan); }} .fig.lime .v {{ color:var(--lime); }}

.tag {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-size:9.5px; letter-spacing:.14em; padding:2px 6px;
  border:1px solid; border-radius:1px; vertical-align:middle; margin-left:6px; }}
.tag.MODEL {{ color:var(--cyan); border-color:rgba(67,217,232,.5); }}
.tag.FEATURE {{ color:var(--lime); border-color:rgba(200,241,105,.45); }}
.tag.RULE {{ color:#AFC0D6; border-color:rgba(175,192,214,.4); }}
.tag.DATASET {{ color:var(--text); border-color:rgba(232,237,243,.35); }}
.tag.NA {{ color:var(--coral); border-color:rgba(255,98,109,.45); }}
.tag.TRUTH {{ color:#F4B860; border-color:rgba(244,184,96,.45); }}

.case {{ display:grid; grid-template-columns:110px 1fr 200px; gap:22px; padding:20px 0; border-bottom:1px solid var(--line); }}
.case .cid {{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--muted); letter-spacing:.12em; }}
.case .cid b {{ display:block; font-size:22px; color:var(--text); letter-spacing:0; margin-top:4px; font-family:'Space Grotesk',sans-serif; }}
.case h3 {{ margin:0 0 6px !important; font-size:22px !important; padding:0 !important; }}
.case ul {{ list-style:none; padding:0; margin:0; color:var(--muted); font-size:14px; columns:2; }}
.case li {{ margin:2px 0; }} .case li b {{ color:var(--text); font-weight:500; }}
.risk {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.15em; }}
.risk .bar {{ height:4px; background:var(--line); margin:8px 0 6px; position:relative; }}
.risk .bar i {{ position:absolute; left:0; top:0; bottom:0; }}
.risk.HIGH {{ color:var(--coral); }} .risk.HIGH i {{ background:var(--coral); box-shadow:0 0 12px rgba(255,98,109,.6); }}
.risk.ELEVATED {{ color:#F4B860; }} .risk.ELEVATED i {{ background:#F4B860; }}
.risk.LOW {{ color:var(--muted); }} .risk.LOW i {{ background:var(--steel); }}

.evidence {{ border-left:2px solid var(--line); padding:4px 0 4px 16px; margin:8px 0 18px; }}
.evidence.hot {{ border-left-color:var(--coral); }} .evidence.cool {{ border-left-color:var(--cyan); }}
.evidence h4 {{ margin:0 0 4px !important; font-size:17px !important; padding:0 !important; }}
.evidence p {{ color:var(--muted); margin:0; font-size:14px; }}
.evidence .big {{ font-family:'Space Grotesk',sans-serif; font-size:28px; color:var(--text); }}

.quote {{ border-left:2px solid var(--cyan); padding:10px 16px; background:rgba(67,217,232,.04); font-size:16px; margin:6px 0 12px; }}
.idline, .idline * {{ font-size:12px !important; }}
.kicker {{ font-size:11px !important; }}
.step, .step * {{ font-size:11px !important; }}
.idline {{ font-family:'JetBrains Mono',monospace; color:var(--muted); font-size:12px; letter-spacing:.06em; }}
.idline b {{ color:var(--text); font-weight:500; }}
.notice {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); letter-spacing:.06em; border:1px dashed var(--line); padding:8px 12px; }}
.step {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.18em; color:var(--muted); }}
.step b {{ color:var(--lime); }}
@keyframes fadeUp {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:none; }} }}
.block-container > div {{ animation:fadeUp .35s ease both; }}
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
