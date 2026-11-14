"""Visual system: near-black, white type, a subtle neon-lime accent, thin technical lines."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, LINE = "#050506", "#0C0C0E", "#1E1F23"
TEXT, MUTED = "#F2F2F0", "#8B8D93"
LIME, AMBER, STEEL = "#C6F432", "#D9A441", "#4A4D55"
ACCENT = CYAN = CORAL = LIME      # one accent; older code refers to these names
GREEN = "#B9BBC1"                 # "looks normal" — neutral, not a colour signal

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{ --bg:{BG}; --surface:{SURFACE}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED}; --lime:{LIME}; --amber:{AMBER}; }}
html {{ scroll-behavior:smooth; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--bg) !important; color:var(--text); }}
.stApp {{ font-family:'Inter',system-ui,sans-serif; font-size:15px; }}
/* fine technical grid */
.stApp::before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size:64px 64px; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display:none !important; }}
.block-container {{ padding-top:1.4rem; max-width:1240px; position:relative; z-index:1; }}
h1,h2,h3,h4 {{ font-family:'Inter',sans-serif !important; color:var(--text); font-weight:500 !important; letter-spacing:-.02em; }}
code, .mono {{ font-family:'JetBrains Mono',monospace !important; font-size:.88em; }}
a {{ color:var(--lime) !important; }}
hr, [data-testid="stDivider"] {{ border-color:var(--line) !important; }}

/* top bar */
.brand {{ display:flex; align-items:center; gap:10px; font-weight:600; font-size:14px; letter-spacing:.06em; white-space:nowrap; }}
.brand small {{ font-family:'JetBrains Mono',monospace; font-weight:400; color:var(--muted); font-size:11px; margin-left:8px;
  letter-spacing:.08em; text-transform:uppercase; }}
[data-testid="stButtonGroup"] button {{ background:transparent !important; border:1px solid var(--line) !important;
  border-radius:0 !important; font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.06em; color:var(--muted) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {{
  color:var(--bg) !important; background:var(--lime) !important; border-color:var(--lime) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color:#0A0A0A !important; }}

/* widgets: thin, square, technical */
.stButton > button, .stDownloadButton > button {{ background:transparent; color:var(--text); border:1px solid var(--line);
  border-radius:0; font-size:13px; font-weight:500; transition:all .2s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--lime); color:var(--lime); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:var(--lime); color:#0A0A0A;
  border-color:var(--lime); font-weight:600; }}
.stButton > button[kind="primary"]:hover {{ background:#D8FF5A; color:#0A0A0A; box-shadow:0 0 0 4px rgba(198,244,50,.12); }}
[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea, [data-testid="stFileUploaderDropzone"] {{
  background:var(--surface) !important; border-radius:0 !important; border-color:var(--line) !important; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:0; background:transparent; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:0 !important; border-color:var(--line) !important; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
[data-testid="stTabs"] button p {{ font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.05em; }}
.stProgress > div > div > div > div {{ background:var(--lime); }}

/* type */
.label, .kicker {{ font-family:'JetBrains Mono',monospace; font-size:11px !important; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 6px; }}
.label b, .kicker b {{ color:var(--lime); font-weight:500; }}
.page-title {{ font-size:34px !important; font-weight:500; margin:0 0 4px !important; padding:0 !important; letter-spacing:-.025em; }}
.lead {{ color:var(--muted); font-size:16px; max-width:720px; margin:0 0 10px; line-height:1.6; }}
.step, .step * {{ font-size:13px !important; color:var(--muted); }}
.idline, .idline * {{ font-size:12.5px !important; }}
.idline {{ color:var(--muted); font-family:'JetBrains Mono',monospace; }} .idline b {{ color:var(--text); font-weight:500; }}
.notice {{ font-size:13px; color:var(--muted); border:1px solid var(--line); padding:10px 12px; line-height:1.5; }}
.section {{ display:flex; align-items:baseline; gap:12px; margin:30px 0 12px; border-top:1px solid var(--line); padding-top:12px; }}
.section .num {{ font-family:'JetBrains Mono',monospace; color:var(--lime); font-size:12px; letter-spacing:.1em; }}
.section .name {{ font-size:20px; font-weight:500; }}
.section .note {{ color:var(--muted); font-size:12px; margin-left:auto; font-family:'JetBrains Mono',monospace; }}

/* figures: lines, not boxes */
.figs {{ display:flex; flex-wrap:wrap; gap:0; margin:6px 0 12px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.fig {{ padding:14px 26px 12px 0; margin-right:26px; border-right:1px solid var(--line); min-width:110px; }}
.fig:last-child {{ border-right:none; }}
.fig .v {{ font-size:28px; font-weight:400; line-height:1.1; letter-spacing:-.02em; }}
.fig .l {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-top:4px; }}
.fig .tag {{ display:none; }}
.fig.coral .v, .fig.cyan .v, .fig.accent .v {{ color:var(--lime); }} .fig.amber .v {{ color:var(--amber); }}
.tag {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-size:10px; padding:0 5px; border:1px solid var(--line);
  color:var(--muted); vertical-align:middle; margin-left:6px; letter-spacing:.08em; }}
.tag.NA {{ color:var(--amber); }}

/* review */
.pill {{ display:inline-block; padding:2px 9px; font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; border:1px solid var(--line); }}
.pill.suspicious {{ color:var(--lime); border-color:rgba(198,244,50,.5); }}
.pill.normal {{ color:var(--text); }}
.pill.unsure {{ color:var(--amber); border-color:rgba(217,164,65,.5); }}
.pill.open {{ color:var(--muted); }}
.scorebar {{ height:3px; background:var(--line); position:relative; margin:8px 0 2px; }}
.scorebar i {{ position:absolute; left:0; top:0; bottom:0; }}
.reason {{ display:flex; gap:10px; align-items:flex-start; padding:9px 0; border-bottom:1px solid var(--line); font-size:14.5px; }}
.reason:last-child {{ border-bottom:none; }}
.reason .dot {{ width:6px; height:6px; background:var(--lime); margin-top:8px; flex:none; box-shadow:0 0 8px rgba(198,244,50,.6); }}
.evidence {{ padding:2px 0 2px 14px; margin:6px 0 14px; border-left:1px solid var(--line); }}
.evidence.hot {{ border-left-color:var(--lime); }}
.evidence h4 {{ margin:0 0 2px !important; font-size:16px !important; padding:0 !important; }}
.evidence p {{ color:var(--muted); margin:0; font-size:14px; line-height:1.5; }}
.evidence .big {{ font-size:22px; }}
.quote {{ border-left:1px solid var(--lime); padding:10px 16px; background:var(--surface); font-size:15px; margin:6px 0 12px; }}
.risk .tag {{ display:none; }}

/* landing */
.hero {{ padding:46px 0 10px; }}
.hero .kicker {{ color:var(--lime); }}
.hero h1 {{ font-size:clamp(44px,6.4vw,92px) !important; line-height:.98; font-weight:400 !important; letter-spacing:-.045em;
  margin:12px 0 18px !important; padding:0 !important; }}
.hero h1 em {{ font-style:normal; color:var(--muted); }}
.hero p {{ color:var(--muted); font-size:17px; max-width:560px; line-height:1.6; }}
.ticker {{ overflow:hidden; border-top:1px solid var(--line); border-bottom:1px solid var(--line); white-space:nowrap; margin:16px 0 8px; }}
.ticker div {{ display:inline-block; padding:9px 0; animation:tick 60s linear infinite; font-family:'JetBrains Mono',monospace;
  font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.ticker span {{ margin:0 28px; }} .ticker b {{ color:var(--lime); font-weight:500; }}
@keyframes tick {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
.reveal {{ animation:reveal linear both; animation-timeline:view(); animation-range:entry 0% cover 28%; }}
@keyframes reveal {{ from {{ opacity:0; transform:translateY(28px); }} to {{ opacity:1; transform:none; }} }}
.numsec {{ display:grid; grid-template-columns:120px 1fr; gap:24px; border-top:1px solid var(--line); padding:34px 0 30px; }}
.numsec .n {{ font-family:'JetBrains Mono',monospace; color:var(--lime); font-size:12px; letter-spacing:.14em; }}
.numsec h2 {{ font-size:34px !important; font-weight:400 !important; margin:0 0 10px !important; padding:0 !important; letter-spacing:-.03em; }}
.numsec p {{ color:var(--muted); font-size:16px; line-height:1.65; max-width:760px; margin:0 0 10px; }}
.cols3 {{ display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--line); margin-top:16px; }}
.cols3 > div {{ padding:16px 18px 6px 0; border-right:1px solid var(--line); margin-right:18px; }}
.cols3 > div:last-child {{ border-right:none; }}
.cols3 .v {{ font-size:40px; font-weight:300; letter-spacing:-.03em; }} .cols3 .v.lime {{ color:var(--lime); }}
.cols3 .l {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.pipe svg {{ width:100%; height:auto; }}
.pipe .node rect {{ fill:#0B0B0D; stroke:#2A2B30; transition:all .25s ease; }}
.pipe .node:hover rect {{ stroke:{LIME}; fill:rgba(198,244,50,.06); }}
.pipe .node text {{ fill:{TEXT}; font-family:Inter,sans-serif; font-size:14px; }}
.pipe .node text.sub {{ fill:{MUTED}; font-family:'JetBrains Mono',monospace; font-size:9.5px; }}
.pipe .node .idx {{ fill:{LIME}; font-family:'JetBrains Mono',monospace; font-size:10px; }}
.pipe .flow {{ stroke:{LIME}; stroke-width:1.2; fill:none; stroke-dasharray:3 7; animation:flow 1.6s linear infinite; opacity:.8; }}
@keyframes flow {{ to {{ stroke-dashoffset:-20; }} }}
"""


def apply() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    tpl = go.layout.Template(pio.templates["plotly_dark"])
    tpl.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[LIME, "#B9BBC1", AMBER, STEEL],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LIME, font=dict(family="JetBrains Mono", color=TEXT)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    pio.templates["radar"] = tpl
    pio.templates.default = "radar"
