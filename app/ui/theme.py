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

/* ---- v3: rounded panels, icon nav, KPI cards (from the team's mockups) ---- */
.brand {{ font-size:14px; font-weight:700; letter-spacing:.02em; gap:10px; }}
[data-testid="stButtonGroup"] > div {{ justify-content:flex-end; flex-wrap:nowrap; }}
.brand svg {{ flex:none; width:34px; }}
.brand .sep {{ flex:none; width:1px; height:24px; background:#3A3B40; margin:0 6px 0 10px; display:inline-block; }}
.brand small {{ margin-left:0; font-size:12px; letter-spacing:.12em; color:#B9BBC1; }}
[data-testid="stButtonGroup"] button {{ border:none !important; border-radius:10px !important; font-family:'Inter',sans-serif !important;
  font-size:14px !important; letter-spacing:0 !important; color:var(--text) !important; padding:6px 12px !important; }}
[data-testid="stButtonGroup"] button p {{ font-family:'Inter',sans-serif !important; font-size:14px !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {{
  box-shadow:0 0 18px rgba(198,244,50,.25); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] span, [data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color:#0A0A0A !important; }}
.stButton > button, .stDownloadButton > button {{ border-radius:10px; min-height:44px; }}
.stButton > button[kind="primary"] {{ font-size:15px; box-shadow:0 0 22px rgba(198,244,50,.18); }}
[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea, [data-testid="stFileUploaderDropzone"] {{ border-radius:10px !important; }}
[data-testid="stFileUploaderDropzone"] {{ border:1.5px dashed rgba(198,244,50,.45) !important; min-height:120px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px !important; background:rgba(12,12,14,.78); }}
[data-testid="stExpander"] {{ border-radius:12px; }}
.ptitle {{ border-left:4px solid var(--lime); padding:2px 0 2px 18px; margin:6px 0 18px; }}
.ptitle h1 {{ font-size:34px !important; font-weight:700 !important; margin:0 !important; padding:0 !important; letter-spacing:-.02em; }}
.ptitle p {{ color:#B9BBC1; margin:4px 0 0; font-size:15px; }}
.panel-h {{ display:flex; align-items:center; gap:10px; font-size:19px; font-weight:600; margin:2px 0 10px; }}
.panel-h svg {{ width:24px; height:24px; stroke:var(--lime); fill:none; stroke-width:1.8; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:4px 0 18px; }}
.kpi {{ border:1px solid #2A2B30; border-radius:16px; padding:18px 20px; display:flex; gap:16px; align-items:center; background:var(--surface); }}
.kpi.hot {{ border:1.5px solid var(--lime); box-shadow:0 0 26px rgba(198,244,50,.16) inset, 0 0 20px rgba(198,244,50,.12); }}
.kpi svg {{ width:42px; height:42px; stroke:var(--lime); fill:none; stroke-width:1.7; flex:none; }}
.kpi .t {{ font-size:14px; font-weight:500; }} .kpi .v {{ font-size:32px; font-weight:700; line-height:1.15; letter-spacing:-.02em; }}
.kpi .s {{ color:#B9BBC1; font-size:13px; }}
.chip {{ border:1px solid #2A2B30; border-radius:14px; padding:12px 18px; display:flex; gap:14px; align-items:center;
  background:var(--surface); font-size:13.5px; color:#B9BBC1; justify-content:flex-end; }}
.chip b {{ color:var(--text); font-weight:600; }}
.legend-row {{ display:grid; grid-template-columns:16px 1fr auto; gap:12px; align-items:center; padding:9px 0; font-size:15px; }}
.legend-row i {{ width:12px; height:12px; border-radius:50%; display:block; }}
.legend-row b {{ font-weight:600; }}
.barrow {{ display:grid; grid-template-columns:120px 1fr 52px; gap:14px; align-items:center; padding:7px 0; font-size:14px;
  border-bottom:1px solid #16171A; }}
.barrow .track {{ height:10px; background:#1E1F23; border-radius:6px; overflow:hidden; }}
.barrow .fill {{ height:10px; background:var(--lime); border-radius:6px; }}
.barrow .fill.low {{ background:#6B6E76; }}
.barrow .num {{ text-align:right; font-weight:600; font-family:'JetBrains Mono',monospace; font-size:13px; }}
.orline {{ display:flex; align-items:center; gap:12px; color:var(--muted); font-size:12px; margin:10px 0; }}
.orline::before, .orline::after {{ content:""; flex:1; height:1px; background:#2A2B30; }}
.orline span {{ border:1px solid #2A2B30; border-radius:10px; padding:2px 12px; }}
.fieldlbl {{ font-size:15px; font-weight:600; margin:10px 0 2px; }} .fieldlbl span {{ color:var(--muted); font-weight:400; }}
.hero .kick {{ display:flex; align-items:center; gap:16px; color:var(--lime); font-family:'JetBrains Mono',monospace; font-size:12.5px;
  letter-spacing:.16em; font-weight:500; }}
.hero .kick::after {{ content:""; width:44px; height:2px; background:var(--lime); }}
.hero h1 {{ font-weight:700 !important; letter-spacing:-.035em; font-size:clamp(40px,5vw,74px) !important; }}
.hero h1 em {{ color:var(--lime) !important; }}
.hero p {{ color:#C9CBD0; }}
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
