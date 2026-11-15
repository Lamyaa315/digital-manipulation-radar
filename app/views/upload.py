"""Step 1 — start an investigation. Left: what to analyse (a tweet's replies, or a campaign / dataset).
Right: analysis settings and the Run button."""
import pandas as pd
import streamlit as st

from app.core import review as R
from app.core.store import ROOT, Store
from app.ui.blocks import compact, html
from app.views.common import goto, pretty

MODELS = ROOT / "models"
MODES = [":material/bolt: Check a tweet's replies (quick)", ":material/stacks: Analyze a campaign / dataset"]
MANUAL_COLS = ["reply", "account", "reply_time", "followers", "following", "account_created_at", "reply_text", "tweet"]
GEAR = ('<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8'
        'l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1'
        'a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0'
        '-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5'
        ' 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1'
        'a1.7 1.7 0 0 0-1.5 1z"/></svg>')


@st.cache_resource(show_spinner=False)
def scorer():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from radar.deploy import Scorer
    return Scorer() if Scorer.available() else None


@st.cache_resource(show_spinner=False)
def simple_scorer():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from radar.simple import SimpleScorer
    return SimpleScorer() if SimpleScorer.available() else None


def _start(inv: R.Investigation) -> None:
    st.session_state["inv"] = inv
    st.session_state["sel"] = inv.queue.index[0] if len(inv.queue) else None
    st.session_state.pop("_last_click", None)
    goto(2)


def _blank(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({c: [""] * n for c in MANUAL_COLS})


def _labels():
    return pd.read_csv(MODELS / "simple_sample_labels.csv", dtype={"account_id": str}).set_index("account_id")["label"]


# ---- runs ----------------------------------------------------------------------------------
def run_simple(df: pd.DataFrame, name: str, n: int, focus_text: str, labels=None) -> None:
    """A plain replies table (typed in, or a file) → simple-input model."""
    from app.core import simple_input as SI
    sc = simple_scorer()
    if sc is None:
        st.error("No simple model. Run `python -m radar.simple` first.")
        return
    r, errors, notes = SI.parse(df)
    for e in errors:
        st.error(e)
    if errors:
        return
    for m in notes:
        st.info(m)
    accs = {a.lstrip("@").lower(): a for a in r["account"]}
    focus = [accs.get(x.lstrip("@").lower(), x) for x in SI.parse_ids(focus_text)]
    missing = [x for x in focus if x not in set(r["account"]) and not (len(x) == 6 and any(a.endswith(x) for a in r["account"]))]
    if missing:
        st.warning(f"Not in the data: {', '.join(missing[:5])}")
    with st.spinner("Computing features and scoring accounts…"):
        _start(SI.build(r, sc, name, n, labels, focus=focus))


def run_full(acc: pd.DataFrame, rep, name: str, n: int, store: Store) -> None:
    """Accounts in the dataset's 76-feature format → full model."""
    sc = scorer()
    if sc is None:
        st.error("No saved model. Run `python -m radar.deploy` first.")
        return
    if acc.dropna(how="all").empty:
        st.error("The accounts file has column names but no rows.")
        return
    missing, bad = sc.check(acc)
    if missing:
        st.error(f"{len(missing)} required columns are missing (e.g. {', '.join(missing[:4])}). Use the example file.")
        return
    if bad:
        st.error(f"These columns contain text that is not a number: {', '.join(bad[:4])}"
                 + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))
        return
    if rep is not None and rep.shape[1] < 2:
        st.error("The replies file needs two columns: account id, tweet id.")
        return
    with st.spinner("Scoring accounts with the model…"):
        _start(R.from_upload(sc, store._normal_stats(), acc, rep, name, n))


def _read(f) -> pd.DataFrame | None:
    try:
        return pd.read_csv(f)
    except Exception as e:  # noqa: BLE001 - show any parse error to the user
        st.error(f"Could not read the file: {e}")
        return None


# ---- page ----------------------------------------------------------------------------------
def render(store: Store) -> None:
    ss = st.session_state
    html('<div class="ptitle"><h1>Start a New Investigation</h1>'
         '<p>Check the replies to a tweet, or analyze a whole campaign — the model scores every account and finds '
         'the ones that act together.</p></div>')
    inv = ss.get("inv")
    if inv is not None:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f'<div class="notice">Current investigation: <b>{inv.name}</b>. Running a new one replaces it.</div>',
                    unsafe_allow_html=True)
        if c2.button("Open results", width="stretch"):
            goto(2)
    if "manual_df" not in ss:
        ss["manual_df"], ss["manual_v"], ss["manual_example"] = _blank(), 0, False

    left, right = st.columns([5, 3], gap="medium")
    with left:
        with st.container(border=True):
            mode = st.segmented_control("Mode", MODES, key="up_mode", default=MODES[0], label_visibility="collapsed",
                                        width="stretch") or MODES[0]
            quick = mode == MODES[0]
            if quick:
                html('<p class="fieldlbl">Tweet link or ID</p>')
                tweet = st.text_input("Tweet link", key="up_tweet", label_visibility="collapsed",
                                      placeholder="https://x.com/username/status/123456…")
                html('<p class="fieldlbl">Replies <span>— one row each; paste the reply link, or the account and time</span></p>')
                edited = st.data_editor(
                    ss["manual_df"], num_rows="dynamic", width="stretch", height=250, key=f"manual_{ss['manual_v']}",
                    column_config={
                        "reply": st.column_config.TextColumn("reply (link)", help="On X: Share → Copy link. Gives account and time."),
                        "account": st.column_config.TextColumn("account", help="@handle or ID; filled from the reply link if empty"),
                        "reply_time": st.column_config.TextColumn("reply time", help="Only without a reply link, e.g. 2025-11-14 10:30"),
                        "followers": st.column_config.TextColumn("followers"),
                        "following": st.column_config.TextColumn("following"),
                        "account_created_at": st.column_config.TextColumn("account created", help="e.g. 2025-10-01"),
                        "reply_text": st.column_config.TextColumn("reply text", help="Used to spot near-copies"),
                        "tweet": st.column_config.TextColumn("tweet (if not the one above)"),
                    })
                html('<div class="orline"><span>OR</span></div>')
                f_simple = st.file_uploader("Drag & drop a replies file here", type="csv", key="up_simple",
                                            help="One row per reply. Needed: account, tweet, reply time. "
                                                 "Common X export column names work.")
                st.caption("Uses the simple-input model (timing, profile, who replies with whom). It is weaker than "
                           "the full model — see About.")
            else:
                camps = store.campaigns()
                html('<p class="fieldlbl">Campaign from the dataset</p>')
                camp = st.selectbox("Campaign", list(camps.index), key="up_camp", label_visibility="collapsed",
                                    format_func=lambda x: f"{pretty(x)} — {compact(camps.at[x, 'accounts'])} accounts")
                st.caption("Real accounts from a known influence operation. Scores come from a model that never saw "
                           "this campaign.")
                html('<div class="orline"><span>OR</span></div>')
                html('<p class="fieldlbl">Your own data in the dataset format <span>— 76 features per account</span></p>')
                f_acc = st.file_uploader("Accounts file (CSV)", type="csv", key="up_acc")
                f_rep = st.file_uploader("Replies file (CSV, optional: account id, tweet id)", type="csv", key="up_rep")

    with right:
        with st.container(border=True):
            html(f'<div class="panel-h">{GEAR}Analysis Settings</div>')
            n = st.select_slider("Accounts to list", [50, 100, 250, 500, 1000], value=100, key="up_n",
                                 help="How many of the highest-scored accounts go into the results.")
            focus_text = st.text_input("Accounts to check first (optional)", key="up_focus",
                                       placeholder="@handles or IDs, comma-separated",
                                       help="Shown first in the results, with a plain verdict.")
            with st.expander("Examples (optional)", icon=":material/science:"):
                ex_fill = st.button("Fill the table with an example (85 replies)", width="stretch", key="ex_fill",
                                    disabled=not quick)
                ex_file = st.button("Sample replies file (2,256 replies)", width="stretch", key="ex_file")
                ex_two = st.button("Sample: two accounts on their own", width="stretch", key="ex_two")
                ex_full = st.button("Sample dataset-format files (300 accounts)", width="stretch", key="ex_full")
                st.caption("Real accounts from the dataset. None of them were used to train the models.")
                c1, c2 = st.columns(2)
                c1.download_button("Replies template", (MODELS / "simple_sample.csv").read_bytes(), "radar_replies_example.csv",
                                   "text/csv", width="stretch", key="dl_simple_ex")
                c2.download_button("76-feature template", (MODELS / "sample_accounts.csv").read_bytes(),
                                   "radar_accounts_example.csv", "text/csv", width="stretch", key="dl_full_ex")
            if quick and st.button("Clear the table", width="stretch", key="man_clear", icon=":material/delete:"):
                ss["manual_df"], ss["manual_example"] = _blank(), False
                ss["manual_v"] += 1
                st.rerun()
            run = st.button("Run Investigation", type="primary", width="stretch", key="run", icon=":material/play_arrow:")

    # ---- actions ---------------------------------------------------------------------------
    if ex_fill:
        ss["manual_df"] = pd.read_csv(MODELS / "example_manual.csv", dtype=str).reindex(columns=MANUAL_COLS).fillna("")
        ss["manual_v"] += 1
        ss["manual_example"] = True
        st.rerun()
    if ex_file:
        run_simple(pd.read_csv(MODELS / "simple_sample.csv"), "Sample: Honduras replies", n, "", _labels())
    if ex_two:
        run_simple(pd.read_csv(MODELS / "example_two_accounts.csv"), "Sample: two accounts on their own", n, "", _labels())
    if ex_full:
        run_full(pd.read_csv(MODELS / "sample_accounts.csv"), pd.read_csv(MODELS / "sample_replies.csv"),
                 "Sample upload (300 accounts, dataset format)", n, store)
    if not run:
        return
    if quick:
        if f_simple is not None:
            df = _read(f_simple)
            if df is not None:
                run_simple(df, f_simple.name, n, focus_text)
            return
        df = edited.astype(str).apply(lambda c: c.str.strip()).replace({"": None, "None": None, "nan": None, "<NA>": None})
        df = df.dropna(how="all", subset=[c for c in MANUAL_COLS if c != "tweet"])
        if df.empty:
            st.error("Add replies first: paste reply links in the table, or drop a replies file.")
            return
        t = (tweet or "").strip()
        if t:
            df["tweet"] = df["tweet"].fillna(t)
        if df["tweet"].isna().any():
            st.error("Paste the tweet link at the top (or fill the “tweet” column for every reply).")
            return
        name = "Typed in" + (" (example)" if ss.get("manual_example") else "")
        run_simple(pd.DataFrame(df), name, n, focus_text, _labels() if ss.get("manual_example") else None)
    else:
        if f_acc is not None:
            acc = _read(f_acc)
            rep = _read(f_rep) if f_rep is not None else None
            if acc is not None:
                run_full(acc, rep, f_acc.name, n, store)
            return
        with st.spinner("Finding groups and ranking accounts…"):
            _start(R.from_campaign(store, camp, n))
