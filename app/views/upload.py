"""Step 1 — start an investigation: a demo campaign from the dataset, or your own files."""
import pandas as pd
import streamlit as st

from app.core import review as R
from app.core.store import ROOT, Store
from app.ui.blocks import compact, html
from app.views.common import goto, pretty

MODELS = ROOT / "models"


@st.cache_resource(show_spinner=False)
def scorer():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from radar.deploy import Scorer
    return Scorer() if Scorer.available() else None


def _start(inv: R.Investigation) -> None:
    st.session_state["inv"] = inv
    st.session_state["sel"] = inv.queue.index[0] if len(inv.queue) else None
    st.session_state.pop("_last_click", None)
    goto(2)


def render(store: Store) -> None:
    html('<h1 class="page-title">Start an investigation</h1>'
         '<p class="lead">Load the accounts that replied to the tweets you are looking into. The model ranks them, '
         'groups the ones that keep replying together, and you review them one by one.</p>')
    inv = st.session_state.get("inv")
    if inv is not None:
        done, total = inv.progress()
        c1, c2 = st.columns([4, 1])
        c1.markdown(f'<div class="notice">Current investigation: <b>{inv.name}</b> — {done} of {total} reviewed. '
                    'Starting a new one replaces it (download your report first).</div>', unsafe_allow_html=True)
        if c2.button("Continue →", type="primary", width="stretch"):
            goto(2)

    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Use a campaign from the dataset")
            st.caption("Real accounts from a known influence operation. Good for trying the tool and for the demo.")
            camps = store.campaigns()
            c = st.selectbox("Campaign", list(camps.index), format_func=lambda x: f"{pretty(x)} — "
                             f"{compact(camps.at[x, 'accounts'])} accounts", key="up_camp")
            n = st.select_slider("Accounts to review", [25, 50, 100, 200], value=100, key="up_n_ds")
            st.caption(f"The {n} highest-scored accounts of {compact(camps.at[c, 'accounts'])}. Scores come from a model "
                       "that never saw this campaign.")
            if st.button("Start with this campaign", type="primary", width="stretch", key="go_ds"):
                with st.spinner("Finding groups and ranking accounts…"):
                    _start(R.from_campaign(store, c, n))

    with right:
        with st.container(border=True):
            st.markdown("#### Use your own data")
            manual_tab, simple_tab, full_tab = st.tabs(["Type it in", "Replies file", "Dataset format (76 features)"])
            with manual_tab:
                manual_entry()
            with simple_tab:
                simple_upload()
            with full_tab:
                full_upload(store)


@st.cache_resource(show_spinner=False)
def simple_scorer():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from radar.simple import SimpleScorer
    return SimpleScorer() if SimpleScorer.available() else None


def simple_upload() -> None:
    from app.core import simple_input as SI
    sc = simple_scorer()
    if sc is None:
        st.error("No simple model. Run `python -m radar.simple` first.")
        return
    st.caption("One row per **reply** — the kind of export you get from X or a social-listening tool. "
               "Needed: `account_id`, `tweet_id` (the tweet being replied to), `reply_time`. "
               "Helpful: `account_created_at`, `followers`, `following`, `reply_text`. "
               "Common X column names (author_id, in_reply_to_tweet_id, created_at, text) work too.")
    st.download_button("Download an example file (2,256 real replies)", (MODELS / "simple_sample.csv").read_bytes(),
                       "radar_replies_example.csv", "text/csv", width="stretch", key="dl_simple_ex",
                       help="Use it as a template: keep the column names, put your own rows under them.")
    f = st.file_uploader("Replies file (CSV)", type="csv", key="up_simple")
    ids = st.text_input("Check specific accounts (optional)", key="up_focus",
                        placeholder="account IDs, separated by commas — they will be shown first")
    n = st.select_slider("Accounts to review", [25, 50, 100, 200], value=100, key="up_n_simple")
    a, b, c = st.columns(3)
    go = a.button("Start with my file", type="primary", width="stretch", disabled=f is None, key="go_simple")
    go_s = b.button("Try the sample", width="stretch", key="go_simple_sample")
    go_2 = c.button("Try: check 2 accounts", width="stretch", key="go_simple_two")
    st.caption(f"Uses a lighter model built only from what such a file contains (timing, profile, who replies "
               f"with whom). On campaigns it never saw it is weaker than the full model: mean PR-AUC "
               f"{sc.meta['loco_mean_pr_auc']:.2f} vs 0.54. Reply text is checked by a separate rule for near-copies.")
    st.caption("**Checking one account?** Coordination is about accounts acting *together*, so the check is "
               "strongest when the file holds every reply to the tweets in question and you name the account above. "
               "A file with only that account's replies still works, but the model then sees its own behaviour only.")
    if not (go or go_s or go_2):
        st.caption("Samples: every reply to 30 tweets attacked in the Honduras operation (2,256 replies, 1,651 accounts), "
                   "or just two of those accounts on their own. None of these accounts were used to train the model.")
        return
    try:
        df = (pd.read_csv(MODELS / "simple_sample.csv") if go_s
              else pd.read_csv(MODELS / "example_two_accounts.csv") if go_2 else pd.read_csv(f))
    except Exception as e:  # noqa: BLE001 - show any parse error to the user
        st.error(f"Could not read the file: {e}")
        return
    r, errors, notes = SI.parse(df)
    for e in errors:
        st.error(e)
    if errors:
        return
    for m in notes:
        st.info(m)
    focus = SI.parse_ids(ids) if go else []
    accs = set(r["account"])
    missing = [x for x in focus if x not in accs and not (len(x) == 6 and any(a.endswith(x) for a in accs))]
    if missing:
        st.warning(f"Not in the file: {', '.join(missing[:5])}")
    labels = None
    if go_s or go_2:
        labels = pd.read_csv(MODELS / "simple_sample_labels.csv", dtype={"account_id": str}).set_index("account_id")["label"]
    name = ("Sample: Honduras replies (simple format)" if go_s else "Sample: two accounts on their own" if go_2
            else f.name)
    with st.spinner("Computing features and scoring accounts…"):
        _start(SI.build(r, sc, name, n, labels, focus=focus))


def full_upload(store: Store) -> None:
    sc = scorer()
    if sc is None:
        st.error("No saved model. Run `python -m radar.deploy` first.")
        return
    st.caption(f"**Accounts file** (required): one row per account, the {len(sc.features)} feature columns "
               "of the released dataset. **Replies file** (optional): two columns — account id, tweet id — "
               "used to find groups of accounts that reply to the same tweets.")
    st.download_button("Download an example file (300 real accounts)", (MODELS / "sample_accounts.csv").read_bytes(),
                       "radar_accounts_example.csv", "text/csv", width="stretch", key="dl_full_ex",
                       help="Use it as a template: keep the column names, put your own rows under them.")
    f_acc = st.file_uploader("Accounts file (CSV)", type="csv", key="up_acc")
    f_rep = st.file_uploader("Replies file (CSV, optional)", type="csv", key="up_rep")
    n2 = st.select_slider("Accounts to review", [25, 50, 100, 200], value=100, key="up_n_up")
    a, b = st.columns(2)
    go_up = a.button("Start with my files", type="primary", width="stretch", disabled=f_acc is None)
    go_sample = b.button("Try the sample files", width="stretch")
    if go_up or go_sample:
        try:
            acc = pd.read_csv(MODELS / "sample_accounts.csv") if go_sample else pd.read_csv(f_acc)
            rep = (pd.read_csv(MODELS / "sample_replies.csv") if go_sample
                   else pd.read_csv(f_rep) if f_rep is not None else None)
        except Exception as e:  # noqa: BLE001 - show any parse error to the user
            st.error(f"Could not read the file: {e}")
            return
        if acc.dropna(how="all").empty:
            st.error("The file has column names but no rows. Add one row per account "
                     "(download the example file to see what goes in it).")
            return
        missing, bad = sc.check(acc)
        if missing:
            st.error(f"{len(missing)} required columns are missing (e.g. {', '.join(missing[:4])}). "
                     "Use the template.")
            return
        if bad:
            st.error(f"These columns contain text that is not a number: {', '.join(bad[:4])}"
                     + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))
            return
        if rep is not None and rep.shape[1] < 2:
            st.error("The replies file needs two columns: account id, tweet id.")
            return
        with st.spinner("Scoring accounts with the model…"):
            _start(R.from_upload(sc, store._normal_stats(), acc, rep,
                                 "Sample upload (300 accounts, dataset format)" if go_sample else f_acc.name, n2))
    if go_sample is False:
        st.caption("The sample: 300 real accounts (30 known IO) and their 787 replies.")


MANUAL_COLS = ["tweet", "reply", "account", "reply_time", "followers", "following", "account_created_at", "reply_text"]


def _blank(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({c: [""] * n for c in MANUAL_COLS})


def manual_entry() -> None:
    """Type or paste replies straight into a table — no file needed."""
    from app.core import simple_input as SI
    sc = simple_scorer()
    if sc is None:
        st.error("No simple model. Run `python -m radar.simple` first.")
        return
    ss = st.session_state
    if "manual_df" not in ss:
        ss["manual_df"], ss["manual_v"], ss["manual_example"] = _blank(), 0, False
    st.caption("One row per **reply**. Fastest: paste the link of the tweet and the link of each reply "
               "(on X: Share → Copy link). The account and the times are read from the links. "
               "Or type the account and the reply time yourself. The other columns are optional but help.")
    a, b = st.columns(2)
    if a.button("Fill with an example (3 tweets, 85 replies)", width="stretch", key="man_ex"):
        ex = pd.read_csv(MODELS / "example_manual.csv", dtype=str)
        ss["manual_df"] = ex.reindex(columns=MANUAL_COLS).fillna("")
        ss["manual_v"] += 1
        ss["manual_example"] = True
        st.rerun()
    if b.button("Clear the table", width="stretch", key="man_clear"):
        ss["manual_df"], ss["manual_example"] = _blank(), False
        ss["manual_v"] += 1
        st.rerun()
    edited = st.data_editor(
        ss["manual_df"], num_rows="dynamic", width="stretch", height=300, key=f"manual_{ss['manual_v']}",
        column_config={
            "tweet": st.column_config.TextColumn("tweet (link or ID)", required=True,
                                                 help="The tweet being replied to, e.g. https://x.com/name/status/18…"),
            "reply": st.column_config.TextColumn("reply (link)", help="Link of the reply — gives account and time"),
            "account": st.column_config.TextColumn("account", help="@handle or ID; filled from the reply link if empty"),
            "reply_time": st.column_config.TextColumn("reply time", help="Only if there is no reply link, e.g. 2025-11-14 10:30"),
            "followers": st.column_config.TextColumn("followers"),
            "following": st.column_config.TextColumn("following"),
            "account_created_at": st.column_config.TextColumn("account created", help="e.g. 2025-10-01"),
            "reply_text": st.column_config.TextColumn("reply text", help="Used to spot near-copies"),
        })
    ids = st.text_input("Check specific accounts (optional)", key="man_focus",
                        placeholder="@handles or IDs, separated by commas — shown first in the results")
    n = st.select_slider("Accounts to list", [25, 50, 100, 200], value=100, key="man_n")
    if not st.button("Check these replies", type="primary", width="stretch", key="man_go"):
        return
    df = edited.astype(str).apply(lambda c: c.str.strip()).replace({"": None, "None": None, "nan": None, "<NA>": None})
    df = df.dropna(how="all")
    if df.empty:
        st.error("The table is empty — paste at least the tweet link and a few reply links.")
        return
    r, errors, notes = SI.parse(pd.DataFrame(df))
    for e in errors:
        st.error(e)
    if errors:
        return
    for m in notes:
        st.info(m)
    labels = None
    if ss.get("manual_example"):
        labels = pd.read_csv(MODELS / "simple_sample_labels.csv", dtype={"account_id": str}).set_index("account_id")["label"]
    focus = [x.lstrip("@").lower() for x in SI.parse_ids(ids)]
    accs = {a.lstrip("@").lower(): a for a in r["account"]}
    focus = [accs.get(x, x) for x in focus]
    with st.spinner("Scoring the accounts…"):
        _start(SI.build(r, sc, "Typed in" + (" (example)" if ss.get("manual_example") else ""), n, labels, focus=focus))

