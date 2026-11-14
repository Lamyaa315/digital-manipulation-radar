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
    goto(1)


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
            goto(1)

    left, right = st.columns(2, gap="large")
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
            st.markdown("#### Upload your own data")
            sc = scorer()
            if sc is None:
                st.error("No saved model. Run `python -m radar.deploy` first.")
                return
            st.caption(f"**Accounts file** (required): one row per account, the {len(sc.features)} feature columns "
                       "of the released dataset. **Replies file** (optional): two columns — account id, tweet id — "
                       "used to find groups of accounts that reply to the same tweets.")
            st.download_button("Download the accounts template", (MODELS / "template.csv").read_bytes(),
                               "radar_accounts_template.csv", "text/csv", width="stretch")
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
                missing, bad = sc.check(acc)
                if missing:
                    st.error(f"{len(missing)} required columns are missing (e.g. {', '.join(missing[:4])}). "
                             "Use the template.")
                    return
                if bad:
                    st.error(f"These columns must contain numbers: {', '.join(bad[:4])}")
                    return
                if rep is not None and rep.shape[1] < 2:
                    st.error("The replies file needs two columns: account id, tweet id.")
                    return
                with st.spinner("Scoring accounts with the model…"):
                    _start(R.from_upload(sc, store._normal_stats(), acc, rep,
                                         "Sample upload (300 accounts)" if go_sample else f_acc.name, n2))
            if go_sample is False:
                st.caption("The sample: 300 real accounts (30 known IO) and their 787 replies.")
