"""Small HTML building blocks (editorial layout instead of stat cards)."""
from html import escape

import streamlit as st

TAG_HELP = {
    "MODEL": "held-out prediction of the trained classifier",
    "FEATURE": "feature released with the dataset",
    "RULE": "computed by a transparent rule in this app",
    "DATASET": "taken directly from the dataset",
    "NA": "not available in the released dataset",
    "TRUTH": "known label — evaluation only",
}


def html(s: str) -> None:
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def tag(kind: str) -> str:
    label = "N/A" if kind == "NA" else kind
    return f'<span class="tag {kind}" title="{TAG_HELP[kind]}">{label}</span>'


def section(num: str, name: str, note: str = "") -> None:
    html(f'<div class="section"><span class="num">{num}</span><span class="name">{escape(name)}</span>'
         f'<span class="note">{note}</span></div>')


def figures(items: list[tuple]) -> None:
    """items: (value, label, color or '', tag or '')"""
    cells = "".join(
        f'<div class="fig {c}"><div class="v">{v}</div><div class="l">{escape(l)}{tag(t) if t else ""}</div></div>'
        for v, l, c, t in items)
    html(f'<div class="figs">{cells}</div>')


def risk(tier: str, score: float) -> str:
    return (f'<div class="risk {tier}">{tier} PRIORITY<div class="bar"><i style="width:{score * 100:.0f}%"></i></div>'
            f'model score {score:.2f}{tag("MODEL")}</div>')


def evidence(title: str, value: str, text: str, kind: str, heat: str = "") -> None:
    html(f'<div class="evidence {heat}"><h4>{escape(title)}{tag(kind)}</h4>'
         f'<div class="big">{value}</div><p>{text}</p></div>')


def compact(n: float) -> str:
    n = float(n)
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,.0f}"


def legend() -> None:
    html('<div class="notice">' + " ".join(f"{tag(k)} {TAG_HELP[k]}" for k in TAG_HELP) + "</div>")
