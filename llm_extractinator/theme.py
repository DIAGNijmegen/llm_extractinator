"""Shared visual theme for the LLM Extractinator Studio & schema builder.

Streamlit's ``config.toml`` theming is only picked up from the *working
directory* the app is launched in, which for this project is the user's own
project folder — so it can't be relied on. Instead we inject the theme as CSS at
runtime, which always applies regardless of where the app is started from.

Usage
-----
    from theme import inject_theme, app_header

    inject_theme()
    app_header("LLM Extractinator", "Structured extraction from unstructured text")
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

# ──────────────────────────── Brand palette ─────────────────────────────
# Derived from the project logo (navy platypus + teal document), but tuned for
# comfort: a soft warm-neutral background instead of a cool bright blue-white,
# so the page doesn't glare against the white cards. Navy + teal stay as the
# brand accents.
NAVY = "#1b2838"       # deep slate-navy — sidebar, headings
NAVY_2 = "#26405a"     # secondary navy (header gradient)
TEAL = "#128a83"       # primary action / accent (calmer than electric teal)
TEAL_HOVER = "#0d6f69"
TEAL_SOFT = "#e8f1ef"  # tinted teal background
BG = "#f4f3f0"         # warm light greige — easy on the eyes, no blue glare
SURFACE = "#fffefb"    # barely-warm white for cards / inputs
BORDER = "#e7e3db"     # warm hairline borders
TEXT = "#343a41"       # body text
MUTED = "#79787d"      # secondary text
SUCCESS = "#2f9e6b"
DANGER = "#cf5563"

_ASSETS = Path(__file__).parent / "assets"

FONT_STACK = (
    'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, '
    '"Apple Color Emoji", "Segoe UI Emoji", sans-serif'
)


@lru_cache(maxsize=8)
def _asset_b64(name: str) -> str | None:
    """Return a base64 data-URI payload for a packaged asset, or None."""
    path = _ASSETS / name
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ──────────────────────────── CSS template ──────────────────────────────
# Placeholders (__NAME__) are substituted from the palette above so we don't
# have to escape the many literal braces CSS needs.
_CSS = """
<style>
:root {
  --lx-navy: __NAVY__;
  --lx-navy2: __NAVY_2__;
  --lx-teal: __TEAL__;
  --lx-teal-hover: __TEAL_HOVER__;
  --lx-teal-soft: __TEAL_SOFT__;
  --lx-bg: __BG__;
  --lx-surface: __SURFACE__;
  --lx-border: __BORDER__;
  --lx-text: __TEXT__;
  --lx-muted: __MUTED__;
}

/* ---- Base ---- */
html, body, [data-testid="stAppViewContainer"], .stApp {
  font-family: __FONT__;
  color: var(--lx-text);
}
[data-testid="stAppViewContainer"], .stApp { background: var(--lx-bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 3.5rem; max-width: 1000px; }

h1, h2, h3, h4 { color: var(--lx-navy); font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.4rem; margin-top: .4rem; }
h3 { font-size: 1.12rem; }
a { color: var(--lx-teal); }

/* ---- Branded header bar ---- */
.lx-header {
  display: flex; align-items: center; gap: 16px;
  background: linear-gradient(100deg, var(--lx-navy) 0%, var(--lx-navy2) 100%);
  border-radius: 16px; padding: 18px 24px; margin-bottom: 20px;
  box-shadow: 0 6px 20px rgba(22,35,63,.18);
}
.lx-header img { height: 52px; width: auto; display: block; }
.lx-header .lx-title { color: #fff; font-size: 1.55rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; }
.lx-header .lx-sub { color: #b9c6dc; font-size: .92rem; margin-top: 2px; }
.lx-header .lx-badge {
  margin-left: auto; color: #cfe9e8; background: rgba(18,163,160,.20);
  border: 1px solid rgba(18,163,160,.45); padding: 4px 12px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; white-space: nowrap;
}

/* ---- Sidebar (dark) ---- */
section[data-testid="stSidebar"] {
  background: var(--lx-navy); border-right: 1px solid rgba(255,255,255,.06);
  min-width: 292px !important; max-width: 292px !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }
section[data-testid="stSidebar"] hr { margin: 1.1rem 0; }

/* Sidebar brand block */
.lx-sb-brand { display: flex; align-items: center; gap: 12px; padding: 0 2px 2px; }
.lx-sb-brand img { height: 40px; width: auto; display: block; }
.lx-sb-brand .n1 { color: #fff; font-weight: 800; font-size: 1.12rem; line-height: 1; letter-spacing: -0.01em; }
.lx-sb-brand .n2 { color: #7fc7c0; font-weight: 700; font-size: .66rem; text-transform: uppercase; letter-spacing: .18em; margin-top: 5px; }

/* Sidebar section label */
.lx-sb-label { color: #8593a8; font-size: .66rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin: 0 0 12px; }

/* Sidebar workflow guide */
.lx-sb-flow { display: flex; flex-direction: column; gap: 13px; }
.lx-sb-step { display: flex; align-items: flex-start; gap: 11px; }
.lx-sb-step .num {
  flex: none; width: 23px; height: 23px; border-radius: 7px;
  background: rgba(18,138,131,.16); border: 1px solid rgba(18,138,131,.5);
  color: #7fd8d1; font-size: .74rem; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
}
.lx-sb-step .txt { color: #e7edf6; font-size: .86rem; font-weight: 600; line-height: 1.15; }
.lx-sb-step .sub { color: #8593a8; font-size: .74rem; font-weight: 400; margin-top: 1px; }

/* Sidebar footer */
.lx-sb-foot { display: flex; align-items: center; gap: 10px; font-size: .74rem; color: #8593a8; }
.lx-sb-foot a { color: #9fc7c2; text-decoration: none; font-weight: 600; }
.lx-sb-foot a:hover { color: #cfe9e8; }
.lx-sb-foot .dot { color: #4a5a72; }

section[data-testid="stSidebar"] * { color: #e7edf6; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #94a4bd !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff; }
section[data-testid="stSidebar"] code { background: rgba(255,255,255,.10); color: #cfe9e8; }
/* Broad + !important so the dark-sidebar button style wins regardless of how
   this Streamlit build wraps its buttons. */
section[data-testid="stSidebar"] button {
  background: rgba(255,255,255,.07) !important;
  color: #e7edf6 !important;
  border: 1px solid rgba(255,255,255,.18) !important;
}
section[data-testid="stSidebar"] button:hover {
  background: rgba(255,255,255,.15) !important;
  border-color: rgba(255,255,255,.32) !important;
  color: #fff !important;
}
section[data-testid="stSidebar"] button * { color: #e7edf6 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.10); }
/* Readable path chip on the dark sidebar (replaces st.code, which renders a
   light block that's unreadable here). */
.lx-workdir {
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.14);
  color: #cfe0dc;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .78rem; line-height: 1.45;
  padding: 8px 11px; border-radius: 8px; word-break: break-all;
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--lx-border);
  background: var(--lx-surface); color: var(--lx-navy); transition: all .15s ease;
  padding: .45rem 1.0rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--lx-teal); color: var(--lx-teal); background: var(--lx-teal-soft);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--lx-teal); border-color: var(--lx-teal); color: #fff;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: var(--lx-teal-hover); border-color: var(--lx-teal-hover); color: #fff;
}

/* ---- Tabs ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--lx-border); }
[data-testid="stTabs"] [data-baseweb="tab"] {
  height: 44px; padding: 0 18px; font-weight: 600; color: var(--lx-muted);
  border-radius: 10px 10px 0 0;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: var(--lx-navy); background: rgba(18,163,160,.06); }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--lx-teal) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--lx-teal); height: 3px; }

/* ---- Inputs ---- */
[data-baseweb="input"], [data-baseweb="select"] > div, .stTextArea textarea {
  border-radius: 10px !important;
}
[data-testid="stTextInput"] input:focus, .stTextArea textarea:focus { border-color: var(--lx-teal) !important; }

/* ---- Metrics as cards ---- */
[data-testid="stMetric"] {
  background: var(--lx-surface); border: 1px solid var(--lx-border);
  border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(22,35,63,.05);
}
[data-testid="stMetricValue"] { color: var(--lx-navy); font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--lx-muted); }

/* ---- Expanders as cards ---- */
[data-testid="stExpander"] {
  border: 1px solid var(--lx-border); border-radius: 14px; background: var(--lx-surface);
  box-shadow: 0 1px 3px rgba(22,35,63,.05); overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; color: var(--lx-navy); }
[data-testid="stExpander"] summary:hover { color: var(--lx-teal); }

/* ---- Alerts ---- */
[data-testid="stAlert"] { border-radius: 12px; }

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--lx-border); border-radius: 12px; }

/* ---- Bordered containers become cards (st.container(border=True)) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--lx-surface); border-radius: 14px;
  box-shadow: 0 1px 3px rgba(27,40,56,.05);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: var(--lx-border) !important; border-radius: 14px; }

/* ---- Section card helper ---- */
.lx-card {
  background: var(--lx-surface); border: 1px solid var(--lx-border);
  border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(22,35,63,.05);
}
.lx-step { color: var(--lx-teal); font-weight: 700; font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; }

/* ---- Status strip ---- */
.lx-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: -4px 0 18px; }
.lx-chip {
  display: flex; align-items: center; gap: 9px; background: var(--lx-surface);
  border: 1px solid var(--lx-border); border-radius: 11px; padding: 8px 14px;
  box-shadow: 0 1px 2px rgba(22,35,63,.04);
}
.lx-chip-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.lx-chip-label { font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; color: var(--lx-muted); font-weight: 700; }
.lx-chip-val { font-size: .86rem; color: var(--lx-navy); font-weight: 600; }
</style>
"""

_SUBS = {
    "__NAVY__": NAVY,
    "__NAVY_2__": NAVY_2,
    "__TEAL__": TEAL,
    "__TEAL_HOVER__": TEAL_HOVER,
    "__TEAL_SOFT__": TEAL_SOFT,
    "__BG__": BG,
    "__SURFACE__": SURFACE,
    "__BORDER__": BORDER,
    "__TEXT__": TEXT,
    "__MUTED__": MUTED,
    "__FONT__": FONT_STACK,
}


def inject_theme() -> None:
    """Inject the brand CSS. Call once per page, right after set_page_config."""
    css = _CSS
    for token, value in _SUBS.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Render the brand block (logo + wordmark) at the top of the sidebar."""
    icon = _asset_b64("logo_icon.png")
    img = f'<img src="data:image/png;base64,{icon}" alt="logo"/>' if icon else ""
    st.markdown(
        f'<div class="lx-sb-brand">{img}'
        '<div><div class="n1">Extractinator</div>'
        '<div class="n2">Studio</div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar_flow(steps: list[tuple[str, str]]) -> None:
    """Render a numbered workflow guide in the sidebar.

    ``steps`` is a list of (title, subtitle) pairs.
    """
    html = '<div class="lx-sb-flow">'
    for i, (title, sub) in enumerate(steps, 1):
        html += (
            '<div class="lx-sb-step">'
            f'<div class="num">{i}</div>'
            f'<div><div class="txt">{title}</div><div class="sub">{sub}</div></div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def status_strip(items: list[tuple[str, str, bool]]) -> None:
    """Render a row of status chips under the header.

    Parameters
    ----------
    items : list of (label, value, ok)
        ``ok`` toggles the leading dot colour (teal when set, grey when not).
    """
    chips = ""
    for label, value, ok in items:
        dot = TEAL if ok else "#c4ccd8"
        chips += (
            '<div class="lx-chip">'
            f'<span class="lx-chip-dot" style="background:{dot}"></span>'
            f'<span class="lx-chip-label">{label}</span>'
            f'<span class="lx-chip-val">{value}</span>'
            "</div>"
        )
    st.markdown(f'<div class="lx-strip">{chips}</div>', unsafe_allow_html=True)


def app_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    """Render the branded header bar with the embedded logo icon."""
    icon = _asset_b64("logo_icon.png")
    img_html = (
        f'<img src="data:image/png;base64,{icon}" alt="logo"/>' if icon else ""
    )
    sub_html = f'<div class="lx-sub">{subtitle}</div>' if subtitle else ""
    badge_html = f'<div class="lx-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="lx-header">
          {img_html}
          <div>
            <div class="lx-title">{title}</div>
            {sub_html}
          </div>
          {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
