"""Design tokens from the Figma file, injected as CSS and small HTML snippets."""
import streamlit as st

TOKENS = {
    "bg": "#F8F9FF",
    "surface": "#FFFFFF",
    "ink": "#0B1C30",
    "ink_2": "#464555",
    "ink_3": "#777587",
    "primary": "#4F46E5",
    "primary_strong": "#3525CD",
    "ok": "#006C4D",
    "err": "#BA1A1A",
    "warn": "#825800",
    "tint": "#EFF4FF",
    "tint_2": "#E5EEFF",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root {
  --bg: #F8F9FF; --surface: #FFFFFF; --ink: #0B1C30; --ink-2: #464555; --ink-3: #777587;
  --primary: #4F46E5; --primary-strong: #3525CD; --ok: #006C4D; --err: #BA1A1A;
  --warn: #825800; --tint: #EFF4FF; --tint-2: #E5EEFF; --border: rgba(199,196,216,0.3);
}
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
.card {
  background: var(--surface); border-radius: 8px; padding: 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 12px;
}
.kpi-label { font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-2); font-weight: 500; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 600; margin: 6px 0; }
.kpi-sub { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.tone-ok { color: var(--ok); } .tone-err { color: var(--err); } .tone-info { color: var(--primary-strong); }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; border-radius: 12px; padding: 2px 8px; display: inline-block; }
.badge-ok { background: rgba(0,108,77,0.12); color: var(--ok); }
.badge-err { background: #FFDAD6; color: #93000A; }
.badge-info { background: rgba(53,37,205,0.10); color: var(--primary-strong); }
.section-title { font-weight: 700; font-size: 15px; margin: 0; }
.section-sub { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-2); }
.hero-title { font-weight: 700; font-size: 22px; letter-spacing: -0.02em; margin: 0; }
.hero-sub { font-size: 13px; color: var(--ink-2); }
.insight { background: var(--tint); border-radius: 4px; padding: 8px 10px; font-size: 12px; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def badge_html(text, tone="info"):
    return f'<span class="badge badge-{tone}">{text}</span>'


def card_html(body):
    return f'<div class="card">{body}</div>'


def kpi_card_html(label, value, sub, tone="info"):
    return card_html(
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub tone-{tone}">{sub}</div>'
    )


def section_header(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<p class="section-title">{title}</p>{sub}'
