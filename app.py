# ============================================================
# Swiss Vacation Planner — app.py
# Shell only: page config, CSS, title, call run_app()
# All step logic lives in functions.py
# ============================================================

from __future__ import annotations

import streamlit as st
from functions import run_app

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Swiss Vacation Planner",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Alpine design, no sidebar
# ─────────────────────────────────────────────

st.markdown("""
<style>

/* Hide sidebar entirely */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }

/* ── Base ── */
html, body, .stApp {
    font-size: 16px;
    background-color: #eef2f7;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a3a5c 0%, #2e6da4 60%, #4a9fd4 100%);
    border-radius: 1.4rem;
    padding: 2rem 2.4rem 1.6rem 2.4rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    opacity: 0.85;
}

/* ── Progress bar ── */
.progress-wrap {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.6rem;
}
.prog-step {
    flex: 1;
    padding: 0.45rem 0.5rem;
    border-radius: 0.6rem;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    background: #d1dde8;
    color: #5a7a9a;
}
.prog-step.done    { background: #34d399; color: #064e3b; }
.prog-step.current { background: #2e6da4; color: white;   }

/* ── Step cards ── */
.block-container div[data-testid="stVerticalBlock"]
div[data-testid="stVerticalBlock"]:has(.step-card) {
    background: white;
    border-radius: 1.2rem;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 16px rgba(26,58,92,0.07);
    border: 1px solid #ccdce8;
    margin-bottom: 1.4rem;
}
.step-card { height: 0; margin: 0; padding: 0; }

/* ── Step headings ── */
.step-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a3a5c;
    margin-bottom: 0.2rem;
}
.step-caption {
    font-size: 0.9rem;
    color: #6a8aaa;
    margin-bottom: 1rem;
}

/* ── Input fields ── */
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border-radius: 0.6rem !important;
    border: 1px solid #b0c8de !important;
    font-size: 1rem !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[role="combobox"] {
    background-color: #ffffff !important;
    border-radius: 0.6rem !important;
    border: 1px solid #b0c8de !important;
    box-shadow: none !important;
}

/* ── Labels ── */
label, .stMarkdown p, .stMarkdown li, .stCheckbox, .stRadio {
    font-size: 1rem !important;
    color: #1a3a5c;
}

/* ── Summary boxes ── */
.summary-box {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 0.8rem;
    padding: 0.7rem 1rem;
    font-size: 0.92rem;
    color: #1a3a5c;
    margin-bottom: 0.8rem;
}x  

/* ── Activity rows (Step 3) ── */
.act-row {
    background: #f7fafc;
    border: 1px solid #dbeafe;
    border-radius: 0.7rem;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.4rem;
}
.act-name { font-weight: 600; color: #1a3a5c; font-size: 0.97rem; }
.act-meta { font-size: 0.82rem; color: #5a8aaa; margin-top: 0.1rem; }

/* ── Timetable ── */
.tt-header {
    font-weight: 700;
    color: #1a3a5c;
    font-size: 0.88rem;
    padding: 0.4rem 0;
    border-bottom: 2px solid #2e6da4;
    margin-bottom: 0.5rem;
}
.tt-slot {
    border-radius: 0.5rem;
    padding: 0.35rem 0.55rem;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
    color: #1a3a5c;
    font-weight: 500;
}
.tt-morning   { background: #fef9c3; }
.tt-afternoon { background: #dcfce7; }
.tt-evening   { background: #fee2e2; }
.tt-night     { background: #ede9fe; }
.tt-free      { background: #f1f5f9; color: #94a3b8; font-style: italic; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #8aa8c0;
    padding: 28px 0 8px 0;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🏔️ Swiss Vacation Planner</div>'
    '<div class="hero-subtitle">'
    "Tell us what you're looking for and we'll build your perfect week in Switzerland."
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────

run_app()