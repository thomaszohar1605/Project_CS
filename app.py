# app.py
from __future__ import annotations

import streamlit as st
from functions import run_app

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Swiss Vacation Planner",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* Force all text to navy blue */
html, body, .stApp, p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stText, .stCheckbox label, .stCheckbox span,
.stSelectbox label, .stRadio label,
.stSlider label, .stDateInput label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-baseweb="select"] span,
.stButton p {
    color: #1a3a5c !important;
}

/* Hide sidebar */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }

/* Base */
html, body, .stApp {
    font-size: 16px;
    background-color: #eef2f7;
    font-family: 'Segoe UI', sans-serif;
}

/* Hero */
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
    color: white !important;
}
.hero-subtitle {
    font-size: 1.05rem;
    opacity: 0.85;
    color: white !important;
}

/* Progress bar - 5 steps */
.prog-step {
    padding: 0.45rem 0.5rem;
    border-radius: 0.6rem;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    background: #d1dde8;
    color: #5a7a9a !important;
}
.prog-step.done    { background: #34d399; color: #064e3b !important; }
.prog-step.current { background: #2e6da4; color: white !important; }

/* Step headings */
.step-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a3a5c !important;
    margin-bottom: 0.2rem;
}
.step-caption {
    font-size: 0.9rem;
    color: #6a8aaa !important;
    margin-bottom: 1rem;
}

/* Summary boxes */
.summary-box {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 0.8rem;
    padding: 0.7rem 1rem;
    font-size: 0.92rem;
    color: #1a3a5c !important;
    margin-bottom: 0.8rem;
}

/* Activity cards */
.act-meta {
    font-size: 0.82rem;
    color: #5a8aaa !important;
    margin-top: 0.1rem;
    margin-bottom: 0.3rem;
}

/* Timetable */
.tt-header {
    font-weight: 700;
    color: #1a3a5c !important;
    font-size: 0.88rem;
    padding: 0.4rem 0;
    border-bottom: 2px solid #2e6da4;
    margin-bottom: 0.5rem;
}
.tt-slot {
    border-radius: 0.5rem;
    padding: 0.35rem 0.55rem;
    font-size: 0.80rem;
    margin-bottom: 0.25rem;
    color: #1a3a5c !important;
    font-weight: 500;
}
.tt-morning   { background: #fef9c3; }
.tt-afternoon { background: #dcfce7; }
.tt-evening   { background: #fee2e2; }
.tt-night     { background: #ede9fe; }
.tt-free      { background: #f1f5f9; color: #94a3b8 !important; font-style: italic; }

/* Input fields */
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border-radius: 0.6rem !important;
    border: 1px solid #b0c8de !important;
    font-size: 1rem !important;
    color: #1a3a5c !important;
}

/* Labels */
label, .stMarkdown p, .stMarkdown li {
    font-size: 1rem !important;
    color: #1a3a5c !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #8aa8c0 !important;
    padding: 28px 0 8px 0;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🏔️ Swiss Vacation Planner</div>'
    '<div class="hero-subtitle">'
    "Tell us where you want to go and we'll build your perfect Swiss trip — day by day."
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Run app ───────────────────────────────────────────────────────────────────

run_app()