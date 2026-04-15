from __future__ import annotations

import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent) if "__file__" in dir() else os.getcwd())

import streamlit as st
from functions import run_app
import pandas as pd
import pydeck as pdk


st.set_page_config(
    page_title="Swiss Vacation Planner",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>

/* Hide sidebar */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }

/* Base */
html, body, .stApp {
    font-size: 16px;
    background-color: #f0f6fc;
    font-family: 'Segoe UI', sans-serif;
    color: #1a3a5c;
}

/* Force all text navy */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown p, .stMarkdown span,
.stCheckbox label, .stCheckbox span,
.stSelectbox label, .stRadio label,
.stSlider label, .stDateInput label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-baseweb="select"] span {
    color: #1a3a5c !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #2e6da4 0%, #4a9fd4 100%);
    border-radius: 1.4rem;
    padding: 2.2rem 2.6rem 1.8rem 2.6rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #dceffe !important;
}

/* Progress bar */
.prog-step {
    padding: 0.45rem 0.5rem;
    border-radius: 0.6rem;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    background: #dce8f0;
    color: #5a7a9a !important;
}
.prog-step.done    { background: #34d399; color: #064e3b !important; }
.prog-step.current { background: #2e6da4; color: #ffffff !important; }

/* Step headings */
.step-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a3a5c !important;
    margin-bottom: 0.2rem;
}
.step-caption {
    font-size: 0.92rem;
    color: #4a7a9b !important;
    margin-bottom: 1.2rem;
}

/* Summary boxes */
.summary-box {
    background: #e8f4fd;
    border: 1px solid #a8cfe8;
    border-radius: 0.8rem;
    padding: 0.8rem 1.1rem;
    font-size: 0.93rem;
    color: #1a3a5c !important;
    margin-bottom: 0.9rem;
}

/* Activity meta */
.act-meta {
    font-size: 0.83rem;
    color: #4a7a9b !important;
    margin-top: 0.1rem;
    margin-bottom: 0.4rem;
}

/* Timetable */
.tt-header {
    font-weight: 700;
    color: #1a3a5c !important;
    font-size: 0.9rem;
    padding: 0.4rem 0;
    border-bottom: 2px solid #2e6da4;
    margin-bottom: 0.5rem;
}
.tt-slot {
    border-radius: 0.5rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.82rem;
    margin-bottom: 0.3rem;
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
    border: 1px solid #a8c8e0 !important;
    font-size: 1rem !important;
    color: #1a3a5c !important;
}

/* Labels */
label, .stMarkdown p, .stMarkdown li {
    font-size: 1rem !important;
    color: #1a3a5c !important;
}

/* Buttons */
.stButton > button {
    background-color: #2e6da4;
    color: #ffffff !important;
    border: none;
    border-radius: 0.6rem;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: background-color 0.2s;
}
.stButton > button:hover {
    background-color: #1a3a5c;
    color: #ffffff !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #8aa8c0 !important;
    padding: 28px 0 8px 0;
    font-size: 0.85rem;
}

/* Dropdown menu background */
ul[data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #e8f4fd !important;
    color: #1a3a5c !important;
}

/* Each option in the dropdown */
li[role="option"],
[data-baseweb="menu"] li,
[role="option"] span {
    background-color: #e8f4fd !important;
    color: #1a3a5c !important;
}

/* Hovered option */
li[role="option"]:hover,
[role="option"]:hover {
    background-color: #b0d4f0 !important;
    color: #1a3a5c !important;
}

/* Placeholder text in dropdowns */
[data-baseweb="select"] [data-testid="stSelectboxPlaceholder"],
[data-baseweb="select"] placeholder,
div[data-baseweb="select"] span {
    color: #4a9fd4 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🏔️ Swiss Vacation Planner</div>'
    '<div class="hero-subtitle">'
    "Tell us where you want to go and we'll build your perfect Swiss trip — day by day."
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Activity Map ───────────────────────────────────────────────────────────────
import pathlib
try:
    _CSV = pathlib.Path(__file__).resolve().parent / "locations.csv"
except NameError:
    _CSV = pathlib.Path("locations.csv")
df = pd.read_csv(_CSV)

st.markdown('<div class="step-heading">📍 Activity Locations across Switzerland</div>', unsafe_allow_html=True)

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=46.8,
        longitude=8.2,
        zoom=7,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_radius=75,
            get_color=[220, 38, 38, 200],
            get_line_color=[255, 255, 255],
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,
        )
    ],
    tooltip={
        "html": "{name}",
        "style": {"color": "white", "backgroundColor": "#1a3a5c", "padding": "6px 10px", "borderRadius": "6px"},
    },
))

# ──────────────────────────────────────────────────────────────────────────────

run_app()
