# ── Imports ───────────────────────────────────────────────────────────────────
# Import corre Python utilities and third-party libraries
from __future__ import annotations

# Ensure local module imports work regardless of execution environment
# Allow imports from the same directory regardless of how the script is run
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent) if "__file__" in dir() else os.getcwd())

# Import Streamlit framework and supporting libraries
import streamlit as st
from functions import run_app  # Main app logic lives in functions.py
import pandas as pd
import pydeck as pdk

# Configure Streamlit page settings and layout
# ── Page Configuration ────────────────────────────────────────────────────────

# Set browser tab title, use full-width layout, and hide the sidebar by default
st.set_page_config(
    page_title="Swiss Vacation Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Inject custom CSS styling into the Streamlit application
# ── Custom CSS Styling ────────────────────────────────────────────────────────

# Inject CSS to style the app with a Swiss-themed palette (red #D52B1E, white, dark text)
st.markdown("""
<style>

/* Hide the sidebar toggle button and sidebar panel entirely */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }

/* Base styles: white background, Segoe UI font, dark text */
html, body, .stApp {
    font-size: 16px;
    background-color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
    color: #1a1a1a;
}

[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}

/* Force consistent dark text colours across all UI components */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown p, .stMarkdown span,
.stCheckbox label, .stCheckbox span,
.stSelectbox label, .stRadio label,
.stSlider label, .stDateInput label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-baseweb="select"] span {
    color: #1a1a1a !important;
}

/* Hero banner styling displayed at top of application */
.hero {
    background: #D52B1E;
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

/* Subtitle styling for application description */
.hero-subtitle {
    font-size: 1.05rem;
    color: #f5c6c3 !important;
}

/* Multi-step progress tracker styling */
.prog-step {
    padding: 0.45rem 0.5rem;
    border-radius: 0.6rem;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    background: #f0f0f0;
    color: #666666 !important;
}
.prog-step.done    { background: #1a1a1a; color: #ffffff !important; }
.prog-step.current { background: #D52B1E; color: #ffffff !important; }

/* Section heading and caption styling */
.step-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1a1a !important;
    margin-bottom: 0.2rem;
}
.step-caption {
    font-size: 0.92rem;
    color: #555555 !important;
    margin-bottom: 1.2rem;
}

/* Summary card styling for generated trip overview */
.summary-box {
    background: #fff0f0;
    border: 1px solid #f5c6c3;
    border-radius: 0.8rem;
    padding: 0.8rem 1.1rem;
    font-size: 0.93rem;
    color: #1a1a1a !important;
    margin-bottom: 0.9rem;
}

/* Metadata styling for activity details and labels */
.act-meta {
    font-size: 0.83rem;
    color: #555555 !important;
    margin-top: 0.1rem;
    margin-bottom: 0.4rem;
}

/* Timetable layout styling for itinerary schedule */
.tt-header {
    font-weight: 700;
    color: #1a1a1a !important;
    font-size: 0.9rem;
    padding: 0.4rem 0;
    border-bottom: 2px solid #D52B1E;
    margin-bottom: 0.5rem;
}
.tt-slot {
    border-radius: 0.5rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.82rem;
    margin-bottom: 0.3rem;
    color: #1a1a1a !important;
    font-weight: 500;
}

/* Colour-coded activity blocks by time of day */
.tt-morning   { background: #fde8e6; color: #1a1a1a !important; }
.tt-afternoon { background: #fde8e6; color: #1a1a1a !important; }
.tt-evening   { background: #fde8e6; color: #1a1a1a !important; }
.tt-night     { background: #fde8e6; color: #1a1a1a !important; }
.tt-free      { background: #f5f5f5; color: #999999 !important; font-style: italic; }

/* Weather information card styling */
.weather-box {
    background: #fde8e6;
    border-left: 3px solid #D52B1E;
    border-radius: 0.4rem;
    padding: 0.3rem 0.6rem;
    font-size: 0.85rem;
    color: #1a1a1a !important;
    margin-bottom: 0.5rem;
}

/* Custom styling for Streamlit form inputs and selectboxes */
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border-radius: 0.6rem !important;
    border: 1px solid #cccccc !important;
    font-size: 1rem !important;
    color: #1a1a1a !important;
}

/* Selectbox container styling */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #cccccc !important;
}

/* Label and markdown text styling */
label, .stMarkdown p, .stMarkdown li {
    font-size: 1rem !important;
    color: #1a1a1a !important;
}

/* Primary action button styling using Swiss-themed colours */
.stButton > button,
.stDownloadButton > button {
    background-color: #D52B1E !important;
    color: #ffffff !important;
    border: none;
    border-radius: 0.6rem;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: background-color 0.2s;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #a01f16 !important;
    color: #ffffff !important;
}

/* Footer styling shown at bottom of application */
.footer {
    text-align: center;
    color: #999999 !important;
    padding: 28px 0 8px 0;
    font-size: 0.85rem;
}

/* Custom dropdown menu styling overriding Streamlit defaults */
ul[data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

li[role="option"],
[data-baseweb="menu"] li,
[role="option"] span {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

li[role="option"]:hover,
[role="option"]:hover {
    background-color: #fee2e2 !important;
    color: #1a1a1a !important;
}

/* Placeholder text styling inside selectboxes */
[data-baseweb="select"] [data-testid="stSelectboxPlaceholder"],
div[data-baseweb="select"] span {
    color: #555555 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────

# Render the top banner with the app title and tagline
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Swiss Vacation Planner</div>'
    '<div class="hero-subtitle">'
    "Tell us where you want to go and we'll build your perfect Swiss trip — day by day."
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
# Locate CSV file containing activity coordinates
# ── Activity Map ──────────────────────────────────────────────────────────────

# Locate the CSV file containing activity coordinates (same directory as this script)
try:
    _CSV = pathlib.Path(__file__).resolve().parent / "locations.csv"
except NameError:
    # Fallback for interactive environments where __file__ is not defined
    _CSV = pathlib.Path("locations.csv")

# Load activity location data from CSV
df_raw = pd.read_csv(_CSV)

st.markdown(
    '<div class="step-heading">Activities across Switzerland</div>',
    unsafe_allow_html=True,
)

# Render an interactive PyDeck map centred on Switzerland
# Each activity is shown as a red dot; hovering reveals name, city, and category
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
            data=df_raw,
            get_position="[lon, lat]",   # Column names from locations.csv
            get_radius=300,              # Circle radius in metres
            get_color=[220, 38, 38, 200], # Swiss red with slight transparency
            get_line_color=[255, 255, 255],
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,               # Enable hover tooltip
        )
        # Tooltip configuration displayed when hovering over markers
    ],
    tooltip={
        "html": "<b>{activity_name}</b><br/>{city}<br/><i>{category}</i>",
        "style": {
            "color": "white",
            "backgroundColor": "#EDF1F6",
            "padding": "6px 10px",
            "borderRadius": "6px",
        },
    },
))
# Main application execution
# Launch planner interface and core application workflow
# ── Main App Logic ────────────────────────────────────────────────────────────

# Hand off to run_app() in functions.py, which handles the multi-step planner UI
run_app()
