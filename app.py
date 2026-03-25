from __future__ import annotations

import streamlit as st

from functions import run_app #run_app contains the main logic for the app (steps,UI, ML calls,...)
from api import get_spotify_user_client #function that creates a spotify client for the user)

st.set_page_config(
    page_title="Group Playlist Recommender", #browser tab title
    layout="wide",
    initial_sidebar_state="expanded" #initial sidebar open by default
)

# Global Styles
st.markdown(
    """
    <style> 
    /* Make global text bigger */
    html, body, .stApp {
        font-size: 18px;  /* base size up from default */
    }
    
    .main-title {
        font-size: 3rem;          /* bigger main title */
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .main-subtitle {
        font-size: 1.1rem;        /* bigger subtitle */
        color: #666;
        margin-bottom: 1.2rem;
    }

    /* Cards for each step – tighter padding so content starts closer to the edge */
    .block-container div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:has(.step-card) {
        background: radial-gradient(circle at top left, #fdfbfb 0, #ebedee 40%, #f7f7f7 100%); 
        border-radius: 1.1rem;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 1rem;
    }

    /* Invisible marker for step cards */
    .step-card {
        height: 0;
        margin: 0;
        padding: 0;
    }

    /* WHITE input backgrounds (number, text, select) */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;  /* make input text bigger */
    }

    /* WHITE + / - buttons on number input */
    div[data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;
    }

    /* Make most labels and normal text a bit larger */
    label, .stMarkdown p, .stMarkdown li, .stCheckbox, .stRadio, .stSlider label {
        font-size: 1rem !important;
    }

    /* Sidebar steps */
    .step-label {                       /* styling for each step in sidebar */
        padding: 0.35rem 0.5rem;
        border-radius: 0.8rem;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .step-done {                                /* green tinted background indicates completed step */
        background: rgba(34, 197, 94, 0.12);
        color: #166534;
    }
    .step-current {                             /* blue tinted background indicates current step */
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
    }
    .step-todo {                                /* grey tinted background indicates upcoming step */
        background: rgba(148, 163, 184, 0.12);
        color: #475569;
    }
    /* Make sidebar wider */
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
    }
    /* Song list rows, in step 3*/
    .song-row {
        background: #fafafa;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .song-title {
        font-weight: 600;
        font-size: 1rem;
        color: #111827;
    }
    .song-artist {
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 20px 0 5px 0;
        font-size: 0.9rem;
    }
    /* Force selectbox (dropdown) background to white */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,  #tells Streamlit it’s okay to render this HTML/CSS.
)

st.markdown(            #main title
    '<div class="main-title">Smart Playlist Generator</div>',
    unsafe_allow_html=True,
)
st.markdown(            #main subtitle
    '<div class="main-subtitle">'
    "Create group playlists that balance everyone’s taste."
    "</div>",
    unsafe_allow_html=True,
)


