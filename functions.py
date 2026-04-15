from __future__ import annotations

import os
import random
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Constants ──────────────────────────────────────────────────────────────────

CITIES = [
    "Zurich", "Bern", "Lucerne", "Geneva", "Lausanne", "Montreux",
    "Zermatt", "Interlaken", "Grindelwald", "Davos", "St. Moritz",
    "Lugano", "Locarno", "St. Gallen", "Appenzell", "Chur",
]

ACTIVITY_KEYWORDS: dict[str, list[str]] = {
    "Outdoor & Nature":       ["Hike", "Ski", "Swim", "Paddle", "Cycle", "Walk", "Paraglid",
                                "Kayak", "Climb", "Snowshoe", "Sled", "Gorge", "Glacier", "Lake",
                                "Mountain", "Forest", "Botanical", "River", "Sunrise", "SUP"],
    "Culture & History":      ["Museum", "Cathedral", "Old Town", "Monument", "Library", "UNESCO",
                                "Tour", "Church", "Castle", "Abbey", "Historical", "Textile",
                                "Photography", "Art", "Architecture", "Literary"],
    "Food & Drink":           ["Fondue", "Wine", "Cheese", "Chocolate", "Food", "Dinner",
                                "Tasting", "Raclette", "Brunch", "Aperitivo", "Beer", "Schnapps",
                                "Cooking", "Gelato", "Farmers Market"],
    "Nightlife & Entertainment": ["Night", "Bar", "Club", "Jazz", "Opera", "Comedy", "Festival",
                                   "Concert", "Cinema", "Show", "Casino", "Pub Crawl", "Improv",
                                   "Disco", "Ghost"],
    "Relaxation & Wellness":  ["Spa", "Yoga", "Wellness", "Thermalbad", "Relax", "Meditation",
                                "Picnic", "Stroll", "Promenade"],
    "Adventure & Sports":     ["Skydiv", "Bungee", "Swing", "Rafting", "Via Ferrata", "Canyon",
                                "Zip", "Abseil", "Paraglid", "Skate", "Basketball", "Bubble",
                                "Escape Room", "Scavenger"],
}

SLOTS = ["Morning", "Afternoon", "Evening"]
SLOT_CLASSES = {"Morning": "tt-morning", "Afternoon": "tt-afternoon", "Evening": "tt-evening"}
SLOT_ICONS  = {"Morning": "🌅", "Afternoon": "☀️", "Evening": "🌙"}

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_activities() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_HERE, "locations.csv"))


def city_activities(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return rows whose name contains the city name (case-insensitive)."""
    # Map city names to the keywords actually used in locations.csv
    city_map = {
        "St. Moritz": "St. Moritz",
        "St. Gallen": "St. Gallen",
    }
    keyword = city_map.get(city, city)
    return df[df["name"].str.contains(keyword, case=False, na=False)].reset_index(drop=True)


def filter_by_preferences(activities: pd.DataFrame, prefs: list[str]) -> pd.DataFrame:
    if not prefs:
        return activities
    keywords = []
    for pref in prefs:
        keywords.extend(ACTIVITY_KEYWORDS.get(pref, []))
    mask = activities["name"].apply(
        lambda name: any(kw.lower() in name.lower() for kw in keywords)
    )
    filtered = activities[mask]
    return filtered if not filtered.empty else activities  # fall back to all if no match


def build_itinerary(activities: pd.DataFrame, num_days: int) -> list[dict]:
    """Assign activities to morning/afternoon/evening slots across num_days."""
    pool = activities["name"].tolist()
    random.shuffle(pool)
    itinerary = []
    idx = 0
    for day in range(1, num_days + 1):
        day_plan: dict[str, str] = {}
        for slot in SLOTS:
            if idx < len(pool):
                day_plan[slot] = pool[idx]
                idx += 1
            else:
                day_plan[slot] = "Free time — explore at your own pace"
        itinerary.append({"day": day, "slots": day_plan})
    return itinerary


# ── Progress bar ───────────────────────────────────────────────────────────────

def render_progress(current_step: int) -> None:
    steps = ["1 · Destination", "2 · Preferences", "3 · Your Itinerary"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        step_num = i + 1
        if step_num < current_step:
            css = "prog-step done"
        elif step_num == current_step:
            css = "prog-step current"
        else:
            css = "prog-step"
        col.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
    st.write("")


# ── Step renderers ─────────────────────────────────────────────────────────────

def step_destination() -> None:
    render_progress(1)
    st.markdown('<div class="step-heading">Where are you heading?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Pick a Swiss destination and how many days you have.</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", CITIES, index=0)
    with col2:
        num_days = st.selectbox("Number of days", list(range(1, 8)), index=2)

    st.write("")
    if st.button("Next →"):
        st.session_state["city"] = city
        st.session_state["num_days"] = num_days
        st.session_state["step"] = 2
        st.rerun()


def step_preferences() -> None:
    render_progress(2)
    st.markdown('<div class="step-heading">What do you enjoy?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Choose one or more activity types — or skip to include everything.</div>',
                unsafe_allow_html=True)

    prefs = []
    cols = st.columns(3)
    for i, category in enumerate(ACTIVITY_KEYWORDS):
        with cols[i % 3]:
            if st.checkbox(category, value=False):
                prefs.append(category)

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Build my itinerary →"):
            st.session_state["prefs"] = prefs
            # Build itinerary now so it stays stable on re-renders
            df = load_activities()
            acts = city_activities(df, st.session_state["city"])
            if acts.empty:
                acts = df  # fallback: use all activities
            acts = filter_by_preferences(acts, prefs)
            st.session_state["itinerary"] = build_itinerary(acts, st.session_state["num_days"])
            st.session_state["step"] = 3
            st.rerun()


def step_itinerary() -> None:
    render_progress(3)

    city     = st.session_state["city"]
    num_days = st.session_state["num_days"]
    prefs    = st.session_state.get("prefs", [])
    itinerary = st.session_state["itinerary"]

    st.markdown(f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>',
                unsafe_allow_html=True)

    # Summary box
    pref_text = ", ".join(prefs) if prefs else "All activities"
    st.markdown(
        f'<div class="summary-box">'
        f'<strong>Destination:</strong> {city} &nbsp;|&nbsp; '
        f'<strong>Days:</strong> {num_days} &nbsp;|&nbsp; '
        f'<strong>Interests:</strong> {pref_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Timetable — lay days out in columns (max 3 per row)
    for row_start in range(0, num_days, 3):
        days_in_row = itinerary[row_start : row_start + 3]
        cols = st.columns(len(days_in_row))
        for col, day_plan in zip(cols, days_in_row):
            with col:
                col.markdown(
                    f'<div class="tt-header">Day {day_plan["day"]}</div>',
                    unsafe_allow_html=True,
                )
                for slot, activity in day_plan["slots"].items():
                    is_free = activity.startswith("Free time")
                    css = "tt-free" if is_free else SLOT_CLASSES[slot]
                    icon = "" if is_free else SLOT_ICONS[slot]
                    col.markdown(
                        f'<div class="tt-slot {css}">{icon} <strong>{slot}</strong><br>'
                        f'<span class="act-meta">{activity}</span></div>',
                        unsafe_allow_html=True,
                    )

    st.write("")
    col_back, col_restart = st.columns([1, 5])
    with col_back:
        if st.button("← Change preferences"):
            st.session_state["step"] = 2
            st.rerun()
    with col_restart:
        if st.button("Start over"):
            for key in ["city", "num_days", "prefs", "itinerary", "step"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown('<div class="footer">Swiss Vacation Planner · Built with Streamlit</div>',
                unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def run_app() -> None:
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    step = st.session_state["step"]
    if step == 1:
        step_destination()
    elif step == 2:
        step_preferences()
    elif step == 3:
        step_itinerary()
