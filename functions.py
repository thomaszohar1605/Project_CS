from __future__ import annotations

import os
import random
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Constants ──────────────────────────────────────────────────────────────────

# Loaded dynamically from the CSV so the list is always in sync
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

SLOTS = ["Morning", "Afternoon", "Evening"]
SLOT_CLASSES = {"Morning": "tt-morning", "Afternoon": "tt-afternoon", "Evening": "tt-evening"}
SLOT_ICONS   = {"Morning": "🌅", "Afternoon": "☀️", "Evening": "🌙"}

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_activities() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_HERE, "locations.csv"))


def get_cities(df: pd.DataFrame) -> list[str]:
    """Return sorted list of cities from the CSV."""
    return sorted(df["city"].dropna().unique().tolist())


def city_activities(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return all activities for the chosen city."""
    return df[df["city"] == city].reset_index(drop=True)


def filter_by_preferences(activities: pd.DataFrame, prefs: list[str]) -> pd.DataFrame:
    """Keep only rows whose category matches one of the selected preferences."""
    if not prefs:
        return activities
    filtered = activities[activities["category"].isin(prefs)]
    return filtered if not filtered.empty else activities   # fall back if nothing matches


def _best_slot(time_slot_str: str) -> str:
    """Pick the first slot listed in the activity's time_slot field."""
    if pd.isna(time_slot_str):
        return random.choice(SLOTS)
    parts = [s.strip() for s in str(time_slot_str).split("|")]
    for slot in SLOTS:
        if slot in parts:
            return slot
    return random.choice(SLOTS)


def build_itinerary(activities: pd.DataFrame, num_days: int) -> list[dict]:
    """Assign activities to morning/afternoon/evening slots across num_days.

    Activities are slotted according to their preferred time_slot from the CSV.
    """
    rows = activities.sample(frac=1).to_dict("records")   # shuffle
    # Bucket by preferred slot
    buckets: dict[str, list[str]] = {s: [] for s in SLOTS}
    for row in rows:
        slot = _best_slot(row.get("time_slot", ""))
        buckets[slot].append(row["activity_name"])

    itinerary = []
    for day in range(1, num_days + 1):
        day_plan: dict[str, str] = {}
        for slot in SLOTS:
            if buckets[slot]:
                day_plan[slot] = buckets[slot].pop()
            else:
                # Try any remaining slot before falling back to free time
                remaining = [a for s in SLOTS for a in buckets[s]]
                if remaining:
                    picked = remaining[0]
                    for s in SLOTS:
                        if picked in buckets[s]:
                            buckets[s].remove(picked)
                            break
                    day_plan[slot] = picked
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

    df = load_activities()
    cities = get_cities(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", cities, index=0)
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
    for i, category in enumerate(CATEGORIES):
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
