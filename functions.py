import os
import random
import pandas as pd
import streamlit as st
from ml_rating import build_itinerary_knn
from weather import get_weather

FOLDER = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

SLOTS = ["Morning", "Afternoon", "Evening"]

SLOT_CLASSES = {
    "Morning":   "tt-morning",
    "Afternoon": "tt-afternoon",
    "Evening":   "tt-evening",
}

SLOT_ICONS = {
    "Morning":   "🌅",
    "Afternoon": "☀️",
    "Evening":   "🌙",
}

CATEGORY_ALLOWED_SLOTS = {
    "Nightlife & Entertainment": ["Evening"],
    "Relaxation & Wellness":     ["Morning", "Afternoon"],
}


def is_allowed_in_slot(row, slot):
    category = row.get("category", "")
    if category in CATEGORY_ALLOWED_SLOTS:
        return slot in CATEGORY_ALLOWED_SLOTS[category]
    return True


@st.cache_data
def load_activities():
    file_path = os.path.join(FOLDER, "locations.csv")
    return pd.read_csv(file_path)


@st.cache_data(ttl=3600)
def get_city_forecast(city, num_days):
    df = load_activities()
    city_rows = df[df["city"] == city]
    if city_rows.empty:
        return []
    lat = float(city_rows.iloc[0]["lat"])
    lon = float(city_rows.iloc[0]["lon"])
    return get_weather(lat, lon, num_days)


def is_bad_weather(weather_label):
    bad_words = ["rain", "drizzle", "snow", "storm", "thunder"]
    return any(w in weather_label.lower() for w in bad_words)


def get_cities(df):
    return sorted(df["city"].dropna().unique().tolist())


def city_activities(df, city):
    return df[df["city"] == city].reset_index(drop=True)


def filter_by_preferences(activities, prefs):
    if not prefs:
        return activities
    filtered = activities[activities["category"].isin(prefs)]
    return filtered if not filtered.empty else activities


def get_best_slot(time_slot_value):
    if pd.isna(time_slot_value):
        return random.choice(SLOTS)
    parts = [p.strip() for p in str(time_slot_value).split("|")]
    for slot in SLOTS:
        if slot in parts:
            return slot
    return random.choice(SLOTS)


# ------------------------------------------------------------------
# Build itinerary — respects the order activities arrive in.
# When called from step_itinerary the DataFrame is already sorted
# by KNN rank, so the best-matching activities are tried first.
# The shuffle that used to be here has been REMOVED so KNN works.
# ------------------------------------------------------------------
def build_itinerary(activities, num_days, forecast):
    # Convert to list of dicts — ORDER IS PRESERVED (no shuffle)
    all_rows = activities.to_dict("records")

    # Sort into time-slot buckets, preserving KNN rank within each bucket
    buckets = {"Morning": [], "Afternoon": [], "Evening": []}
    for row in all_rows:
        best_slot = get_best_slot(row.get("time_slot", ""))
        buckets[best_slot].append(row)

    already_used = set()
    itinerary = []

    for day_number in range(1, num_days + 1):
        day_index = day_number - 1
        prefer_indoor = (
            day_index < len(forecast)
            and is_bad_weather(forecast[day_index]["label"])
        )

        day_plan = {}

        for slot in SLOTS:
            chosen = None

            # Pass 1: correct slot + weather match
            for row in buckets[slot]:
                if row["activity_name"] in already_used:
                    continue
                if not is_allowed_in_slot(row, slot):
                    continue
                setting = str(row.get("indoor_outdoor", "")).lower()
                if prefer_indoor and setting in ("indoor", "both"):
                    chosen = row; break
                elif not prefer_indoor and setting in ("outdoor", "both"):
                    chosen = row; break

            # Pass 2: any bucket + weather match
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        if row["activity_name"] in already_used:
                            continue
                        if not is_allowed_in_slot(row, slot):
                            continue
                        setting = str(row.get("indoor_outdoor", "")).lower()
                        if prefer_indoor and setting in ("indoor", "both"):
                            chosen = row; break
                        elif not prefer_indoor and setting in ("outdoor", "both"):
                            chosen = row; break
                    if chosen:
                        break

            # Pass 3: last resort — any unused allowed activity
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        if (row["activity_name"] not in already_used
                                and is_allowed_in_slot(row, slot)):
                            chosen = row; break
                    if chosen:
                        break

            if chosen:
                day_plan[slot] = chosen["activity_name"]
                already_used.add(chosen["activity_name"])
            else:
                day_plan[slot] = "Free time — explore at your own pace"

        itinerary.append({"day": day_number, "slots": day_plan})

    return itinerary


# ------------------------------------------------------------------
# Progress bar — 4 steps
# ------------------------------------------------------------------
def render_progress(current_step):
    steps = [
        "1 · Destination",
        "2 · Preferences",
        "3 · Rate Activities",
        "4 · Your Itinerary",
    ]
    cols = st.columns(4)
    for i, label in enumerate(steps):
        step_number = i + 1
        if step_number < current_step:
            css = "prog-step done"
        elif step_number == current_step:
            css = "prog-step current"
        else:
            css = "prog-step"
        cols[i].markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
    st.write("")


# ------------------------------------------------------------------
# STEP 1 — Destination
# ------------------------------------------------------------------
def step_destination():
    render_progress(1)
    st.markdown('<div class="step-heading">Where are you heading?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Pick a Swiss destination and how many days you have.</div>', unsafe_allow_html=True)

    df = load_activities()
    cities = get_cities(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", cities)
    with col2:
        num_days = st.selectbox("Number of days", [1, 2, 3, 4, 5, 6, 7], index=2)

    st.write("")
    if st.button("Next →"):
        st.session_state["city"] = city
        st.session_state["num_days"] = num_days
        st.session_state["step"] = 2
        st.rerun()


# ------------------------------------------------------------------
# STEP 2 — Preferences
# ------------------------------------------------------------------
def step_preferences():
    render_progress(2)
    st.markdown('<div class="step-heading">What do you enjoy?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Choose one or more activity types — or skip to include everything.</div>', unsafe_allow_html=True)

    selected_prefs = []
    cols = st.columns(3)
    for i, category in enumerate(CATEGORIES):
        with cols[i % 3]:
            if st.checkbox(category):
                selected_prefs.append(category)

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Next →"):
            st.session_state["prefs"] = selected_prefs

            df = load_activities()
            acts = city_activities(df, st.session_state["city"])
            if acts.empty:
                acts = df
            acts = filter_by_preferences(acts, selected_prefs)

            n_sample = min(5, len(acts))
            sample_df = acts.sample(n=n_sample).reset_index(drop=True)
            st.session_state["sample_activities"] = sample_df.to_dict("records")
            st.session_state["filtered_activities"] = acts

            st.session_state["step"] = 3
            st.rerun()


# ------------------------------------------------------------------
# STEP 3 — Rate 5 sample activities
# ------------------------------------------------------------------
def step_rating():
    render_progress(3)
    st.markdown('<div class="step-heading">Rate these activities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-caption">'
        'Rate each activity from 1 (not for me) to 5 (love it) — '
        'the ML model will use these to personalise your itinerary.'
        '</div>',
        unsafe_allow_html=True,
    )

    sample = st.session_state.get("sample_activities", [])

    ratings = {}
    for act in sample:
        name = act["activity_name"]
        cat  = act.get("category", "")
        st.markdown(f"**{name}** · *{cat}*")
        ratings[name] = st.slider(
            label=f"Rating for {name}",
            min_value=1, max_value=5, value=3, step=1,
            key=f"knn_rate_{name}",
            label_visibility="collapsed",
        )
        st.write("")

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 2
            st.rerun()
    with col_next:
        if st.button("Build my personalised itinerary →"):
            st.session_state["knn_ratings"] = [
                {"activity_name": name, "rating": rating}
                for name, rating in ratings.items()
            ]
            # Clear cached itinerary so step 4 rebuilds with new ratings
            st.session_state.pop("itinerary", None)
            st.session_state["step"] = 4
            st.rerun()


# ------------------------------------------------------------------
# STEP 4 — KNN-ranked itinerary with weather
# ------------------------------------------------------------------
def step_itinerary():
    render_progress(4)

    city     = st.session_state["city"]
    num_days = st.session_state["num_days"]
    prefs    = st.session_state.get("prefs", [])

    # Build itinerary once and cache it
    if "itinerary" not in st.session_state:

        acts_raw = st.session_state.get("filtered_activities", None)
        if acts_raw is None:
            df = load_activities()
            acts_raw = city_activities(df, city)
            acts_raw = filter_by_preferences(acts_raw, prefs)

        # ── Weather API call (shown to user via spinner) ──────────────
        with st.spinner("📡 Fetching weather forecast…"):
            forecast = get_city_forecast(city, num_days)

        # ── KNN ranking ───────────────────────────────────────────────
        knn_ratings = st.session_state.get("knn_ratings", [])
        with st.spinner("🤖 Running KNN model to personalise your itinerary…"):
            if knn_ratings:
                ranked_names = build_itinerary_knn(knn_ratings, acts_raw)
                name_order   = {name: i for i, name in enumerate(ranked_names)}
                acts_copy    = acts_raw.copy()
                acts_copy["_knn_rank"] = acts_copy["activity_name"].map(
                    lambda n: name_order.get(n, 9999)
                )
                acts_sorted = acts_copy.sort_values("_knn_rank").drop(columns=["_knn_rank"])
            else:
                acts_sorted = acts_raw

        # ── Build timetable (order is now KNN-ranked) ─────────────────
        st.session_state["itinerary"] = build_itinerary(acts_sorted, num_days, forecast)
        st.session_state["forecast"]  = forecast   # cache forecast too

    itinerary = st.session_state["itinerary"]
    forecast  = st.session_state.get("forecast", [])

    # ── Success banner ────────────────────────────────────────────────
    st.success("✅ Your personalised itinerary is ready — ranked by the KNN model based on your ratings!")

    # ── Page heading ──────────────────────────────────────────────────
    st.markdown(
        f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>',
        unsafe_allow_html=True,
    )

    pref_text = ", ".join(prefs) if prefs else "All activities"
    st.markdown(
        f'<div class="summary-box">'
        f'<strong>Destination:</strong> {city} &nbsp;|&nbsp; '
        f'<strong>Days:</strong> {num_days} &nbsp;|&nbsp; '
        f'<strong>Interests:</strong> {pref_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Timetable (no weather inside — shown separately below) ───────
    for row_start in range(0, num_days, 3):
        days_in_this_row = itinerary[row_start: row_start + 3]
        cols = st.columns(len(days_in_this_row))

        for i, day_plan in enumerate(days_in_this_row):
            with cols[i]:
                cols[i].markdown(
                    f'<div class="tt-header">Day {day_plan["day"]}</div>',
                    unsafe_allow_html=True,
                )

                for slot in SLOTS:
                    activity = day_plan["slots"][slot]
                    if activity.startswith("Free time"):
                        css, icon = "tt-free", ""
                    else:
                        css  = SLOT_CLASSES[slot]
                        icon = SLOT_ICONS[slot]

                    cols[i].markdown(
                        f'<div class="tt-slot {css}">'
                        f'{icon} <strong>{slot}</strong><br>'
                        f'<span class="act-meta">{activity}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Weather forecast (below the timetable) ────────────────────────
    if forecast:
        st.write("")
        st.markdown("**🌤 Weather forecast:**")
        wcols = st.columns(min(num_days, 7))
        for i, w in enumerate(forecast[:num_days]):
            with wcols[i]:
                st.markdown(
                    f"<div style='background:#e8f4fd;border-radius:0.6rem;"
                    f"padding:0.4rem 0.5rem;text-align:center;font-size:0.8rem;'>"
                    f"<strong>Day {i+1}</strong><br>"
                    f"{w['label']}<br>"
                    f"{w['min']}° / {w['max']}°C<br>"
                    f"🌧 {w['rain']} mm"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.write("")

    # ── Navigation ────────────────────────────────────────────────────
    col_back, col_restart = st.columns([1, 5])
    with col_back:
        if st.button("← Change preferences"):
            st.session_state["step"] = 2
            st.session_state.pop("itinerary", None)
            st.session_state.pop("forecast", None)
            st.rerun()
    with col_restart:
        if st.button("Start over"):
            for key in ["city", "num_days", "prefs", "itinerary", "forecast",
                        "step", "knn_ratings", "sample_activities",
                        "filtered_activities"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown(
        '<div class="footer">Swiss Vacation Planner · Built with Streamlit</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def run_app():
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    step = st.session_state["step"]
    if step == 1:
        step_destination()
    elif step == 2:
        step_preferences()
    elif step == 3:
        step_rating()
    elif step == 4:
        step_itinerary()