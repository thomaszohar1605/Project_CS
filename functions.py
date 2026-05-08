import os
import random
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ml_rating import score_activities, get_top_activities, explain_score
from weather import get_weather

# ── Constants ─────────────────────────────────────────────────────────────────
FOLDER = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

CATEGORY_COLORS = {
    "Outdoor & Nature":        "#34d399",
    "Culture & History":       "#60a5fa",
    "Food & Drink":            "#f97316",
    "Nightlife & Entertainment": "#a78bfa",
    "Relaxation & Wellness":   "#f472b6",
    "Adventure & Sports":      "#fbbf24",
}

CATEGORY_EMOJI = {
    "Outdoor & Nature":        "🌿",
    "Culture & History":       "🏛️",
    "Food & Drink":            "🍽️",
    "Nightlife & Entertainment": "🎉",
    "Relaxation & Wellness":   "🧘",
    "Adventure & Sports":      "⚡",
}

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


# ── Data helpers ───────────────────────────────────────────────────────────────
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
    label = weather_label.lower()
    return any(w in label for w in bad_words)


def get_cities(df):
    return sorted(df["city"].dropna().unique().tolist())


def city_activities(df, city):
    return df[df["city"] == city].reset_index(drop=True)


def is_allowed_in_slot(row, slot):
    category = row.get("category", "")
    if category in CATEGORY_ALLOWED_SLOTS:
        return slot in CATEGORY_ALLOWED_SLOTS[category]
    return True


def get_best_slot(time_slot_value):
    if pd.isna(time_slot_value):
        return random.choice(SLOTS)
    parts = [p.strip() for p in str(time_slot_value).split("|")]
    for slot in SLOTS:
        if slot in parts:
            return slot
    return random.choice(SLOTS)


# ── Itinerary builder ─────────────────────────────────────────────────────────
def build_itinerary(activities: pd.DataFrame, num_days: int, forecast: list) -> list:
    """
    Build a day-by-day itinerary using ML-ranked activities.
    Activities are already sorted by ml_score (highest first).
    We still respect weather preferences and slot restrictions.
    """
    # Sort by ml_score so the best matches come first
    if "ml_score" in activities.columns:
        activities = activities.sort_values("ml_score", ascending=False)
    activities = activities.reset_index(drop=True)

    all_rows = activities.to_dict("records")

    # Bucket by preferred time slot (preserving ml_score order within each bucket)
    buckets = {"Morning": [], "Afternoon": [], "Evening": []}
    for row in all_rows:
        best_slot = get_best_slot(row.get("time_slot", ""))
        buckets[best_slot].append(row)

    already_used = set()
    itinerary = []

    for day_number in range(1, num_days + 1):
        day_index = day_number - 1
        if day_index < len(forecast) and is_bad_weather(forecast[day_index]["label"]):
            prefer_indoor = True
        else:
            prefer_indoor = False

        day_plan = {}

        for slot in SLOTS:
            chosen = None

            # Pass 1: right slot + weather match
            for row in buckets[slot]:
                name = row["activity_name"]
                setting = str(row.get("indoor_outdoor", "")).lower()
                if name in already_used:
                    continue
                if not is_allowed_in_slot(row, slot):
                    continue
                if prefer_indoor and setting in ("indoor", "both"):
                    chosen = row
                    break
                elif not prefer_indoor and setting in ("outdoor", "both"):
                    chosen = row
                    break

            # Pass 2: any slot + weather match
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        name = row["activity_name"]
                        setting = str(row.get("indoor_outdoor", "")).lower()
                        if name in already_used:
                            continue
                        if not is_allowed_in_slot(row, slot):
                            continue
                        if prefer_indoor and setting in ("indoor", "both"):
                            chosen = row
                            break
                        elif not prefer_indoor and setting in ("outdoor", "both"):
                            chosen = row
                            break
                    if chosen:
                        break

            # Pass 3: any unused activity allowed in this slot
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        name = row["activity_name"]
                        if name not in already_used and is_allowed_in_slot(row, slot):
                            chosen = row
                            break
                    if chosen:
                        break

            if chosen is not None:
                day_plan[slot] = {
                    "name":     chosen["activity_name"],
                    "category": chosen.get("category", ""),
                    "score":    round(float(chosen.get("ml_score", 3.0)), 1),
                }
                already_used.add(chosen["activity_name"])
            else:
                day_plan[slot] = {
                    "name":     "Free time — explore at your own pace",
                    "category": "",
                    "score":    0.0,
                }

        itinerary.append({"day": day_number, "slots": day_plan})

    return itinerary


# ── Progress bar ───────────────────────────────────────────────────────────────
def render_progress(current_step):
    steps = ["1 · Destination", "2 · Preferences", "3 · Your Itinerary"]
    cols = st.columns(3)
    for i in range(3):
        step_number = i + 1
        if step_number < current_step:
            css_class = "prog-step done"
        elif step_number == current_step:
            css_class = "prog-step current"
        else:
            css_class = "prog-step"
        cols[i].markdown(
            f'<div class="{css_class}">{steps[i]}</div>',
            unsafe_allow_html=True,
        )
    st.write("")


# ── Step 1 — Destination ──────────────────────────────────────────────────────
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


# ── Step 2 — Preferences with ML sliders ──────────────────────────────────────
def step_preferences():
    render_progress(2)
    st.markdown('<div class="step-heading">What do you enjoy?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-caption">'
        'Rate each type of activity from 1 (not interested) to 5 (love it). '
        'Our ML model will build your itinerary based on these scores.'
        '</div>',
        unsafe_allow_html=True,
    )

    prefs = {}
    cols = st.columns(2)
    for i, cat in enumerate(CATEGORIES):
        col = cols[i % 2]
        emoji = CATEGORY_EMOJI[cat]
        with col:
            st.markdown(
                f'<div style="font-weight:600; color:#1a3a5c; margin-top:0.8rem;">'
                f'{emoji} {cat}</div>',
                unsafe_allow_html=True,
            )
            prefs[cat] = st.slider(
                label=cat,
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                key=f"pref_{cat}",
                label_visibility="collapsed",
            )
            # Visual star feedback
            stars = "⭐" * prefs[cat] + "☆" * (5 - prefs[cat])
            st.markdown(
                f'<div style="font-size:1.1rem; color:#2e6da4; margin-bottom:0.3rem;">{stars}</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Build my itinerary →"):
            st.session_state["prefs"] = prefs

            df = load_activities()
            acts = city_activities(df, st.session_state["city"])
            if acts.empty:
                acts = df

            # ── ML: score every activity then pick the best pool ──
            acts_scored = score_activities(acts, prefs)
            # Keep all but sorted by ML score (itinerary builder uses top ones first)
            acts_scored = acts_scored.sort_values("ml_score", ascending=False).reset_index(drop=True)

            forecast = get_city_forecast(st.session_state["city"], st.session_state["num_days"])
            st.session_state["itinerary"] = build_itinerary(
                acts_scored, st.session_state["num_days"], forecast
            )
            st.session_state["step"] = 3
            st.rerun()


# ── Bar chart visualisation ────────────────────────────────────────────────────
def render_activity_chart(itinerary: list):
    """
    Draw a grouped bar chart: X-axis = days, one bar per category,
    height = number of activities of that category on that day.
    """
    # Count activities per day per category
    days = [f"Day {d['day']}" for d in itinerary]
    cat_counts = {cat: [] for cat in CATEGORIES}

    for day_plan in itinerary:
        day_cats = [
            day_plan["slots"][slot]["category"]
            for slot in SLOTS
            if day_plan["slots"][slot]["category"]
        ]
        for cat in CATEGORIES:
            cat_counts[cat].append(day_cats.count(cat))

    fig = go.Figure()
    for cat in CATEGORIES:
        counts = cat_counts[cat]
        if max(counts) == 0:
            continue  # skip unused categories
        fig.add_trace(go.Bar(
            name=f"{CATEGORY_EMOJI[cat]} {cat}",
            x=days,
            y=counts,
            marker_color=CATEGORY_COLORS[cat],
            text=[str(v) if v > 0 else "" for v in counts],
            textposition="inside",
            textfont=dict(color="white", size=12),
        ))

    fig.update_layout(
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a3a5c", family="Segoe UI"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#1a3a5c"),
        ),
        xaxis=dict(
            title="",
            gridcolor="#dce8f0",
            tickfont=dict(color="#1a3a5c"),
        ),
        yaxis=dict(
            title="Activities",
            gridcolor="#dce8f0",
            tickvals=[0, 1, 2, 3],
            tickfont=dict(color="#1a3a5c"),
            title_font=dict(color="#1a3a5c"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Step 3 — Itinerary ────────────────────────────────────────────────────────
def step_itinerary():
    render_progress(3)

    city      = st.session_state["city"]
    num_days  = st.session_state["num_days"]
    prefs     = st.session_state.get("prefs", {})
    itinerary = st.session_state["itinerary"]

    st.markdown(
        f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>',
        unsafe_allow_html=True,
    )

    # Summary box
    top_cats = sorted(prefs, key=prefs.get, reverse=True)[:3]
    top_str  = ", ".join(f"{CATEGORY_EMOJI[c]} {c}" for c in top_cats)
    st.markdown(
        f'<div class="summary-box">'
        f'<strong>Destination:</strong> {city} &nbsp;|&nbsp; '
        f'<strong>Days:</strong> {num_days} &nbsp;|&nbsp; '
        f'<strong>Top interests:</strong> {top_str}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Weather
    forecast = get_city_forecast(city, num_days)
    if forecast:
        st.markdown(
            '<div style="font-size:0.82rem; color:#4a7a9b; margin-bottom:0.4rem;">'
            '🌤️ <strong>Live weather forecast</strong> — real-time data for today and coming days'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Timetable ──────────────────────────────────────────────────────────────
    for row_start in range(0, num_days, 3):
        days_in_row = itinerary[row_start: row_start + 3]
        cols = st.columns(len(days_in_row))

        for i, day_plan in enumerate(days_in_row):
            with cols[i]:
                cols[i].markdown(
                    f'<div class="tt-header">Day {day_plan["day"]}</div>',
                    unsafe_allow_html=True,
                )

                day_index = day_plan["day"] - 1
                if day_index < len(forecast):
                    w = forecast[day_index]
                    cols[i].markdown(
                        f'<div style="font-size:0.82rem; color:#ffffff; background:#2e6da4; '
                        f'border-radius:0.4rem; padding:0.25rem 0.5rem; margin-bottom:0.4rem;">'
                        f'🌡️ {w["label"]} · {w["min"]}°/{w["max"]}°C · 🌧️ {w["rain"]} mm'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                for slot in SLOTS:
                    activity = day_plan["slots"][slot]
                    name     = activity["name"]
                    cat      = activity["category"]
                    score    = activity["score"]

                    if name.startswith("Free time"):
                        css   = "tt-free"
                        icon  = ""
                        badge = ""
                    else:
                        css   = SLOT_CLASSES[slot]
                        icon  = SLOT_ICONS[slot]
                        color = CATEGORY_COLORS.get(cat, "#2e6da4")
                        badge = (
                            f'<span style="background:{color}; color:#ffffff; '
                            f'font-size:0.7rem; border-radius:0.3rem; '
                            f'padding:0.1rem 0.35rem; margin-left:0.3rem;">'
                            f'{CATEGORY_EMOJI.get(cat,"")}</span>'
                        )

                    cols[i].markdown(
                        f'<div class="tt-slot {css}">'
                        f'{icon} <strong style="color:#1a3a5c;">{slot}</strong>{badge}<br>'
                        f'<span class="act-meta" style="color:#1a3a5c;">{name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.write("")

    # ── Activity distribution chart ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="step-heading" style="font-size:1.1rem;">📊 Activity breakdown</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">How your days are distributed across activity types.</div>',
        unsafe_allow_html=True,
    )
    render_activity_chart(itinerary)

    # ── ML preference summary ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="step-heading" style="font-size:1.1rem;">🤖 Your preference profile</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">'
        'These are the ratings you gave. Our KNN model used them to score and rank '
        'every activity before building your itinerary.'
        '</div>',
        unsafe_allow_html=True,
    )

    pref_cols = st.columns(3)
    for i, cat in enumerate(CATEGORIES):
        score = prefs.get(cat, 3)
        stars = "⭐" * score + "☆" * (5 - score)
        color = CATEGORY_COLORS.get(cat, "#2e6da4")
        pref_cols[i % 3].markdown(
            f'<div style="background:#e8f4fd; border-left:4px solid {color}; '
            f'border-radius:0.5rem; padding:0.5rem 0.8rem; margin-bottom:0.6rem;">'
            f'<div style="font-weight:700; color:#1a3a5c; font-size:0.88rem;">'
            f'{CATEGORY_EMOJI[cat]} {cat}</div>'
            f'<div style="font-size:1rem; color:#1a3a5c;">{stars}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Navigation ─────────────────────────────────────────────────────────────
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

    st.markdown(
        '<div class="footer">Swiss Vacation Planner · Built with Streamlit · '
        'Weather: Open-Meteo (real-time) · ML: scikit-learn KNN</div>',
        unsafe_allow_html=True,
    )


# ── Entry point ────────────────────────────────────────────────────────────────
def run_app():
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    step = st.session_state["step"]
    if step == 1:
        step_destination()
    elif step == 2:
        step_preferences()
    elif step == 3:
        step_itinerary()
