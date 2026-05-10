"""
functions.py  —  Swiss Vacation Planner
========================================
3-step UI flow:
  Step 1 → Destination & days
  Step 2 → Rate 6 categories with sliders (1–5)
  Step 3 → Final itinerary + activity chart
"""

from __future__ import annotations
import os, random
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_rating import get_knn_ranked_activities, CATEGORIES
from weather import get_weather

# ── Constants ──────────────────────────────────────────────────────────
FOLDER = os.path.dirname(os.path.abspath(__file__))

CATEGORY_COLORS = {
    "Outdoor & Nature":          "#34d399",
    "Culture & History":         "#60a5fa",
    "Food & Drink":              "#f97316",
    "Nightlife & Entertainment": "#a78bfa",
    "Relaxation & Wellness":     "#f472b6",
    "Adventure & Sports":        "#fbbf24",
}

CATEGORY_EMOJI = {
    "Outdoor & Nature":          "🌿",
    "Culture & History":         "🏛️",
    "Food & Drink":              "🍽️",
    "Nightlife & Entertainment": "🎉",
    "Relaxation & Wellness":     "🧘",
    "Adventure & Sports":        "⚡",
}

SLOTS = ["Morning", "Afternoon", "Evening"]

SLOT_CSS = {
    "Morning":   "tt-morning",
    "Afternoon": "tt-afternoon",
    "Evening":   "tt-evening",
}
SLOT_ICON = {
    "Morning":   "🌅",
    "Afternoon": "☀️",
    "Evening":   "🌙",
}

# Nightlife only in Evening; Wellness not in Evening
SLOT_RESTRICTIONS = {
    "Nightlife & Entertainment": ["Evening"],
    "Relaxation & Wellness":     ["Morning", "Afternoon"],
}


# ── Data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_activities() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FOLDER, "locations.csv"))


@st.cache_data(ttl=3600)
def get_city_forecast(city: str, num_days: int) -> list:
    df = load_activities()
    rows = df[df["city"] == city]
    if rows.empty:
        return []
    lat = float(rows.iloc[0]["lat"])
    lon = float(rows.iloc[0]["lon"])
    return get_weather(lat, lon, num_days)


# ── Helpers ────────────────────────────────────────────────────────────

def is_bad_weather(label: str) -> bool:
    return any(w in label.lower() for w in
               ["rain", "drizzle", "snow", "storm", "thunder"])


def is_allowed_in_slot(row, slot: str) -> bool:
    cat = row.get("category", "")
    allowed = SLOT_RESTRICTIONS.get(cat)
    return (slot in allowed) if allowed else True


def get_best_slot(time_slot_value) -> str:
    if pd.isna(time_slot_value):
        return random.choice(SLOTS)
    parts = [p.strip() for p in str(time_slot_value).split("|")]
    for s in SLOTS:
        if s in parts:
            return s
    return random.choice(SLOTS)


# ── Itinerary builder ──────────────────────────────────────────────────

def build_itinerary(ranked_df: pd.DataFrame,
                    num_days: int,
                    forecast: list) -> list:
    """
    Build a day-by-day itinerary from ML-ranked activities.
    Activities higher in ranked_df (better knn_score) are placed first.
    """
    all_rows = ranked_df.to_dict("records")   # already sorted best→worst

    # Bucket by preferred time slot (preserving ML rank within each bucket)
    buckets: dict[str, list] = {s: [] for s in SLOTS}
    for row in all_rows:
        buckets[get_best_slot(row.get("time_slot", ""))].append(row)

    used: set[str] = set()
    itinerary = []

    for day_num in range(1, num_days + 1):
        day_idx = day_num - 1
        prefer_indoor = (day_idx < len(forecast) and
                         is_bad_weather(forecast[day_idx]["label"]))

        day_plan: dict[str, dict] = {}

        for slot in SLOTS:
            chosen = None

            # Pass 1: correct slot + weather match
            for row in buckets[slot]:
                if row["activity_name"] in used:
                    continue
                if not is_allowed_in_slot(row, slot):
                    continue
                setting = str(row.get("indoor_outdoor", "")).lower()
                if prefer_indoor and setting in ("indoor", "both"):
                    chosen = row; break
                elif not prefer_indoor and setting in ("outdoor", "both"):
                    chosen = row; break

            # Pass 2: any slot + weather match
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        if row["activity_name"] in used:
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

            # Pass 3: any unused activity allowed in this slot
            if chosen is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        if (row["activity_name"] not in used and
                                is_allowed_in_slot(row, slot)):
                            chosen = row; break
                    if chosen:
                        break

            if chosen:
                day_plan[slot] = {
                    "name":      chosen["activity_name"],
                    "category":  chosen.get("category", ""),
                    "knn_score": round(float(chosen.get("knn_score", 0.5)), 2),
                }
                used.add(chosen["activity_name"])
            else:
                day_plan[slot] = {
                    "name":      "Free time — explore at your own pace",
                    "category":  "",
                    "knn_score": 0.0,
                }

        itinerary.append({"day": day_num, "slots": day_plan})

    return itinerary


# ── Progress bar ───────────────────────────────────────────────────────

def render_progress(current: int) -> None:
    labels = [
        "1 · Destination",
        "2 · Preferences",
        "3 · Your Itinerary",
    ]
    cols = st.columns(3)
    for i, label in enumerate(labels):
        step = i + 1
        if step < current:
            css = "prog-step done"
        elif step == current:
            css = "prog-step current"
        else:
            css = "prog-step"
        cols[i].markdown(f'<div class="{css}">{label}</div>',
                         unsafe_allow_html=True)
    st.write("")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Destination & Days
# ══════════════════════════════════════════════════════════════════════

def step_destination() -> None:
    render_progress(1)
    st.markdown('<div class="step-heading">Where are you heading?</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Pick a Swiss destination and how many days you have.</div>',
                unsafe_allow_html=True)

    df     = load_activities()
    cities = sorted(df["city"].dropna().unique().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", cities)
    with col2:
        num_days = st.selectbox("Number of days",
                                [1, 2, 3, 4, 5, 6, 7], index=2)

    st.write("")
    if st.button("Next →"):
        st.session_state["city"]     = city
        st.session_state["num_days"] = num_days
        st.session_state["step"]     = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Category preference sliders
# ══════════════════════════════════════════════════════════════════════

def step_preferences() -> None:
    render_progress(2)
    st.markdown('<div class="step-heading">How much do you enjoy each type of activity?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="step-caption">'
        'Rate each category from 1 (not my thing) to 5 (love it). '
        'Our ML model will use these ratings to build a personalised itinerary.'
        '</div>',
        unsafe_allow_html=True,
    )

    prefs: dict[str, int] = {}
    col_a, col_b = st.columns(2)

    for i, cat in enumerate(CATEGORIES):
        col = col_a if i % 2 == 0 else col_b
        emoji = CATEGORY_EMOJI[cat]
        color = CATEGORY_COLORS[cat]

        with col:
            # Category label with coloured left-border card
            st.markdown(
                f'<div style="border-left:4px solid {color}; '
                f'background:#e8f4fd; border-radius:0.5rem; '
                f'padding:0.45rem 0.75rem; margin-top:0.9rem; '
                f'font-weight:700; color:#1a3a5c; font-size:0.97rem;">'
                f'{emoji} {cat}</div>',
                unsafe_allow_html=True,
            )
            rating = st.slider(
                label=cat,
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                key=f"pref_{cat}",
                label_visibility="collapsed",
            )
            # Star feedback
            stars = "⭐" * rating + "☆" * (5 - rating)
            label_map = {1: "Not interested", 2: "Slightly interested",
                         3: "Neutral", 4: "Interested", 5: "Love it!"}
            st.markdown(
                f'<div style="font-size:1rem; color:#2e6da4; '
                f'margin-bottom:0.2rem;">{stars} '
                f'<span style="font-size:0.8rem; color:#4a7a9b;">'
                f'{label_map[rating]}</span></div>',
                unsafe_allow_html=True,
            )
            prefs[cat] = rating

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("🏔️ Build my itinerary →"):
            st.session_state["prefs"] = prefs

            # Run ML immediately — no intermediate step
            df   = load_activities()
            acts = df[df["city"] == st.session_state["city"]].reset_index(drop=True)
            if acts.empty:
                acts = df

            ranked    = get_knn_ranked_activities(acts, prefs)
            forecast  = get_city_forecast(st.session_state["city"],
                                          st.session_state["num_days"])
            itinerary = build_itinerary(ranked,
                                        st.session_state["num_days"],
                                        forecast)

            st.session_state["ranked"]    = ranked
            st.session_state["itinerary"] = itinerary
            st.session_state["step"]      = 3
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Final itinerary + chart
# ══════════════════════════════════════════════════════════════════════

def render_activity_chart(itinerary: list) -> None:
    """Stacked bar chart: days × category counts."""
    days_labels = [f"Day {d['day']}" for d in itinerary]
    cat_counts  = {cat: [] for cat in CATEGORIES}

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
            continue
        fig.add_trace(go.Bar(
            name=f"{CATEGORY_EMOJI[cat]} {cat}",
            x=days_labels,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a3a5c")),
        xaxis=dict(title="", gridcolor="#dce8f0",
                   tickfont=dict(color="#1a3a5c")),
        yaxis=dict(title="Activities", gridcolor="#dce8f0",
                   tickvals=[0, 1, 2, 3],
                   tickfont=dict(color="#1a3a5c"),
                   title_font=dict(color="#1a3a5c")),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def step_itinerary() -> None:
    render_progress(3)

    city      = st.session_state["city"]
    num_days  = st.session_state["num_days"]
    prefs     = st.session_state.get("prefs", {})
    itinerary = st.session_state["itinerary"]

    st.markdown(
        f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>',
        unsafe_allow_html=True,
    )

    # Top interests summary
    top_cats = sorted(prefs, key=prefs.get, reverse=True)[:3]
    top_str  = " · ".join(
        f'{CATEGORY_EMOJI.get(c, "")} {c}' for c in top_cats)
    st.markdown(
        f'<div class="summary-box">'
        f'📍 <strong>{city}</strong> &nbsp;|&nbsp; '
        f'🗓️ <strong>{num_days} day{"s" if num_days>1 else ""}</strong> &nbsp;|&nbsp; '
        f'❤️ <strong>{top_str}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Weather note
    forecast = get_city_forecast(city, num_days)
    if forecast:
        st.markdown(
            '<div style="font-size:0.82rem; color:#4a7a9b; margin-bottom:0.6rem;">'
            '🌤️ <strong>Live weather forecast</strong> — '
            'real-time data from Open-Meteo for today and the coming days'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Timetable ──────────────────────────────────────────────────────
    for row_start in range(0, num_days, 3):
        chunk = itinerary[row_start: row_start + 3]
        cols  = st.columns(len(chunk))

        for i, day_plan in enumerate(chunk):
            with cols[i]:
                cols[i].markdown(
                    f'<div class="tt-header">Day {day_plan["day"]}</div>',
                    unsafe_allow_html=True,
                )

                # Weather card for this day
                day_idx = day_plan["day"] - 1
                if day_idx < len(forecast):
                    w = forecast[day_idx]
                    cols[i].markdown(
                        f'<div style="font-size:0.8rem; color:#ffffff; '
                        f'background:#2e6da4; border-radius:0.4rem; '
                        f'padding:0.25rem 0.5rem; margin-bottom:0.4rem;">'
                        f'🌡️ {w["label"]} · {w["min"]}°/{w["max"]}°C '
                        f'· 🌧️ {w["rain"]} mm</div>',
                        unsafe_allow_html=True,
                    )

                # Each time slot
                for slot in SLOTS:
                    act  = day_plan["slots"][slot]
                    name = act["name"]
                    cat  = act["category"]
                    score = act["knn_score"]

                    if name.startswith("Free time"):
                        css   = "tt-free"
                        icon  = ""
                        badge = ""
                    else:
                        css   = SLOT_CSS[slot]
                        icon  = SLOT_ICON[slot]
                        color = CATEGORY_COLORS.get(cat, "#2e6da4")
                        badge = (
                            f'<span style="background:{color}; color:#fff; '
                            f'font-size:0.68rem; border-radius:0.3rem; '
                            f'padding:0.05rem 0.3rem; margin-left:0.25rem; '
                            f'vertical-align:middle;">'
                            f'{CATEGORY_EMOJI.get(cat, "")}</span>'
                        )

                    cols[i].markdown(
                        f'<div class="tt-slot {css}">'
                        f'{icon} <strong style="color:#1a3a5c;">{slot}</strong>'
                        f'{badge}<br>'
                        f'<span class="act-meta" style="color:#1a3a5c;">{name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Activity breakdown chart ───────────────────────────────────────
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

    # ── Navigation ─────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_restart = st.columns([1, 5])
    with col_back:
        if st.button("← Change preferences"):
            st.session_state["step"] = 2
            st.rerun()
    with col_restart:
        if st.button("Start over"):
            for k in ["city", "num_days", "prefs",
                      "ranked", "itinerary", "step"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown(
        '<div class="footer">'
        'Swiss Vacation Planner · Streamlit · '
        'Weather: Open-Meteo (real-time) · ML: Cosine-similarity KNN'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

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
