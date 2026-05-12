
#Step 1 -> Destination, days & season
#Step 2 -> Rate 6 categories 1 to 5
#Step 3 -> Final itinerary + pdf


from __future__ import annotations

#import the relevant libraries 

import os, random, datetime, math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

#Machine learning with KNN 

from ml_rating import get_knn_ranked_activities, CATEGORIES

SEASONS = ["spring", "summer", "fall", "winter"]
from weather import get_weather

#Variables with the ML (category, season)

#Loader of location.csv
FOLDER = os.path.dirname(os.path.abspath(__file__))

# Colour for the activity chart
CATEGORY_COLORS = {
    "Outdoor & Nature":          "#34d399",
    "Culture & History":         "#60a5fa",
    "Food & Drink":              "#f97316",
    "Nightlife & Entertainment": "#a78bfa",
    "Relaxation & Wellness":     "#f472b6",
    "Adventure & Sports":        "#fbbf24",
}

SEASON_LABELS = {
    "spring": "Spring",
    "summer": "Summer",
    "fall":   "Fall",
    "winter": "Winter",
}

# Time slot for the planning 
SLOTS = ["Morning", "Afternoon", "Evening"]

# CSS (Cascading Style Sheets):controls background colour
SLOT_CSS = {
    "Morning":   "tt-morning",
    "Afternoon": "tt-afternoon",
    "Evening":   "tt-evening",
}


# Restriction made, becasue nightlife & entertainment can only happen in the evening 
SLOT_RESTRICTIONS = {
    "Nightlife & Entertainment": ["Evening"],
}

# Activities from categories rated at or below this value are removed for the KNN ranking 
LOW_RATING_THRESHOLD = 2

# Weather data without the api

# Typical Swiss weather conditions 
# Used when the user's chosen season does not match the current real-world season 
SEASONAL_WEATHER = {
    "spring": [
        {"label": "Partly Cloudy", "min": 8,  "max": 16, "rain": 2.1},
        {"label": "Sunny",         "min": 10, "max": 18, "rain": 0.4},
        {"label": "Light Rain",    "min": 7,  "max": 13, "rain": 5.3},
        {"label": "Partly Cloudy", "min": 9,  "max": 15, "rain": 1.8},
        {"label": "Sunny",         "min": 11, "max": 19, "rain": 0.2},
        {"label": "Cloudy",        "min": 8,  "max": 14, "rain": 3.0},
        {"label": "Sunny",         "min": 12, "max": 20, "rain": 0.3},
    ],
    "summer": [
        {"label": "Sunny",         "min": 18, "max": 27, "rain": 0.2},
        {"label": "Partly Cloudy", "min": 17, "max": 25, "rain": 1.5},
        {"label": "Thunderstorm",  "min": 16, "max": 23, "rain": 8.4},
        {"label": "Sunny",         "min": 19, "max": 28, "rain": 0.1},
        {"label": "Partly Cloudy", "min": 18, "max": 26, "rain": 0.8},
        {"label": "Sunny",         "min": 20, "max": 29, "rain": 0.0},
        {"label": "Light Rain",    "min": 15, "max": 22, "rain": 4.2},
    ],
    "fall": [
        {"label": "Cloudy",        "min": 6,  "max": 14, "rain": 4.1},
        {"label": "Partly Cloudy", "min": 7,  "max": 15, "rain": 2.3},
        {"label": "Light Rain",    "min": 5,  "max": 12, "rain": 6.0},
        {"label": "Sunny",         "min": 8,  "max": 16, "rain": 0.5},
        {"label": "Cloudy",        "min": 4,  "max": 11, "rain": 3.7},
        {"label": "Light Rain",    "min": 4,  "max": 10, "rain": 5.5},
        {"label": "Partly Cloudy", "min": 6,  "max": 13, "rain": 2.0},
    ],
    "winter": [
        {"label": "Snow",          "min": -3, "max": 2,  "rain": 3.2},
        {"label": "Cloudy",        "min": -1, "max": 4,  "rain": 1.5},
        {"label": "Sunny",         "min": -2, "max": 5,  "rain": 0.1},
        {"label": "Snow",          "min": -4, "max": 1,  "rain": 4.0},
        {"label": "Partly Cloudy", "min": -1, "max": 4,  "rain": 0.8},
        {"label": "Cloudy",        "min": -2, "max": 3,  "rain": 2.1},
        {"label": "Sunny",         "min": -1, "max": 6,  "rain": 0.0},
    ],
}


def get_current_season() -> str:
    """Return the real-world season based on today's month."""
    month = datetime.date.today().month
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"

# - If the chosen season matches today's real season, fetch live data from
      Open-Meteo and label it 'Live forecast'.
    - Otherwise return typical hardcoded Swiss conditions for that season and
      label it 'Typical conditions for <season>'.

def get_forecast_for_season(city: str, season: str, num_days: int) -> tuple[list, str]:
    current = get_current_season()
    if season.lower() == current:
        # Seasons match — live forecast is actually relevant
        forecast = get_city_forecast(city, num_days)
        label    = "Live forecast"
    else:
        # Seasons differ — use typical seasonal data instead
        pool     = SEASONAL_WEATHER.get(season.lower(), SEASONAL_WEATHER["summer"])
        forecast = (pool * 3)[:num_days]   # repeat the 7-entry list if needed
        label    = f"Typical conditions for {season.capitalize()}"
    return forecast, label


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_activities() -> pd.DataFrame:
    # Load the full activity dataset from CSV; cached so it is only read once
    return pd.read_csv(os.path.join(FOLDER, "locations.csv"))


@st.cache_data(ttl=3600)
def get_city_forecast(city: str, num_days: int) -> list:
    # Look up the city coordinates then fetch a live weather forecast.
    # Cached for 1 hour to avoid redundant API calls.
    df   = load_activities()
    rows = df[df["city"] == city]
    if rows.empty:
        return []
    lat = float(rows.iloc[0]["lat"])
    lon = float(rows.iloc[0]["lon"])
    return get_weather(lat, lon, num_days)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_bad_weather(label: str) -> bool:
    # Return True if the weather label indicates conditions unsuitable for outdoor activities
    return any(w in label.lower() for w in
               ["rain", "drizzle", "snow", "storm", "thunder"])


def is_allowed_in_slot(row, slot: str) -> bool:
    # Check whether an activity is permitted in the requested time slot.
    # Uses SLOT_RESTRICTIONS; categories not listed are allowed in all slots.
    cat     = row.get("category", "")
    allowed = SLOT_RESTRICTIONS.get(cat)
    return (slot in allowed) if allowed else True


def get_best_slot(time_slot_value) -> str:
    # Determine the preferred time slot from the CSV "time_slot" value.
    # Falls back to a random slot if the value is missing or unrecognised.
    if pd.isna(time_slot_value):
        return random.choice(SLOTS)
    parts = [p.strip() for p in str(time_slot_value).split("|")]
    for s in SLOTS:
        if s in parts:
            return s
    return random.choice(SLOTS)


def filter_by_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Hard pre-filter: keep only activities whose 'seasons' column contains
    the chosen season keyword.

    The CSV stores seasons as pipe-separated strings e.g. "spring|summer|fall".
    Activities with no seasons value are kept as year-round.
    """
    season_clean = season.strip().lower()

    def _has_season(val) -> bool:
        if pd.isna(val) or str(val).strip() == "":
            return True   # no season listed -> treat as year-round
        return season_clean in str(val).lower()

    return df[df["seasons"].apply(_has_season)].reset_index(drop=True)


# ── Post-filter ───────────────────────────────────────────────────────────────

def apply_preference_filter(ranked: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    """
    Remove activities whose category was rated at or below LOW_RATING_THRESHOLD.

    This hard filter runs after KNN ranking to guarantee that disliked
    categories never appear in the itinerary regardless of KNN score.
    """
    for cat in CATEGORIES:
        if prefs.get(cat, 3) <= LOW_RATING_THRESHOLD:
            ranked = ranked[ranked["category"] != cat]
    return ranked.reset_index(drop=True)


# ── Itinerary builder ─────────────────────────────────────────────────────────

def build_itinerary(ranked_df: pd.DataFrame,
                    num_days: int,
                    forecast: list) -> list:
    """
    Build a day-by-day itinerary from ML-ranked activities.
    Activities higher in ranked_df (better knn_score) are placed first.
    """
    all_rows = ranked_df.to_dict("records")   # sorted best to worst by KNN score

    # Bucket activities by preferred time slot, preserving ML rank order
    buckets: dict[str, list] = {s: [] for s in SLOTS}
    for row in all_rows:
        buckets[get_best_slot(row.get("time_slot", ""))].append(row)

    used: set[str] = set()   # tracks activity names already placed
    itinerary = []

    for day_num in range(1, num_days + 1):
        day_idx       = day_num - 1
        # Prefer indoor activities on bad-weather days
        prefer_indoor = (day_idx < len(forecast) and
                         is_bad_weather(forecast[day_idx]["label"]))

        day_plan: dict[str, dict] = {}

        for slot in SLOTS:
            chosen = None

            # Pass 1: correct slot bucket + weather preference match
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

            # Pass 2: any bucket + weather preference match
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

            # Pass 3: any unused activity allowed in this slot (ignore weather)
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
                # No suitable activity found — fill with free time
                day_plan[slot] = {
                    "name":      "Free time — explore at your own pace",
                    "category":  "",
                    "knn_score": 0.0,
                }

        itinerary.append({"day": day_num, "slots": day_plan})

    return itinerary


# ── Progress bar ──────────────────────────────────────────────────────────────

def render_progress(current: int) -> None:
    # Render the 3-step progress bar at the top of each page.
    # Completed steps: black. Current step: Swiss red. Upcoming: light grey.
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Destination, Days & Season
# ══════════════════════════════════════════════════════════════════════════════

def step_destination() -> None:
    render_progress(1)
    st.markdown('<div class="step-heading">Where are you heading?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="step-caption">'
        'Pick a Swiss destination, how many days you have, and which season you are travelling in.'
        '</div>',
        unsafe_allow_html=True,
    )

    df     = load_activities()
    cities = sorted(df["city"].dropna().unique().tolist())

    # Row 1: destination and number of days
    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", cities)
    with col2:
        num_days = st.selectbox("Number of days", [1, 2, 3, 4, 5, 6, 7], index=2)

    # Season selector
    st.markdown(
        '<div style="font-weight:700; color:#1a1a1a; font-size:0.97rem; '
        'margin-top:1rem; margin-bottom:0.4rem;">When are you travelling?</div>',
        unsafe_allow_html=True,
    )
    current_season = st.session_state.get("season_choice", "summer")
    season_cols = st.columns(4)
    for i, season in enumerate(SEASONS):
        is_selected = (season == current_season)
        with season_cols[i]:
            if st.button(SEASON_LABELS[season], key=f"season_btn_{season}"):
                st.session_state["season_choice"] = season
                st.rerun()
            btn_color = "#D52B1E" if is_selected else "#f0f0f0"
            txt_color = "#ffffff" if is_selected else "#1a1a1a"
            indicator = "Selected" if is_selected else ""
            st.markdown(
                f'<div style="background:{btn_color}; color:{txt_color}; '
                f'border-radius:0.5rem; text-align:center; padding:0.25rem 0; '
                f'font-size:0.78rem; font-weight:600; margin-top:-0.5rem;">'
                f'{indicator}</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    season_choice = st.session_state.get("season_choice", "summer")
    city_acts   = df[df["city"] == city]
    season_acts = filter_by_season(city_acts, season_choice)
    st.markdown(
        f'<div class="summary-box">'
        f'<strong>{city}</strong> &nbsp;|&nbsp; '
        f'{SEASON_LABELS[season_choice]} &nbsp;|&nbsp; '
        f'<strong>{len(season_acts)}</strong> activities available this season'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Next →"):
        st.session_state["city"]     = city
        st.session_state["num_days"] = num_days
        st.session_state["season"]   = season_choice
        st.session_state["step"]     = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Category preference sliders
# ══════════════════════════════════════════════════════════════════════════════

def step_preferences() -> None:
    render_progress(2)

    season = st.session_state.get("season", "summer")
    st.markdown(
        '<div class="step-heading">How much do you enjoy each type of activity?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="step-caption">'
        f'Rating for your {SEASON_LABELS[season]} trip to '
        f'<strong>{st.session_state.get("city", "")}</strong>. '
        f'Rate each category from 1 (not my thing) to 5 (love it).'
        f'</div>',
        unsafe_allow_html=True,
    )

    prefs: dict[str, int] = {}
    col_a, col_b = st.columns(2)

    for i, cat in enumerate(CATEGORIES):
        col = col_a if i % 2 == 0 else col_b

        with col:
            st.markdown(
                f'<div style="font-weight:700; color:#1a1a1a; '
                f'font-size:0.97rem; margin-top:0.9rem;">{cat}</div>',
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
            label_map = {1: "Not interested", 2: "Slightly interested",
                         3: "Neutral", 4: "Interested", 5: "Love it!"}
            st.markdown(
                f'<div style="font-size:0.85rem; color:#555555; '
                f'margin-bottom:0.2rem;">{rating} — {label_map[rating]}</div>',
                unsafe_allow_html=True,
            )
            prefs[cat] = rating

    st.write("")
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="pref_back"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Build my itinerary →"):
            st.session_state["prefs"] = prefs

            df     = load_activities()
            season = st.session_state.get("season", "summer")

            acts = df[df["city"] == st.session_state["city"]].reset_index(drop=True)
            if acts.empty:
                acts = df

            acts = filter_by_season(acts, season)
            if acts.empty:
                acts = df[df["city"] == st.session_state["city"]].reset_index(drop=True)

            try:
                ranked = get_knn_ranked_activities(acts, prefs, season=season)
            except TypeError:
                ranked = get_knn_ranked_activities(acts, prefs)

            ranked    = apply_preference_filter(ranked, prefs)
            forecast  = get_city_forecast(st.session_state["city"], st.session_state["num_days"])
            itinerary = build_itinerary(ranked, st.session_state["num_days"], forecast)

            st.session_state["ranked"]    = ranked
            st.session_state["itinerary"] = itinerary
            st.session_state["step"]      = 3
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Final itinerary + chart
# ══════════════════════════════════════════════════════════════════════════════

def render_activity_chart(itinerary: list) -> None:
    """Stacked bar chart showing activity category distribution across days."""
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
            name=cat,
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


def generate_itinerary_pdf(
    itinerary: list,
    city: str,
    num_days: int,
    season: str,
    forecast: list,
    df: pd.DataFrame,
) -> bytes:
    """
    Build a one-page landscape PDF summary of the personalised itinerary.

    Layout: Swiss-red header, then days arranged in up to 3 columns.
    Each day block shows the weather strip, then Morning / Afternoon / Evening
    with activity name and a truncated description from locations.csv.

    Returns raw PDF bytes ready for st.download_button.
    """
    from fpdf import FPDF

    # ── Description lookup ────────────────────────────────────────────────────
    desc_lookup: dict[str, str] = {}
    if "description" in df.columns:
        for _, row in df.iterrows():
            n = str(row.get("activity_name", "")).strip()
            d = str(row.get("description", "")).strip()
            if n:
                desc_lookup[n] = d

    # ── Colours ───────────────────────────────────────────────────────────────
    RED        = (213, 43, 30)
    WHITE      = (255, 255, 255)
    DARK       = (26, 26, 26)
    GREY       = (110, 110, 110)
    LIGHT_RED  = (253, 232, 230)
    LIGHT_GREY = (246, 246, 246)

    # ── Page setup (landscape A4) ─────────────────────────────────────────────
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)   # we control layout manually
    pdf.add_page()

    PAGE_W, PAGE_H = 297, 210
    MARGIN    = 10
    HEADER_H  = 20
    FOOTER_H  = 8
    COL_GAP   = 4

    # ── Header banner ─────────────────────────────────────────────────────────
    pdf.set_fill_color(*RED)
    pdf.rect(0, 0, PAGE_W, HEADER_H, style="F")

    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(MARGIN, 3)
    pdf.cell(160, 8, "Swiss Vacation Planner", ln=False)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(MARGIN, 12)
    pdf.cell(
        0, 6,
        f"{city}   |   {season.capitalize()}   |   "
        f"{num_days} day{'s' if num_days > 1 else ''}",
        ln=False,
    )

    # ── Column grid dimensions ────────────────────────────────────────────────
    content_top = HEADER_H + 3
    content_h   = PAGE_H - content_top - FOOTER_H - 2

    n_cols       = 3 if num_days > 2 else (2 if num_days == 2 else 1)
    days_per_col = math.ceil(num_days / n_cols)
    col_w        = (PAGE_W - 2 * MARGIN - (n_cols - 1) * COL_GAP) / n_cols
    day_h        = content_h / days_per_col

    # Fixed heights within each day block
    DAY_HDR_H = 5.5    # red day label bar
    WEATHER_H = 4.0    # light-red weather strip
    slots_h   = day_h - DAY_HDR_H - WEATHER_H - 1
    SLOT_H    = slots_h / 3

    # Characters that fit on one line per column (approx: 1 char ≈ 1.9 mm at 8pt)
    NAME_MAX  = max(12, int(col_w / 1.95))
    DESC_MAX  = max(20, int(col_w / 1.55))

    def _trunc(text: str, limit: int) -> str:
        """Truncate text to limit characters, adding '..' if cut."""
        return text if len(text) <= limit else text[:limit - 2] + ".."

    # ── Day blocks ────────────────────────────────────────────────────────────
    for day_idx, day_plan in enumerate(itinerary):
        col_idx = day_idx // days_per_col
        row_idx = day_idx % days_per_col

        col_x = MARGIN + col_idx * (col_w + COL_GAP)
        day_y = content_top + row_idx * day_h

        # Day header bar
        pdf.set_fill_color(*RED)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(col_x, day_y)
        pdf.cell(col_w, DAY_HDR_H, f"  Day {day_plan['day']}", ln=False, fill=True)

        # Weather strip
        w_idx = day_plan["day"] - 1
        wx = day_y + DAY_HDR_H
        if w_idx < len(forecast):
            w = forecast[w_idx]
            pdf.set_fill_color(*LIGHT_RED)
            pdf.set_text_color(*GREY)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_xy(col_x, wx)
            pdf.cell(
                col_w, WEATHER_H,
                f"  {w['label']}  {w['min']}/{w['max']}C  |  rain: {w['rain']} mm",
                ln=False, fill=True,
            )

        # Slot cards
        slots_start = day_y + DAY_HDR_H + WEATHER_H

        for s_idx, slot in enumerate(SLOTS):
            act     = day_plan["slots"][slot]
            name    = act["name"]
            cat     = act["category"]
            desc    = desc_lookup.get(name, "")
            is_free = name.startswith("Free time")

            sy = slots_start + s_idx * SLOT_H

            # Alternating background
            pdf.set_fill_color(*(LIGHT_GREY if s_idx % 2 == 0 else WHITE))
            pdf.rect(col_x, sy, col_w, SLOT_H, style="F")

            # Slot label (e.g. MORNING)
            pdf.set_xy(col_x + 1.5, sy + 1)
            pdf.set_text_color(*RED)
            pdf.set_font("Helvetica", "B", 6.5)
            pdf.cell(col_w - 2, 3, slot.upper(), ln=False)

            if is_free:
                pdf.set_xy(col_x + 1.5, sy + 4.2)
                pdf.set_text_color(*GREY)
                pdf.set_font("Helvetica", "I", 7.5)
                pdf.cell(col_w - 2, 3.5, "Free time", ln=False)
            else:
                # Activity name
                pdf.set_xy(col_x + 1.5, sy + 4.2)
                pdf.set_text_color(*DARK)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(col_w - 2, 3.5, _trunc(name, NAME_MAX), ln=False)

                # Description (one truncated line)
                if desc and SLOT_H > 13:
                    pdf.set_xy(col_x + 1.5, sy + 8)
                    pdf.set_text_color(*GREY)
                    pdf.set_font("Helvetica", "", 6.5)
                    pdf.cell(col_w - 2, 3, _trunc(desc, DESC_MAX), ln=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(*LIGHT_RED)
    pdf.rect(0, PAGE_H - FOOTER_H, PAGE_W, FOOTER_H, style="F")
    pdf.set_xy(0, PAGE_H - FOOTER_H + 1)
    pdf.set_text_color(*GREY)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(
        PAGE_W, 5,
        "Swiss Vacation Planner  |  ML: Cosine-similarity KNN  |  Weather: Open-Meteo",
        align="C",
    )

    return bytes(pdf.output())


def step_itinerary() -> None:
    render_progress(3)

    city      = st.session_state["city"]
    num_days  = st.session_state["num_days"]
    season    = st.session_state.get("season", "summer")
    prefs     = st.session_state.get("prefs", {})
    itinerary = st.session_state["itinerary"]

    st.markdown(
        f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>',
        unsafe_allow_html=True,
    )

    # Summary bar: destination, season, duration, top 3 interests
    top_cats = sorted(prefs, key=prefs.get, reverse=True)[:3]
    top_str  = " · ".join(c for c in top_cats)
    st.markdown(
        f'<div class="summary-box">'
        f'<strong>{city}</strong> &nbsp;|&nbsp; '
        f'<strong>{num_days} day{"s" if num_days > 1 else ""}</strong> &nbsp;|&nbsp; '
        f'<strong>{top_str}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Fetch the right weather source depending on whether the chosen season
    # matches today's real-world season
    forecast, weather_source = get_forecast_for_season(city, season, num_days)
    if forecast:
        st.markdown(
            f'<div style="font-size:0.82rem; color:#555555; margin-bottom:0.6rem;">'
            f'<strong>{weather_source}</strong> — '
            f'{"real-time data from Open-Meteo" if "Live" in weather_source else "typical Swiss conditions for the selected season"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Timetable ─────────────────────────────────────────────────────────────
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
                        f'<div class="weather-box">'
                        f'{w["label"]} · {w["min"]}°/{w["max"]}°C '
                        f'· {w["rain"]} mm rain</div>',
                        unsafe_allow_html=True,
                    )

                # One card per time slot
                for slot in SLOTS:
                    act   = day_plan["slots"][slot]
                    name  = act["name"]
                    cat   = act["category"]

                    if name.startswith("Free time"):
                        css   = "tt-free"
                        icon  = ""
                        badge = ""
                    else:
                        css   = SLOT_CSS[slot]
                        icon  = SLOT_ICON[slot]
                        color = CATEGORY_COLORS.get(cat, "#2e6da4")
                        badge = ""

                    cols[i].markdown(
                        f'<div class="tt-slot {css}">'
                        f'{icon} <strong>{slot}</strong>'
                        f'{badge}<br>'
                        f'<span class="act-meta">{name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Activity breakdown chart ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="step-heading" style="font-size:1.1rem;">Activity breakdown</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">How your days are distributed across activity types.</div>',
        unsafe_allow_html=True,
    )
    render_activity_chart(itinerary)

    # ── Download itinerary as PDF ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="step-heading" style="font-size:1.1rem;">Download your itinerary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">Get a PDF with your full day-by-day plan, '
        'activity descriptions, and weather forecast.</div>',
        unsafe_allow_html=True,
    )

    # Generate the PDF (pass the already-fetched forecast so we don't re-call the API)
    try:
        pdf_bytes = generate_itinerary_pdf(
            itinerary=itinerary,
            city=city,
            num_days=num_days,
            season=season,
            forecast=forecast,
            df=load_activities(),
        )
        st.download_button(
            label="Download itinerary (PDF)",
            data=pdf_bytes,
            file_name=f"{city}_{season}_itinerary.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.warning(f"PDF generation unavailable: {e}. Install fpdf2 with: pip install fpdf2")

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_back3, col_restart = st.columns([1, 1])
    with col_back3:
        if st.button("← Change preferences"):
            st.session_state["step"] = 2
            st.rerun()
    with col_restart:
        if st.button("Start over"):
            for k in ["city", "num_days", "season", "season_choice",
                      "prefs", "ranked", "itinerary", "step"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown(
        '<div class="footer">'
        'Swiss Vacation Planner · Streamlit · '
        'Weather: Open-Meteo (real-time) · ML: Cosine-similarity KNN'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_app() -> None:
    # Initialise step to 1 on first load, then route to the correct step function
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    step = st.session_state["step"]
    if step == 1:
        step_destination()
    elif step == 2:
        step_preferences()
    elif step == 3:
        step_itinerary()
