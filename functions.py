import os
import random
import pandas as pd
import streamlit as st

from weather import get_weather

# This is the folder where our Python file lives
# We use it to find the locations.csv file
FOLDER = os.path.dirname(os.path.abspath(__file__))

# The 6 activity categories we support
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

# The 3 time slots of a day
SLOTS = ["Morning", "Afternoon", "Evening"]

# CSS class for each slot (used for colouring in the timetable)
SLOT_CLASSES = {
    "Morning":   "tt-morning",
    "Afternoon": "tt-afternoon",
    "Evening":   "tt-evening",
}

# Emoji icon for each slot
SLOT_ICONS = {
    "Morning":   "🌅",
    "Afternoon": "☀️",
    "Evening":   "🌙",
}

#location.csv reader

@st.cache_data
def load_activities():
    file_path = os.path.join(FOLDER, "locations.csv")
    df = pd.read_csv(file_path)
    return df



# Weather reader 
@st.cache_data(ttl=3600)
def get_city_forecast(city, num_days):
    df = load_activities()

    # Find the rows for this city
    city_rows = df[df["city"] == city]

    # If the city is not in the CSV, return an empty list
    if city_rows.empty:
        return []

    # Take the latitude and longitude from the first row
    lat = float(city_rows.iloc[0]["lat"])
    lon = float(city_rows.iloc[0]["lon"])

    # Call the weather API and return the forecast
    return get_weather(lat, lon, num_days)


# ------------------------------------------------------------------
# Check if the weather is bad (rainy, snowy, stormy...)
# Returns True if it's better to stay indoors
# ------------------------------------------------------------------
def is_bad_weather(weather_label):
    bad_words = ["rain", "drizzle", "snow", "storm", "thunder"]
    weather_label = weather_label.lower()
    for word in bad_words:
        if word in weather_label:
            return True
    return False


# ------------------------------------------------------------------
# Return a sorted list of all cities in the CSV
# ------------------------------------------------------------------
def get_cities(df):
    cities = df["city"].dropna().unique().tolist()
    cities = sorted(cities)
    return cities


# ------------------------------------------------------------------
# Return only the activities for the chosen city
# ------------------------------------------------------------------
def city_activities(df, city):
    filtered = df[df["city"] == city]
    filtered = filtered.reset_index(drop=True)
    return filtered


# ------------------------------------------------------------------
# Keep only activities that match the user's chosen categories
# If nothing matches (or no preference chosen), return everything
# ------------------------------------------------------------------
def filter_by_preferences(activities, prefs):
    # If the user didn't pick any preference, return all activities
    if len(prefs) == 0:
        return activities

    # Keep only rows where the category is in the user's list
    filtered = activities[activities["category"].isin(prefs)]

    # If the filter removed everything, fall back to all activities
    if filtered.empty:
        return activities

    return filtered


# ------------------------------------------------------------------
# Find the best time slot for an activity
# The CSV has a "time_slot" column like "Morning|Afternoon"
# We pick the first one we recognise
# ------------------------------------------------------------------
def get_best_slot(time_slot_value):
    # If there's no value, pick a random slot
    if pd.isna(time_slot_value):
        return random.choice(SLOTS)

    # Split by "|" to get a list of slots e.g. ["Morning", "Afternoon"]
    parts = str(time_slot_value).split("|")
    parts = [p.strip() for p in parts]

    # Return the first slot that matches our list
    for slot in SLOTS:
        if slot in parts:
            return slot

    # If nothing matched, pick randomly
    return random.choice(SLOTS)


# ------------------------------------------------------------------
# Build a day-by-day itinerary
# Each day has Morning, Afternoon, Evening slots
# We try to respect the weather (indoor on rainy days, outdoor on sunny days)
# ------------------------------------------------------------------
def build_itinerary(activities, num_days, forecast):
    # Shuffle the activities so the order is random each time
    activities = activities.sample(frac=1).reset_index(drop=True)

    # Convert the DataFrame to a list of dictionaries (easier to work with)
    all_rows = activities.to_dict("records")

    # Sort activities into buckets by their best time slot
    # e.g. buckets["Morning"] = [list of morning activities]
    buckets = {
        "Morning":   [],
        "Afternoon": [],
        "Evening":   [],
    }
    for row in all_rows:
        best_slot = get_best_slot(row.get("time_slot", ""))
        buckets[best_slot].append(row)

    # This set keeps track of activities we already placed
    # so we don't repeat the same activity twice
    already_used = set()

    # This will be our final itinerary: a list of day plans
    itinerary = []

    for day_number in range(1, num_days + 1):
        # Check if this day is rainy or sunny
        day_index = day_number - 1
        if day_index < len(forecast) and is_bad_weather(forecast[day_index]["label"]):
            prefer_indoor = True   # bad weather → stay inside
        else:
            prefer_indoor = False  # good weather → go outside

        # Build the plan for this day (one activity per slot)
        day_plan = {}

        for slot in SLOTS:
            chosen_activity = None

            # Step 1: Try to find an activity in the right slot
            # that also matches the weather preference
            for row in buckets[slot]:
                name = row["activity_name"]
                setting = str(row.get("indoor_outdoor", "")).lower()

                if name in already_used:
                    continue  # skip activities we already used

                if prefer_indoor and (setting == "indoor" or setting == "both"):
                    chosen_activity = row
                    break
                elif not prefer_indoor and (setting == "outdoor" or setting == "both"):
                    chosen_activity = row
                    break

            # Step 2: If we didn't find one, try all slots (still matching weather)
            if chosen_activity is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        name = row["activity_name"]
                        setting = str(row.get("indoor_outdoor", "")).lower()

                        if name in already_used:
                            continue

                        if prefer_indoor and (setting == "indoor" or setting == "both"):
                            chosen_activity = row
                            break
                        elif not prefer_indoor and (setting == "outdoor" or setting == "both"):
                            chosen_activity = row
                            break

                    if chosen_activity is not None:
                        break

            # Step 3: Last resort — just take any unused activity
            if chosen_activity is None:
                for any_slot in SLOTS:
                    for row in buckets[any_slot]:
                        name = row["activity_name"]
                        if name not in already_used:
                            chosen_activity = row
                            break
                    if chosen_activity is not None:
                        break

            # Save the result for this slot
            if chosen_activity is not None:
                day_plan[slot] = chosen_activity["activity_name"]
                already_used.add(chosen_activity["activity_name"])
            else:
                day_plan[slot] = "Free time — explore at your own pace"

        # Add this day to the itinerary
        itinerary.append({"day": day_number, "slots": day_plan})

    return itinerary


# ------------------------------------------------------------------
# Draw the progress bar at the top of the page
# Shows which step the user is on (1, 2, or 3)
# ------------------------------------------------------------------
def render_progress(current_step):
    steps = ["1 · Destination", "2 · Preferences", "3 · Your Itinerary"]
    cols = st.columns(3)

    for i in range(3):
        step_number = i + 1
        label = steps[i]

        if step_number < current_step:
            css_class = "prog-step done"      # already completed
        elif step_number == current_step:
            css_class = "prog-step current"   # currently on this step
        else:
            css_class = "prog-step"           # not reached yet

        cols[i].markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)

    st.write("")


# ------------------------------------------------------------------
# STEP 1 — Ask the user where they want to go and for how many days
# ------------------------------------------------------------------
def step_destination():
    render_progress(1)

    st.markdown('<div class="step-heading">Where are you heading?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Pick a Swiss destination and how many days you have.</div>', unsafe_allow_html=True)

    # Load the data and get the list of cities
    df = load_activities()
    cities = get_cities(df)

    # Show the dropdowns side by side
    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("Destination", cities)
    with col2:
        num_days = st.selectbox("Number of days", [1, 2, 3, 4, 5, 6, 7], index=2)

    st.write("")

    # When the user clicks Next, save their choices and go to step 2
    if st.button("Next →"):
        st.session_state["city"] = city
        st.session_state["num_days"] = num_days
        st.session_state["step"] = 2
        st.rerun()


# ------------------------------------------------------------------
# STEP 2 — Ask the user what kind of activities they enjoy
# ------------------------------------------------------------------
def step_preferences():
    render_progress(2)

    st.markdown('<div class="step-heading">What do you enjoy?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Choose one or more activity types — or skip to include everything.</div>', unsafe_allow_html=True)

    # Show checkboxes in 3 columns
    selected_prefs = []
    cols = st.columns(3)

    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        col = cols[i % 3]  # puts items into columns 0, 1, 2, 0, 1, 2, ...
        with col:
            if st.checkbox(category):
                selected_prefs.append(category)

    st.write("")

    col_back, col_next = st.columns([1, 5])

    with col_back:
        if st.button("← Back"):
            st.session_state["step"] = 1
            st.rerun()

    with col_next:
        if st.button("Build my itinerary →"):
            # Save the preferences
            st.session_state["prefs"] = selected_prefs

            # Load activities for the chosen city
            df = load_activities()
            acts = city_activities(df, st.session_state["city"])

            # If somehow there are no activities, use all activities as a backup
            if acts.empty:
                acts = df

            # Filter by the user's preferences
            acts = filter_by_preferences(acts, selected_prefs)

            # Get the weather forecast
            forecast = get_city_forecast(st.session_state["city"], st.session_state["num_days"])

            # Build the itinerary and save it
            st.session_state["itinerary"] = build_itinerary(acts, st.session_state["num_days"], forecast)
            st.session_state["step"] = 3
            st.rerun()


# ------------------------------------------------------------------
# STEP 3 — Show the final itinerary to the user
# ------------------------------------------------------------------
def step_itinerary():
    render_progress(3)

    # Read saved values from the session
    city = st.session_state["city"]
    num_days = st.session_state["num_days"]
    prefs = st.session_state.get("prefs", [])
    itinerary = st.session_state["itinerary"]

    # Page heading
    st.markdown(f'<div class="step-heading">Your {num_days}-day {city} itinerary</div>', unsafe_allow_html=True)

    # Summary line showing the user's choices
    if len(prefs) == 0:
        pref_text = "All activities"
    else:
        pref_text = ", ".join(prefs)

    st.markdown(
        f'<div class="summary-box">'
        f'<strong>Destination:</strong> {city} &nbsp;|&nbsp; '
        f'<strong>Days:</strong> {num_days} &nbsp;|&nbsp; '
        f'<strong>Interests:</strong> {pref_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Get the weather forecast
    forecast = get_city_forecast(city, num_days)

    # Show the timetable — max 3 days per row
    for row_start in range(0, num_days, 3):
        days_in_this_row = itinerary[row_start : row_start + 3]
        cols = st.columns(len(days_in_this_row))

        for i in range(len(days_in_this_row)):
            day_plan = days_in_this_row[i]
            col = cols[i]

            with col:
                # Day header
                col.markdown(f'<div class="tt-header">Day {day_plan["day"]}</div>', unsafe_allow_html=True)

                # Weather for this day (if available)
                day_index = day_plan["day"] - 1
                if day_index < len(forecast):
                    w = forecast[day_index]
                    col.markdown(
                        f'<div style="font-size:0.85rem; color:#1a3a5c; margin-bottom:0.5rem;">'
                        f'{w["label"]} · {w["min"]}°/{w["max"]}°C · rain {w["rain"]} mm'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Show each time slot
                for slot in SLOTS:
                    activity = day_plan["slots"][slot]

                    if activity.startswith("Free time"):
                        css = "tt-free"
                        icon = ""
                    else:
                        css = SLOT_CLASSES[slot]
                        icon = SLOT_ICONS[slot]

                    col.markdown(
                        f'<div class="tt-slot {css}">{icon} <strong>{slot}</strong><br>'
                        f'<span class="act-meta">{activity}</span></div>',
                        unsafe_allow_html=True,
                    )

    st.write("")

    # Navigation buttons
    col_back, col_restart = st.columns([1, 5])
    with col_back:
        if st.button("← Change preferences"):
            st.session_state["step"] = 2
            st.rerun()
    with col_restart:
        if st.button("Start over"):
            st.session_state.pop("city", None)
            st.session_state.pop("num_days", None)
            st.session_state.pop("prefs", None)
            st.session_state.pop("itinerary", None)
            st.session_state.pop("step", None)
            st.rerun()


# ------------------------------------------------------------------
# Entry point — called from app.py
# Decides which step to show based on st.session_state["step"]
# ------------------------------------------------------------------
def run_app():
    # If this is the first time the app runs, start at step 1
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    step = st.session_state["step"]

    if step == 1:
        step_destination()
    elif step == 2:
        step_preferences()
    elif step == 3:
        step_itinerary()
