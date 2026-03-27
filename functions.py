# functions.py
from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd
import streamlit as st

from ml import (
    build_week_programme,
    get_knn_suggestions,
    get_refined_recommendations,
)


# -------------------------------------------------
# Session state
# -------------------------------------------------
def initialize_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 1

    if "selected_region" not in st.session_state:
        st.session_state.selected_region = None

    if "selected_season" not in st.session_state:
        st.session_state.selected_season = None

    if "selected_setting" not in st.session_state:
        st.session_state.selected_setting = None

    if "selected_vacation_type" not in st.session_state:
        st.session_state.selected_vacation_type = None

    if "activities_df" not in st.session_state:
        st.session_state.activities_df = load_dummy_activities()

    if "suggested_activities" not in st.session_state:
        st.session_state.suggested_activities = None

    if "selected_favourites" not in st.session_state:
        st.session_state.selected_favourites = []

    if "final_programme" not in st.session_state:
        st.session_state.final_programme = None


# -------------------------------------------------
# Dummy dataset
# -------------------------------------------------
def load_dummy_activities() -> pd.DataFrame:
    data = [
        {
            "name": "Jungfrau Winter Sports",
            "region": "South",
            "season": "Winter",
            "setting": "Mountains",
            "vacation_type": "Sports",
            "time_slot": "Morning",
            "duration": 4,
        },
        {
            "name": "Zermatt Ski Area",
            "region": "South",
            "season": "Winter",
            "setting": "Mountains",
            "vacation_type": "Sports",
            "time_slot": "Morning",
            "duration": 5,
        },
        {
            "name": "Lugano Lake Walk",
            "region": "South",
            "season": "Summer",
            "setting": "City",
            "vacation_type": "Chill",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Bellinzona Castle Visit",
            "region": "South",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Locarno Night Bar",
            "region": "South",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Parties",
            "time_slot": "Night",
            "duration": 3,
        },
        {
            "name": "Geneva Museum Tour",
            "region": "West",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Lausanne Old Town Walk",
            "region": "West",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Morning",
            "duration": 2,
        },
        {
            "name": "Montreux Lakeside Relax",
            "region": "West",
            "season": "Summer",
            "setting": "City",
            "vacation_type": "Chill",
            "time_slot": "Evening",
            "duration": 2,
        },
        {
            "name": "Verbier Hiking Trail",
            "region": "West",
            "season": "Summer",
            "setting": "Mountains",
            "vacation_type": "Sports",
            "time_slot": "Morning",
            "duration": 4,
        },
        {
            "name": "Fribourg Pub Crawl",
            "region": "West",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Parties",
            "time_slot": "Night",
            "duration": 3,
        },
        {
            "name": "Zurich Club Night",
            "region": "East",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Parties",
            "time_slot": "Night",
            "duration": 4,
        },
        {
            "name": "St. Gallen Abbey Visit",
            "region": "East",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Davos Mountain Biking",
            "region": "East",
            "season": "Summer",
            "setting": "Mountains",
            "vacation_type": "Sports",
            "time_slot": "Morning",
            "duration": 4,
        },
        {
            "name": "Arosa Wellness Spa",
            "region": "East",
            "season": "Any",
            "setting": "Mountains",
            "vacation_type": "Chill",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Lake Constance Sunset",
            "region": "East",
            "season": "Summer",
            "setting": "City",
            "vacation_type": "Chill",
            "time_slot": "Evening",
            "duration": 2,
        },
        {
            "name": "Interlaken Adventure Park",
            "region": "South",
            "season": "Summer",
            "setting": "Mountains",
            "vacation_type": "Sports",
            "time_slot": "Afternoon",
            "duration": 3,
        },
        {
            "name": "Bern Historical Museum",
            "region": "West",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Afternoon",
            "duration": 2,
        },
        {
            "name": "Chur Cultural Walk",
            "region": "East",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Visit",
            "time_slot": "Morning",
            "duration": 2,
        },
        {
            "name": "Ascona Lakeside Dinner",
            "region": "South",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Chill",
            "time_slot": "Evening",
            "duration": 2,
        },
        {
            "name": "Neuchâtel Wine Bar",
            "region": "West",
            "season": "Any",
            "setting": "City",
            "vacation_type": "Parties",
            "time_slot": "Night",
            "duration": 2,
        },
    ]
    return pd.DataFrame(data)


# -------------------------------------------------
# UI helpers
# -------------------------------------------------
def render_progress_bar() -> None:
    labels = {
        1: "1. Region",
        2: "2. Criteria",
        3: "3. Activities",
        4: "4. Programme",
    }

    cols = st.columns(4)
    for i, col in enumerate(cols, start=1):
        if st.session_state.step == i:
            col.markdown(f"**{labels[i]}**")
        else:
            col.markdown(labels[i])


def render_region_selector() -> None:
    st.markdown("### Choose your region in Switzerland")
    st.caption("For now we use buttons. Later you can replace this with the interactive map.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("West", use_container_width=True):
            st.session_state.selected_region = "West"
            st.session_state.step = 2
            st.rerun()

    with col2:
        if st.button("East", use_container_width=True):
            st.session_state.selected_region = "East"
            st.session_state.step = 2
            st.rerun()

    with col3:
        if st.button("South", use_container_width=True):
            st.session_state.selected_region = "South"
            st.session_state.step = 2
            st.rerun()


def render_criteria_step() -> None:
    st.markdown("### Trip criteria")
    st.info(f"Selected region: **{st.session_state.selected_region}**")

    season = st.selectbox(
        "Season",
        ["Summer", "Autumn", "Winter", "Spring"],
        index=None,
        placeholder="Choose a season",
    )

    setting = st.selectbox(
        "Setting",
        ["Mountains", "City"],
        index=None,
        placeholder="Choose a setting",
    )

    vacation_type = st.selectbox(
        "Vacation type",
        ["Chill", "Sports", "Parties", "Visit"],
        index=None,
        placeholder="Choose a vacation type",
    )

    if st.button("Confirm criteria", use_container_width=True):
        if season is None or setting is None or vacation_type is None:
            st.warning("Please choose all criteria before continuing.")
            return

        st.session_state.selected_season = season
        st.session_state.selected_setting = setting
        st.session_state.selected_vacation_type = vacation_type

        suggestions = get_knn_suggestions(
            activities_df=st.session_state.activities_df,
            selected_region=st.session_state.selected_region,
            selected_season=st.session_state.selected_season,
            selected_setting=st.session_state.selected_setting,
            selected_vacation_type=st.session_state.selected_vacation_type,
            n_results=10,
        )

        st.session_state.suggested_activities = suggestions
        st.session_state.step = 3
        st.rerun()


def render_suggestions_step() -> None:
    st.markdown("### Suggested activities")
    st.caption("Choose the activities you like most.")

    suggestions = st.session_state.suggested_activities
    if suggestions is None or suggestions.empty:
        st.warning("No activities available yet.")
        return

    selected_names: List[str] = []

    for _, row in suggestions.iterrows():
        label = (
            f"**{row['name']}**  \n"
            f"Region: {row['region']} | "
            f"Season: {row['season']} | "
            f"Setting: {row['setting']} | "
            f"Type: {row['vacation_type']} | "
            f"Slot: {row['time_slot']}"
        )
        checked = st.checkbox(label, key=f"activity_{row['name']}")
        if checked:
            selected_names.append(row["name"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to criteria", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col2:
        if st.button("Generate programme", use_container_width=True):
            if len(selected_names) == 0:
                st.warning("Please select at least one activity.")
                return

            st.session_state.selected_favourites = selected_names

            favourites_df = st.session_state.activities_df[
                st.session_state.activities_df["name"].isin(selected_names)
            ].copy()

            recommended_df = get_refined_recommendations(
                activities_df=st.session_state.activities_df,
                favourite_names=selected_names,
                n_results=28,
            )

            programme = build_week_programme(
                favourite_df=favourites_df,
                recommended_df=recommended_df,
            )

            st.session_state.final_programme = programme
            st.session_state.step = 4
            st.rerun()


def render_programme_step() -> None:
    st.markdown("### Your 1-week programme")

    programme = st.session_state.final_programme
    if programme is None or programme.empty:
        st.warning("No programme generated yet.")
        return

    st.dataframe(programme, use_container_width=True)

    if st.button("Start again", use_container_width=True):
        reset_app()
        st.rerun()




def reset_app() -> None:
    keys_to_clear = [
        "step",
        "selected_region",
        "selected_season",
        "selected_setting",
        "selected_vacation_type",
        "suggested_activities",
        "selected_favourites",
        "final_programme",
        "activities_df",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    initialize_session_state()


# -------------------------------------------------
# Main entry point
# -------------------------------------------------
def run_app() -> None:
    initialize_session_state()
    render_progress_bar()

    st.write("")

    if st.session_state.step == 1:
        render_region_selector()

    elif st.session_state.step == 2:
        render_criteria_step()

    elif st.session_state.step == 3:
        render_suggestions_step()

    elif st.session_state.step == 4:
        render_programme_step()
