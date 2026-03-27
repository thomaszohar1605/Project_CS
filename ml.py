# ml.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# -------------------------------------------------
# Encoding maps
# -------------------------------------------------
REGION_MAP: Dict[str, int] = {
    "West": 0,
    "East": 1,
    "South": 2,
}

SEASON_MAP: Dict[str, int] = {
    "Spring": 0,
    "Summer": 1,
    "Autumn": 2,
    "Winter": 3,
    "Any": 1,   # neutral fallback
}

SETTING_MAP: Dict[str, int] = {
    "City": 0,
    "Mountains": 1,
    "Any": 0,
}

VACATION_TYPE_MAP: Dict[str, int] = {
    "Chill": 0,
    "Sports": 1,
    "Visit": 2,
    "Parties": 3,
    "Any": 1,
}

TIME_SLOT_MAP: Dict[str, int] = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3,
}


# -------------------------------------------------
# Hard constraints
# -------------------------------------------------
def apply_hard_constraints(
    df: pd.DataFrame,
    selected_season: str,
) -> pd.DataFrame:
    """
    Remove physically impossible activities before KNN.
    For now this works with simple name keyword rules because
    your current test dataset is still dummy data.
    """
    filtered = df.copy()

    if selected_season != "Winter":
        filtered = filtered[
            ~filtered["name"].str.lower().str.contains("ski|winter sports", na=False)
        ]

    if selected_season == "Winter":
        filtered = filtered[
            ~filtered["name"].str.lower().str.contains("lake walk|sunset", na=False)
        ]

    return filtered.reset_index(drop=True)


# -------------------------------------------------
# Feature building
# -------------------------------------------------
def encode_activities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numerical features for each activity.
    """
    features = df.copy()

    features["region_enc"] = features["region"].map(REGION_MAP).fillna(0)
    features["season_enc"] = features["season"].map(SEASON_MAP).fillna(1)
    features["setting_enc"] = features["setting"].map(SETTING_MAP).fillna(0)
    features["vacation_type_enc"] = features["vacation_type"].map(VACATION_TYPE_MAP).fillna(1)
    features["time_slot_enc"] = features["time_slot"].map(TIME_SLOT_MAP).fillna(1)
    features["duration_enc"] = features["duration"].fillna(2)

    return features


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Return activity dataframe with encoded columns and numpy feature matrix.
    """
    encoded = encode_activities(df)

    feature_cols = [
        "region_enc",
        "season_enc",
        "setting_enc",
        "vacation_type_enc",
        "time_slot_enc",
        "duration_enc",
    ]

    X = encoded[feature_cols].to_numpy(dtype=float)
    return encoded, X


def build_user_vector(
    selected_region: str,
    selected_season: str,
    selected_setting: str,
    selected_vacation_type: str,
) -> np.ndarray:
    """
    Build the first user preference vector from the chosen criteria.
    """
    return np.array(
        [
            REGION_MAP.get(selected_region, 0),
            SEASON_MAP.get(selected_season, 1),
            SETTING_MAP.get(selected_setting, 0),
            VACATION_TYPE_MAP.get(selected_vacation_type, 1),
            1.5,   # neutral time slot between afternoon/evening
            2.5,   # neutral duration
        ],
        dtype=float,
    )


def build_refined_user_vector(
    favourites_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the second user vector from the selected favourite activities.
    """
    encoded, X = get_feature_matrix(favourites_df)
    _ = encoded  # kept for readability
    return np.mean(X, axis=0)


# -------------------------------------------------
# KNN model
# -------------------------------------------------
def fit_knn_model(X: np.ndarray, n_neighbors: int = 10) -> NearestNeighbors:
    model = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(X)),
        metric="cosine",
    )
    model.fit(X)
    return model


def find_nearest_activities(
    df: pd.DataFrame,
    query_vector: np.ndarray,
    n_results: int = 10,
) -> pd.DataFrame:
    """
    Return the n closest activities to the given query vector.
    """
    if df.empty:
        return df.copy()

    encoded, X = get_feature_matrix(df)
    model = fit_knn_model(X, n_neighbors=min(n_results, len(df)))

    distances, indices = model.kneighbors(query_vector.reshape(1, -1))
    nearest_df = encoded.iloc[indices[0]].copy()
    nearest_df["similarity_distance"] = distances[0]

    cols_to_drop = [
        "region_enc",
        "season_enc",
        "setting_enc",
        "vacation_type_enc",
        "time_slot_enc",
        "duration_enc",
    ]
    nearest_df = nearest_df.drop(columns=cols_to_drop, errors="ignore")

    return nearest_df.reset_index(drop=True)


# -------------------------------------------------
# Public functions for the app
# -------------------------------------------------
def get_knn_suggestions(
    activities_df: pd.DataFrame,
    selected_region: str,
    selected_season: str,
    selected_setting: str,
    selected_vacation_type: str,
    n_results: int = 10,
) -> pd.DataFrame:
    """
    First KNN pass:
    Use the user's criteria to generate the 10 suggestions.
    """
    if activities_df.empty:
        return activities_df.copy()

    filtered = apply_hard_constraints(activities_df, selected_season)

    # Keep same region strongly by prefiltering here
    regional_pool = filtered[filtered["region"] == selected_region].copy()

    # fallback if region pool is too small
    if len(regional_pool) < max(5, n_results):
        regional_pool = filtered.copy()

    query_vector = build_user_vector(
        selected_region=selected_region,
        selected_season=selected_season,
        selected_setting=selected_setting,
        selected_vacation_type=selected_vacation_type,
    )

    return find_nearest_activities(
        df=regional_pool,
        query_vector=query_vector,
        n_results=n_results,
    )


def get_refined_recommendations(
    activities_df: pd.DataFrame,
    favourite_names: List[str],
    selected_region: str,
    n_results: int = 28,
) -> pd.DataFrame:
    """
    Second KNN pass:
    Use the user's selected favourites to recommend more similar activities.
    """
    if activities_df.empty:
        return activities_df.copy()

    favourites_df = activities_df[activities_df["name"].isin(favourite_names)].copy()
    if favourites_df.empty:
        return pd.DataFrame()

    regional_df = activities_df[activities_df["region"] == selected_region].copy()

    remaining_df = regional_df[
        ~regional_df["name"].isin(favourite_names)
    ].copy()

    if remaining_df.empty:
        return favourites_df.copy()

    refined_vector = build_refined_user_vector(favourites_df)

    recommendations = find_nearest_activities(
        df=remaining_df,
        query_vector=refined_vector,
        n_results=n_results,
    )

    return recommendations.reset_index(drop=True)


def build_week_programme(
    favourite_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a simple 7-day x 4-slot programme from favourites + recommendations.
    This is not ML anymore; this is scheduling logic.
    """
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    slots = ["Morning", "Afternoon", "Evening", "Night"]

    pool = pd.concat([favourite_df, recommended_df], ignore_index=True).drop_duplicates(
        subset=["name"]
    )

    if pool.empty:
        return pd.DataFrame()

    slot_buckets = {
        slot: pool[pool["time_slot"] == slot].copy() for slot in slots
    }

    used_names = set()
    rows = []

    for day in days:
        row = {"Day": day}

        for slot in slots:
            bucket = slot_buckets.get(slot, pd.DataFrame()).copy()
            bucket = bucket[~bucket["name"].isin(used_names)]

            if bucket.empty:
                fallback = pool[~pool["name"].isin(used_names)].copy()
                if fallback.empty:
                    row[slot] = "Free time"
                    continue
                chosen = fallback.iloc[0]
            else:
                chosen = bucket.iloc[0]

            row[slot] = chosen["name"]
            used_names.add(chosen["name"])

        rows.append(row)

    return pd.DataFrame(rows)
