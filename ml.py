# ml.py
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ── Encoding maps ─────────────────────────────────────────────────────────────

REGION_MAP: Dict[str, int] = {
    "Central": 0,
    "Lake Geneva": 1,
    "Alpine": 2,
    "Ticino": 3,
    "East": 4,
}

TYPE_MAP: Dict[str, int] = {
    "Sightseeing": 0,
    "Museum": 1,
    "Hiking": 2,
    "Skiing": 3,
    "Adventure": 4,
    "Nature": 5,
    "Relaxation": 6,
    "Food": 7,
    "Nightlife": 8,
    "Shopping": 9,
    "Wellness": 10,
}

SEASON_MAP: Dict[str, int] = {
    "Spring": 0,
    "Summer": 1,
    "Autumn": 2,
    "Winter": 3,
}


# ── Hard constraints ──────────────────────────────────────────────────────────

def apply_hard_constraints(
    df: pd.DataFrame,
    selected_season: str,
    selected_region: str,
    max_budget: float,
) -> pd.DataFrame:
    """
    Remove activities that are impossible or out of budget before KNN runs.
    """
    filtered = df.copy()

    # Region filter
    filtered = filtered[filtered["region"] == selected_region]

    # Season filter — keep activities whose best_season list includes
    # the selected season OR contains "Any"
    filtered = filtered[
        filtered["best_season"].apply(
            lambda seasons: selected_season in seasons or "Any" in seasons
        )
    ]

    # Budget filter
    filtered = filtered[filtered["estimated_cost"] <= max_budget]

    return filtered.reset_index(drop=True)


# ── Feature encoding ──────────────────────────────────────────────────────────

def encode_activities(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Encode categorical columns into numbers for KNN.
    Returns (encoded_df, feature_matrix).
    """
    enc = df.copy()

    enc["region_enc"]  = enc["region"].map(REGION_MAP).fillna(0)
    enc["type_enc"]    = enc["type"].map(TYPE_MAP).fillna(0)
    enc["cost_enc"]    = enc["estimated_cost"].fillna(0) / 400   # normalise 0–1
    enc["rating_enc"]  = enc["rating"].fillna(3.0) / 5.0         # normalise 0–1
    enc["dur_enc"]     = enc["duration_hours"].fillna(2) / 8     # normalise 0–1

    # Season encoding: average of all seasons in the best_season list
    def mean_season(seasons):
        vals = [SEASON_MAP.get(s, 1) for s in seasons if s in SEASON_MAP]
        return np.mean(vals) / 3.0 if vals else 0.5

    enc["season_enc"] = enc["best_season"].apply(mean_season)

    feature_cols = [
        "region_enc",
        "type_enc",
        "cost_enc",
        "rating_enc",
        "dur_enc",
        "season_enc",
    ]

    X = enc[feature_cols].to_numpy(dtype=float)
    return enc, X


# ── User vector ───────────────────────────────────────────────────────────────

def build_user_vector(
    selected_region: str,
    selected_season: str,
    selected_type: str,
    max_budget: float,
) -> np.ndarray:
    """
    Build a query vector from the user's chosen preferences.
    This is what KNN compares against every activity.
    """
    return np.array([
        REGION_MAP.get(selected_region, 0) / 4,       # normalised region
        TYPE_MAP.get(selected_type, 0) / 10,           # normalised type
        min(max_budget, 400) / 400,                    # normalised budget
        0.9,                                           # prefer high-rated
        0.4,                                           # prefer medium duration
        SEASON_MAP.get(selected_season, 1) / 3,        # season
    ], dtype=float)


def build_refined_user_vector(favourites_df: pd.DataFrame) -> np.ndarray:
    """
    Build a refined vector from the user's favourite activities (Step 2 KNN).
    Takes the average feature vector of all selected favourites.
    """
    _, X = encode_activities(favourites_df)
    return np.mean(X, axis=0)


# ── KNN model ─────────────────────────────────────────────────────────────────

def fit_knn(X: np.ndarray, n_neighbors: int) -> NearestNeighbors:
    model = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(X)),
        metric="cosine",
    )
    model.fit(X)
    return model


def find_nearest(
    df: pd.DataFrame,
    query_vector: np.ndarray,
    n_results: int,
) -> pd.DataFrame:
    """
    Return the n activities closest to the query vector.
    """
    if df.empty:
        return df.copy()

    enc_df, X = encode_activities(df)
    model = fit_knn(X, n_results)
    distances, indices = model.kneighbors(query_vector.reshape(1, -1))

    result = enc_df.iloc[indices[0]].copy()
    result["similarity_distance"] = distances[0]

    # Drop encoded columns — keep only original fields + distance
    drop_cols = ["region_enc","type_enc","cost_enc","rating_enc","dur_enc","season_enc"]
    result = result.drop(columns=drop_cols, errors="ignore")

    return result.reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────

def get_knn_suggestions(
    activities_df: pd.DataFrame,
    selected_region: str,
    selected_season: str,
    selected_type: str,
    max_budget: float,
    n_results: int = 10,
) -> pd.DataFrame:
    """
    PASS 1 — Suggest activities based on user criteria.
    Called after the user sets region / season / type / budget.
    """
    if activities_df.empty:
        return activities_df.copy()

    filtered = apply_hard_constraints(
        activities_df, selected_season, selected_region, max_budget
    )

    if filtered.empty:
        return pd.DataFrame()

    query = build_user_vector(selected_region, selected_season, selected_type, max_budget)

    return find_nearest(filtered, query, n_results)


def get_refined_recommendations(
    activities_df: pd.DataFrame,
    favourite_names: List[str],
    selected_region: str,
    selected_season: str,
    max_budget: float,
    n_results: int = 28,
) -> pd.DataFrame:
    """
    PASS 2 — Recommend more activities based on the user's chosen favourites.
    Called after the user picks their favourite suggestions.
    """
    if activities_df.empty:
        return activities_df.copy()

    favourites_df = activities_df[activities_df["name"].isin(favourite_names)].copy()
    if favourites_df.empty:
        return pd.DataFrame()

    # Pool = same region + season + budget, minus already chosen favourites
    pool = apply_hard_constraints(
        activities_df, selected_season, selected_region, max_budget
    )
    pool = pool[~pool["name"].isin(favourite_names)].copy()

    if pool.empty:
        return favourites_df.copy()

    refined_vector = build_refined_user_vector(favourites_df)

    return find_nearest(pool, refined_vector, n_results).reset_index(drop=True)


def build_week_programme(
    favourite_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
    n_days: int = 7,
) -> pd.DataFrame:
    """
    Build a day-by-day programme from favourites + recommendations.
    Each day gets a Morning, Afternoon, Evening, and Night slot.
    Adapts to the actual number of travel days (max 7).
    """
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][:n_days]
    slots = ["Morning","Afternoon","Evening","Night"]

    # Preferred slot per activity type
    SLOT_PREFERENCE = {
        "Hiking":      "Morning",
        "Skiing":      "Morning",
        "Adventure":   "Morning",
        "Sightseeing": "Afternoon",
        "Museum":      "Afternoon",
        "Nature":      "Afternoon",
        "Food":        "Evening",
        "Relaxation":  "Evening",
        "Wellness":    "Afternoon",
        "Shopping":    "Afternoon",
        "Nightlife":   "Night",
    }

    pool = pd.concat([favourite_df, recommended_df], ignore_index=True).drop_duplicates("name")
    if pool.empty:
        return pd.DataFrame()

    # Sort pool so favourites come first
    fav_names = list(favourite_df["name"]) if not favourite_df.empty else []
    pool["_is_fav"] = pool["name"].isin(fav_names)
    pool = pool.sort_values("_is_fav", ascending=False).drop(columns="_is_fav")

    # Build slot buckets using preferred slot per type
    slot_buckets: Dict[str, list] = {s: [] for s in slots}
    for _, row in pool.iterrows():
        preferred = SLOT_PREFERENCE.get(row["type"], "Afternoon")
        slot_buckets[preferred].append(row["name"])

    used: set = set()
    rows = []

    def pick(slot: str) -> str:
        for name in slot_buckets[slot]:
            if name not in used:
                used.add(name)
                return name
        # fallback: any unused activity
        for name in pool["name"]:
            if name not in used:
                used.add(name)
                return name
        return "Free time"

    for day in days:
        rows.append({
            "Day":       day,
            "Morning":   pick("Morning"),
            "Afternoon": pick("Afternoon"),
            "Evening":   pick("Evening"),
            "Night":     pick("Night"),
        })

    return pd.DataFrame(rows)
