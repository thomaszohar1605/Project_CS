"""
ml_rating.py
============
Real machine learning module using K-Nearest Neighbors (KNN).

How it works:
- The user rates 6 activity categories on a scale of 1 to 5.
- Each activity in the CSV belongs to one of these categories.
- We use KNN to find the activities most compatible with the user's preferences,
  by comparing the user's preference vector to the feature vector of each activity.
- Activities are ranked by their predicted compatibility score (1-5).
- The itinerary builder then picks the highest-scoring activities first.
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

# The 6 categories — same order everywhere in the app
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

# One-hot encoding: for each category, its position in the CATEGORIES list
CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}


def build_feature_vector(category: str, indoor_outdoor: str, max_useful_days: float) -> list:
    """
    Build a numeric feature vector for one activity.

    The vector has:
      - 6 one-hot values for category (one 1, rest 0)
      - 1 value for indoor_outdoor  (0 = indoor, 0.5 = both, 1 = outdoor)
      - 1 value for duration (normalised 0-1, capped at 7 days)

    Total length: 8
    """
    # One-hot category
    one_hot = [0.0] * len(CATEGORIES)
    idx = CATEGORY_INDEX.get(category, -1)
    if idx >= 0:
        one_hot[idx] = 1.0

    # Indoor / outdoor encoding
    setting = str(indoor_outdoor).strip().lower()
    if setting == "outdoor":
        io_val = 1.0
    elif setting == "both":
        io_val = 0.5
    else:  # indoor
        io_val = 0.0

    # Duration (normalised, cap at 7)
    duration_norm = min(float(max_useful_days), 7.0) / 7.0

    return one_hot + [io_val, duration_norm]


def build_preference_vector(prefs: dict) -> np.ndarray:
    """
    Turn the user's slider ratings into a feature vector in the same space
    as the activity vectors, so KNN can compare them.

    prefs = {"Outdoor & Nature": 4, "Culture & History": 2, ...}

    We set io_val and duration_norm to neutral (0.5) because the user
    rates categories, not indoor/outdoor or duration directly.
    """
    one_hot = [prefs.get(cat, 3) / 5.0 for cat in CATEGORIES]
    io_val = 0.5        # neutral
    duration_norm = 0.5  # neutral
    return np.array(one_hot + [io_val, duration_norm])


def score_activities(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    """
    Use KNN to assign a compatibility score (1-5) to every activity.

    We build a synthetic training set where each category has a few
    "ideal" examples with known scores derived from the user's preferences.
    KNN then generalises to all real activities.

    Returns df with a new column "ml_score" (float, higher = better fit).
    """
    df = df.copy()

    # --- Build synthetic training data from user preferences ---
    # For each category, we create 3 synthetic examples:
    #   1. Pure indoor version
    #   2. Pure outdoor version
    #   3. "Both" version
    # Their target score = the user's preference for that category.
    X_train = []
    y_train = []

    for cat, score in prefs.items():
        for io_val in [0.0, 0.5, 1.0]:
            one_hot = [0.0] * len(CATEGORIES)
            idx = CATEGORY_INDEX.get(cat, -1)
            if idx >= 0:
                one_hot[idx] = 1.0
            vec = one_hot + [io_val, 0.5]
            X_train.append(vec)
            y_train.append(float(score))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # --- Build feature vectors for all real activities ---
    feature_rows = []
    for _, row in df.iterrows():
        vec = build_feature_vector(
            category=str(row.get("category", "")),
            indoor_outdoor=str(row.get("indoor_outdoor", "both")),
            max_useful_days=row.get("max_useful_days", 3),
        )
        feature_rows.append(vec)

    X_activities = np.array(feature_rows)

    # --- Scale everything together ---
    all_X = np.vstack([X_train, X_activities])
    scaler = MinMaxScaler()
    all_X_scaled = scaler.fit_transform(all_X)

    X_train_scaled = all_X_scaled[:len(X_train)]
    X_act_scaled   = all_X_scaled[len(X_train):]

    # --- Fit KNN and predict scores for real activities ---
    k = min(5, len(X_train))
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(X_train_scaled, y_train)

    scores = knn.predict(X_act_scaled)
    df["ml_score"] = scores

    return df


def get_top_activities(df: pd.DataFrame, prefs: dict, n: int = 30) -> pd.DataFrame:
    """
    Score all activities with the ML model and return the top-n,
    sorted from highest to lowest compatibility score.
    """
    scored = score_activities(df, prefs)
    scored = scored.sort_values("ml_score", ascending=False)
    return scored.head(n).reset_index(drop=True)


def explain_score(category: str, prefs: dict) -> str:
    """
    Return a short human-readable reason why this activity was chosen.
    """
    score = prefs.get(category, 3)
    if score >= 4:
        return f"Great match — you love {category}! ({'⭐' * score})"
    elif score == 3:
        return f"Good fit — you're interested in {category}."
    else:
        return f"Included for variety ({category})."
