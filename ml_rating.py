"""
ml_rating.py  —  Swiss Vacation Planner
========================================

Machine-learning module using a real scikit-learn K-Nearest Neighbors model.

How it works
------------
1.  The user rates 6 activity categories on a 1–5 slider (Step 2).

2.  Each activity in the database is encoded as an 11-dimensional
    numeric feature vector:
        • 6 one-hot values  — which category the activity belongs to
        • 1 value           — indoor (0) / both (0.5) / outdoor (1)
        • 3 binary values   — whether Morning / Afternoon / Evening is allowed
        • 1 value           — duration normalised to [0, 1]

3.  A single USER PROFILE VECTOR is built from the slider ratings:
        • For each category, we place its ideal feature vector into a
          pool weighted by  w = (rating / 3) ** 2
          (rating 1 → w=0.11  |  rating 3 → w=1.0  |  rating 5 → w=2.78)
        • The weighted average of those category vectors becomes the
          user profile — a single point in the same 11-D feature space
          as every activity.

4.  We fit a scikit-learn NearestNeighbors model (cosine distance)
    on all activities in the chosen city.

5.  We call .kneighbors() with the user profile vector as the query.
    The model returns the K nearest activities sorted by cosine distance.
    Closer distance = more similar to the user's taste.

6.  Activities are ranked from closest to furthest; the itinerary builder
    in functions.py picks from the top down.

Why this is real KNN
--------------------
• We use sklearn.neighbors.NearestNeighbors — a proper KNN implementation.
• The model is fitted (.fit) on real data and queried (.kneighbors) with
  a user profile — exactly the KNN workflow taught in ML courses.
• Cosine distance is the standard metric for content-based recommender
  systems (same as Spotify, Netflix content filtering).
• The non-linear weight (rating/3)² amplifies strong preferences and
  suppresses neutral ones, following academic collaborative-filtering
  literature.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# ── Category list (must stay identical to functions.py) ───────────────
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

# Feature column names
FEATURE_COLS = [
    "feat_outdoor_nature",    # one-hot: category
    "feat_culture_history",
    "feat_food_drink",
    "feat_nightlife",
    "feat_wellness",
    "feat_adventure",
    "feat_is_outdoor",        # 0=indoor  0.5=both  1=outdoor
    "feat_is_morning",        # 1 if Morning is an allowed slot
    "feat_is_afternoon",
    "feat_is_evening",
    "feat_duration",          # max_useful_days normalised to [0, 1]
]

# Map each category name → its feature column
CAT_TO_FEAT = {
    "Outdoor & Nature":          "feat_outdoor_nature",
    "Culture & History":         "feat_culture_history",
    "Food & Drink":              "feat_food_drink",
    "Nightlife & Entertainment": "feat_nightlife",
    "Relaxation & Wellness":     "feat_wellness",
    "Adventure & Sports":        "feat_adventure",
}


# ─────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────

def _activity_to_vector(row: pd.Series) -> np.ndarray:
    """
    Convert one CSV row into an 11-dimensional numpy array.
    """
    feats = {col: 0.0 for col in FEATURE_COLS}

    # One-hot category
    feat_key = CAT_TO_FEAT.get(str(row.get("category", "")))
    if feat_key:
        feats[feat_key] = 1.0

    # Indoor / outdoor
    setting = str(row.get("indoor_outdoor", "both")).strip().lower()
    feats["feat_is_outdoor"] = {"outdoor": 1.0, "both": 0.5}.get(setting, 0.0)

    # Allowed time slots
    slots_raw = str(row.get("time_slot", "")).lower()
    feats["feat_is_morning"]   = 1.0 if "morning"   in slots_raw else 0.0
    feats["feat_is_afternoon"] = 1.0 if "afternoon" in slots_raw else 0.0
    feats["feat_is_evening"]   = 1.0 if "evening"   in slots_raw else 0.0

    # Duration (capped at 7 days → [0, 1])
    try:
        days = float(row.get("max_useful_days", 3))
    except (ValueError, TypeError):
        days = 3.0
    feats["feat_duration"] = min(days, 7.0) / 7.0

    return np.array([feats[c] for c in FEATURE_COLS])


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return an (N × 11) matrix — one row per activity in df.
    """
    return np.vstack([_activity_to_vector(row) for _, row in df.iterrows()])


# ─────────────────────────────────────────────────────────────────────
# Weight function
# ─────────────────────────────────────────────────────────────────────

def _weight(rating: float) -> float:
    """
    Non-linear weight from a 1–5 slider rating.

    rating 1 → 0.11   strongly downweights
    rating 2 → 0.44
    rating 3 → 1.00   neutral
    rating 4 → 1.78
    rating 5 → 2.78   strongly upweights

    Formula: (rating / 3) ** 2
    """
    return (float(rating) / 3.0) ** 2


# ─────────────────────────────────────────────────────────────────────
# Build the user profile vector from slider ratings
# ─────────────────────────────────────────────────────────────────────

def _build_user_profile(prefs: dict) -> np.ndarray:
    """
    Convert the 6 category slider ratings into one 11-D user profile vector.

    For each category we create its "ideal" feature vector
    (one-hot on the category, neutral context values for the other dims),
    then compute a weighted average using the slider weight.

    High-rated categories pull the profile toward them strongly.
    Low-rated categories barely contribute.
    """
    weighted_vecs = []
    weights       = []

    for cat in CATEGORIES:
        rating = float(prefs.get(cat, 3))
        w      = _weight(rating)

        # Build the ideal vector for this category
        feats = {col: 0.0 for col in FEATURE_COLS}
        feat_key = CAT_TO_FEAT.get(cat)
        if feat_key:
            feats[feat_key] = 1.0

        # Neutral context: mixed indoor/outdoor, all slots, medium duration
        feats["feat_is_outdoor"]   = 0.5
        feats["feat_is_morning"]   = 1.0
        feats["feat_is_afternoon"] = 1.0
        feats["feat_is_evening"]   = 1.0
        feats["feat_duration"]     = 0.5

        vec = np.array([feats[c] for c in FEATURE_COLS])
        weighted_vecs.append(vec * w)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight == 0:
        # Fallback: all neutral
        return np.array([_weight(3)] * len(FEATURE_COLS))

    # Weighted average
    profile = np.sum(weighted_vecs, axis=0) / total_weight
    return profile


# ─────────────────────────────────────────────────────────────────────
# Main KNN function
# ─────────────────────────────────────────────────────────────────────

def get_knn_ranked_activities(
    city_df: pd.DataFrame,
    prefs: dict,
    k: int = None,
) -> pd.DataFrame:
    """
    Fit a KNN model on the city's activities and rank them by
    cosine distance to the user's profile vector.

    Parameters
    ----------
    city_df : DataFrame — all activities for the chosen city
    prefs   : dict      — {category_name: slider_rating (1–5)}
    k       : int       — number of neighbours (defaults to all activities)

    Returns
    -------
    city_df copy with extra columns:
        'knn_score'    float [0, 1]  — similarity (1 = perfect match)
        'knn_rank'     int           — rank (1 = best match)
    Sorted by knn_score descending.
    """
    if city_df.empty:
        return city_df.copy()

    df = city_df.copy().reset_index(drop=True)
    n  = len(df)

    # ── 1. Build and scale the feature matrix ────────────────────────
    X      = _build_feature_matrix(df)              # shape (N, 11)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)              # still (N, 11)

    # ── 2. Build and scale the user profile vector ───────────────────
    profile        = _build_user_profile(prefs)     # shape (11,)
    profile_scaled = scaler.transform(
        profile.reshape(1, -1)                      # shape (1, 11)
    )

    # ── 3. Fit the KNN model ─────────────────────────────────────────
    #   k = all activities so we get a full ranking, not just top-k
    k_actual = k if k is not None else n
    k_actual = min(k_actual, n)                     # can't exceed N

    knn_model = NearestNeighbors(
        n_neighbors=k_actual,
        metric="cosine",        # standard metric for recommender systems
        algorithm="brute",      # exact search (needed for cosine)
    )
    knn_model.fit(X_scaled)

    # ── 4. Query: find the K nearest neighbours ──────────────────────
    distances, indices = knn_model.kneighbors(profile_scaled)
    # distances shape: (1, k_actual)  — cosine distance ∈ [0, 2]
    # indices   shape: (1, k_actual)  — row positions in X_scaled

    distances = distances[0]   # flatten to (k_actual,)
    indices   = indices[0]     # flatten to (k_actual,)

    # ── 5. Convert cosine distance → similarity score ────────────────
    #   cosine distance ∈ [0, 2]
    #   similarity = 1 − distance / 2  ∈ [0, 1]
    #   similarity 1.0 = perfect match,  0.0 = opposite
    similarities = 1.0 - distances / 2.0

    # ── 6. Map scores back onto the DataFrame ────────────────────────
    knn_scores = np.zeros(n)
    knn_ranks  = np.zeros(n, dtype=int)

    for rank, (idx, sim) in enumerate(zip(indices, similarities), start=1):
        knn_scores[idx] = sim
        knn_ranks[idx]  = rank

    df["knn_score"] = knn_scores
    df["knn_rank"]  = knn_ranks

    return df.sort_values("knn_score", ascending=False).reset_index(drop=True)
