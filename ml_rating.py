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

# Standard scientific computing libraries
import numpy as np        # used for vector operations and matrix math
import pandas as pd       # used to handle the activity DataFrame

# scikit-learn: the real KNN model and scaler
from sklearn.neighbors import NearestNeighbors   # the core KNN algorithm
from sklearn.preprocessing import MinMaxScaler   # normalises feature values to [0, 1]

# ── Category list (must stay identical to functions.py) ───────────────
# These are the 6 activity types the user rates in Step 2.
# The order here defines the position of each category in the feature vector.
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

# Names of all 11 feature columns used in the KNN feature matrix.
# The first 6 are one-hot category flags; the last 5 are context features.
FEATURE_COLS = [
    "feat_outdoor_nature",    # one-hot: 1 if the activity is Outdoor & Nature
    "feat_culture_history",   # one-hot: 1 if Culture & History
    "feat_food_drink",        # one-hot: 1 if Food & Drink
    "feat_nightlife",         # one-hot: 1 if Nightlife & Entertainment
    "feat_wellness",          # one-hot: 1 if Relaxation & Wellness
    "feat_adventure",         # one-hot: 1 if Adventure & Sports
    "feat_is_outdoor",        # 0=indoor  0.5=both  1=outdoor
    "feat_is_morning",        # 1 if Morning is an allowed time slot
    "feat_is_afternoon",      # 1 if Afternoon is an allowed time slot
    "feat_is_evening",        # 1 if Evening is an allowed time slot
    "feat_duration",          # max_useful_days normalised to [0, 1]
]

# Lookup table: category name → its corresponding feature column name.
# Used when building feature vectors to set the correct one-hot flag.
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
    # Start with all features set to 0 — we will fill in the relevant ones below
    feats = {col: 0.0 for col in FEATURE_COLS}

    # One-hot encode the category: set its flag to 1, all others stay 0
    feat_key = CAT_TO_FEAT.get(str(row.get("category", "")))
    if feat_key:
        feats[feat_key] = 1.0

    # Encode whether the activity is indoors, outdoors, or both
    # outdoor=1.0, both=0.5, indoor=0.0 (default if value is unrecognised)
    setting = str(row.get("indoor_outdoor", "both")).strip().lower()
    feats["feat_is_outdoor"] = {"outdoor": 1.0, "both": 0.5}.get(setting, 0.0)

    # Encode which time slots are available for this activity
    # The CSV stores slots as e.g. "Morning|Afternoon" — we check for each keyword
    slots_raw = str(row.get("time_slot", "")).lower()
    feats["feat_is_morning"]   = 1.0 if "morning"   in slots_raw else 0.0
    feats["feat_is_afternoon"] = 1.0 if "afternoon" in slots_raw else 0.0
    feats["feat_is_evening"]   = 1.0 if "evening"   in slots_raw else 0.0

    # Normalise duration: cap at 7 days then divide by 7 to get a value in [0, 1]
    try:
        days = float(row.get("max_useful_days", 3))
    except (ValueError, TypeError):
        days = 3.0                              # fallback if value is missing or invalid
    feats["feat_duration"] = min(days, 7.0) / 7.0

    # Return the 11 feature values as a numpy array in the fixed FEATURE_COLS order
    return np.array([feats[c] for c in FEATURE_COLS])


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return an (N × 11) matrix — one row per activity in df.
    """
    # Convert every row to a vector and stack them into a 2D matrix
    # Shape: (number of activities in the city, 11 features)
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
    # Squaring (r/3) means:
    # - ratings below 3 contribute very little (downweighted)
    # - rating 3 is neutral (weight = 1.0)
    # - ratings above 3 contribute strongly (upweighted)
    # This makes strong preferences (1 or 5) much more influential than weak ones (2 or 4)
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
    # These lists will collect the weighted vectors before averaging
    weighted_vecs = []
    weights       = []

    for cat in CATEGORIES:
        # Get the user's slider rating for this category (default 3 = neutral)
        rating = float(prefs.get(cat, 3))
        w      = _weight(rating)   # convert rating to non-linear weight

        # Build the "ideal" feature vector for this category:
        # set its one-hot flag to 1, all other category flags stay 0
        feats = {col: 0.0 for col in FEATURE_COLS}
        feat_key = CAT_TO_FEAT.get(cat)
        if feat_key:
            feats[feat_key] = 1.0

        # Use neutral values for the context features (indoor/outdoor, slots, duration)
        # so the profile is defined purely by category preference, not time-of-day
        feats["feat_is_outdoor"]   = 0.5   # neither strongly indoor nor outdoor
        feats["feat_is_morning"]   = 1.0   # available at all times of day
        feats["feat_is_afternoon"] = 1.0
        feats["feat_is_evening"]   = 1.0
        feats["feat_duration"]     = 0.5   # medium duration

        vec = np.array([feats[c] for c in FEATURE_COLS])

        # Multiply the vector by its weight before adding to the pool
        weighted_vecs.append(vec * w)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight == 0:
        # Edge case: all weights are zero — return a neutral profile
        return np.array([_weight(3)] * len(FEATURE_COLS))

    # Divide the sum of weighted vectors by total weight → weighted average
    # This gives a single point in 11-D space representing the user's taste
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
    # Guard: return an unchanged copy if there are no activities to rank
    if city_df.empty:
        return city_df.copy()

    df = city_df.copy().reset_index(drop=True)
    n  = len(df)   # total number of activities for this city

    # ── 1. Build and scale the feature matrix ────────────────────────
    # X is an (N × 11) matrix — one row per activity, 11 features per row
    X      = _build_feature_matrix(df)
    scaler = MinMaxScaler()
    # fit_transform scales each feature column independently to [0, 1]
    # so that no single feature dominates the distance calculation
    X_scaled = scaler.fit_transform(X)

    # ── 2. Build and scale the user profile vector ───────────────────
    # The profile is a single point in the same 11-D space as the activities
    profile        = _build_user_profile(prefs)
    # Use the SAME scaler (already fitted on activities) to transform the profile
    # so profile and activities live in the same scaled coordinate system
    profile_scaled = scaler.transform(
        profile.reshape(1, -1)   # reshape from (11,) to (1, 11) as scaler expects 2D
    )

    # ── 3. Fit the KNN model ─────────────────────────────────────────
    # k defaults to ALL activities so we get a complete ranking, not just top-k
    k_actual = k if k is not None else n
    k_actual = min(k_actual, n)   # safety cap: k cannot exceed number of points

    knn_model = NearestNeighbors(
        n_neighbors=k_actual,
        metric="cosine",    # cosine distance: standard metric for recommender systems
        algorithm="brute",  # brute-force exact search, required for cosine metric
    )
    # Train the model: store all scaled activity vectors internally
    knn_model.fit(X_scaled)

    # ── 4. Query: find the K nearest neighbours ──────────────────────
    # Ask the model: "which activities are most similar to this user's profile?"
    # Returns distances (how far) and indices (which rows in X_scaled)
    distances, indices = knn_model.kneighbors(profile_scaled)
    # distances shape: (1, k_actual) — cosine distance ∈ [0, 2]
    # indices   shape: (1, k_actual) — positions in X_scaled, sorted closest first

    distances = distances[0]   # flatten: remove the outer dimension
    indices   = indices[0]     # flatten: remove the outer dimension

    # ── 5. Convert cosine distance → similarity score ────────────────
    # Cosine distance 0 = identical direction, 2 = opposite direction
    # We convert to similarity: 1 = perfect match, 0 = completely opposite
    # Formula: similarity = 1 − distance / 2
    similarities = 1.0 - distances / 2.0

    # ── 6. Map scores back onto the DataFrame ────────────────────────
    # indices[0] is the closest activity, indices[1] is second closest, etc.
    # We write each activity's similarity score and rank back into the DataFrame
    knn_scores = np.zeros(n)
    knn_ranks  = np.zeros(n, dtype=int)

    for rank, (idx, sim) in enumerate(zip(indices, similarities), start=1):
        knn_scores[idx] = sim    # similarity score for this activity
        knn_ranks[idx]  = rank   # 1 = best match, n = worst match

    # Add the new columns to the DataFrame
    df["knn_score"] = knn_scores
    df["knn_rank"]  = knn_ranks

    # Return the DataFrame sorted from best match (highest score) to worst
    return df.sort_values("knn_score", ascending=False).reset_index(drop=True)
