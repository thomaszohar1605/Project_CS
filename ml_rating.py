# Machine-learning module using a real scikit-learn K-Nearest Neighbors model.

from __future__ import annotations

# Imports the relevant libraries 

import numpy as np
import pandas as pd

# Nearest Neighbors = Finds the most similar activities to the user's preferences.
# MinMaxScaler = Puts all feature values on the same scale before comparing.
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Category list 

# Activites rated in step 2 
# Each category's position in the feature vector 
CATEGORIES = [
    "Outdoor & Nature",
    "Culture & History",
    "Food & Drink",
    "Nightlife & Entertainment",
    "Relaxation & Wellness",
    "Adventure & Sports",
]

# Season list 

# The seasons from Step 1.
SEASONS = ["spring", "summer", "fall", "winter"]

# Feature column definitions 

# 15 feature columns total (total vector)
# 6 one-hot category flags
# 1 indoor/outdoor value
# 3 time-slot flags
# 1 normalised duration
# 4 season flags
FEATURE_COLS = [
    "feat_outdoor_nature",    
    "feat_culture_history",   
    "feat_food_drink",        
    "feat_nightlife",         
    "feat_wellness",          
    "feat_adventure",         
    "feat_is_outdoor",        
    "feat_is_morning",        
    "feat_is_afternoon",      
    "feat_is_evening",        
    "feat_duration",          
    "feat_season_spring",     
    "feat_season_summer",     
    "feat_season_fall",      
    "feat_season_winter",   
]

# Lookup: category name -> its one-hot feature column
# When the code needs to encode a category into the feature vector, instead of writing
CAT_TO_FEAT = {
    "Outdoor & Nature":          "feat_outdoor_nature",
    "Culture & History":         "feat_culture_history",
    "Food & Drink":              "feat_food_drink",
    "Nightlife & Entertainment": "feat_nightlife",
    "Relaxation & Wellness":     "feat_wellness",
    "Adventure & Sports":        "feat_adventure",
}

# Lookup: season name -> its feature column
#same as cat_to_feat but for season
SEASON_TO_FEAT = {
    "spring": "feat_season_spring",
    "summer": "feat_season_summer",
    "fall":   "feat_season_fall",
    "winter": "feat_season_winter",
}

# Time-of-day category 

# Defines which time slots are natural for each category.
# This is made in order to prevent having event you graded low, for instance nightlife when you graded it low it shoudn't be recommended 
CAT_TIME_PROFILE = {
    "Outdoor & Nature":          {"morning": 1.0, "afternoon": 1.0, "evening": 0.0},
    "Culture & History":         {"morning": 1.0, "afternoon": 1.0, "evening": 0.0},
    "Food & Drink":              {"morning": 1.0, "afternoon": 1.0, "evening": 1.0},
    "Nightlife & Entertainment": {"morning": 0.0, "afternoon": 0.0, "evening": 1.0},
    "Relaxation & Wellness":     {"morning": 1.0, "afternoon": 1.0, "evening": 0.5},
    "Adventure & Sports":        {"morning": 1.0, "afternoon": 1.0, "evening": 0.0},
}



# Feature engineering --------------------------------------------------------
# convert one csv row into 15 dimensional into numpy array 
   
# Start at 0
def _activity_to_vector(row: pd.Series) -> np.ndarray:
    feats = {col: 0.0 for col in FEATURE_COLS}
    feat_key = CAT_TO_FEAT.get(str(row.get("category", "")))
    if feat_key:
        feats[feat_key] = 1.0

    # Encode indoor/outdoor 
    setting = str(row.get("indoor_outdoor", "both")).strip().lower()
    feats["feat_is_outdoor"] = {"outdoor": 1.0, "both": 0.5}.get(setting, 0.0)

    # Encode time slots 
    slots_raw = str(row.get("time_slot", "")).lower()
    feats["feat_is_morning"]   = 1.0 if "morning"   in slots_raw else 0.0
    feats["feat_is_afternoon"] = 1.0 if "afternoon" in slots_raw else 0.0
    feats["feat_is_evening"]   = 1.0 if "evening"   in slots_raw else 0.0

    # Duration of 7 days then divide to get a value in [0, 1]
    try:
        days = float(row.get("max_useful_days", 3))
    except (ValueError, TypeError):
        days = 3.0
    feats["feat_duration"] = min(days, 7.0) / 7.0

    # Encode which seasons and activity available
    # CSV stores seasons as pipe-separated 
    seasons_raw = str(row.get("seasons", "")).lower()
    for season, feat_col in SEASON_TO_FEAT.items():
        feats[feat_col] = 1.0 if season in seasons_raw else 0.0

    return np.array([feats[c] for c in FEATURE_COLS])


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return an (N x 15) matrix — one row per activity in df.
    """
    return np.vstack([_activity_to_vector(row) for _, row in df.iterrows()])


# Weight function -------------------------------------------------------------

  # Non-linear weight from the 1-5 rating
#rating 1 -> 0.11   strongly downweights
#rating 2 -> 0.44
#rating 3 -> 1.00   neutral
#rating 4 -> 1.78
#rating 5 -> 2.78   strongly upweights

    Formula: (rating / 3) ** 2
def _weight(rating: float) -> float:
    return (float(rating) / 3.0) ** 2

# Build the user profile vector -------------------------------------------------------------

 
# Convert the 6 category ratings and the chosen season into one 15-dimensional user profile vector.

# Category preferences are weighted by (rating/3)^2. Time-of-day flags use per-category profilesto avoid recommending evening activities when Nightlife is rated low.

# The four season flags are set so the chosen season = 1.0 and all other seasons = 0.0, pulling the KNN profile toward activities that run in the selected season.
 

def _build_user_profile(prefs: dict, season: str) -> np.ndarray:
   
    weighted_vecs = []
    weights       = []

    for cat in CATEGORIES:
        rating = float(prefs.get(cat, 3))
        w      = _weight(rating)

        feats    = {col: 0.0 for col in FEATURE_COLS}
        feat_key = CAT_TO_FEAT.get(cat)
        if feat_key:
            feats[feat_key] = 1.0

        # Per-category time-of-day profile
        time_profile = CAT_TIME_PROFILE[cat]
        feats["feat_is_morning"]   = time_profile["morning"]
        feats["feat_is_afternoon"] = time_profile["afternoon"]
        feats["feat_is_evening"]   = time_profile["evening"]

        feats["feat_is_outdoor"] = 0.5   
        feats["feat_duration"]   = 0.5   

        # Set the chosen season flag to 1.0, all others to 0.0
        season_clean = season.strip().lower()
        for s, feat_col in SEASON_TO_FEAT.items():
            feats[feat_col] = 1.0 if s == season_clean else 0.0

        vec = np.array([feats[c] for c in FEATURE_COLS])
        weighted_vecs.append(vec * w)
        weights.append(w)

    total_weight = sum(weights)
    if total_weight == 0:
        return np.array([_weight(3)] * len(FEATURE_COLS))

    # Weighted average -> single point in 15-D feature space
    profile = np.sum(weighted_vecs, axis=0) / total_weight
    return profile



# Main KNN function -------------------------------------------------------------


# Fit a KNN model on the city's activities and rank them by cosine distance to the user's profile vector.

# The parameters used are 
    # - activities for the chosen city (city_df) = DataFrame
    # - {category_name: slider_rating (1-5)} (prefs) = dict
    # - chosen season: "spring" | "summer" | "fall" | "winter" (season) = str
    # - number of neighbours: (k)= int

# Returns
# city_df copy with extra columns:
    #'knn_score'  float [0, 1]  — similarity (1 = perfect match)
    #'knn_rank'   int           — rank (1 = best match)
# Sorted by knn_score descending 

def get_knn_ranked_activities(
    city_df: pd.DataFrame,
    prefs: dict,
    season: str = "summer",
    k: int = None,
) -> pd.DataFrame:
   
    # Guard: return unchanged copy if there are no activities to rank
    if city_df.empty:
        return city_df.copy()

    df = city_df.copy().reset_index(drop=True)
    n  = len(df)

    # ── 1. Build and scale the feature matrix ────────────────────────────────
    X        = _build_feature_matrix(df)
    scaler   = MinMaxScaler()
    # Scale each feature column independently to [0, 1]
    X_scaled = scaler.fit_transform(X)

    # ── 2. Build and scale the user profile vector ───────────────────────────
    profile        = _build_user_profile(prefs, season)
    # Use the same scaler so profile and activities share the same space
    profile_scaled = scaler.transform(profile.reshape(1, -1))

    # ── 3. Fit the KNN model ─────────────────────────────────────────────────
    # Default to ALL activities so we get a complete ranking
    k_actual = k if k is not None else n
    k_actual = min(k_actual, n)   # safety cap

    knn_model = NearestNeighbors(
        n_neighbors=k_actual,
        metric="cosine",    # standard metric for content-based recommenders
        algorithm="brute",  # required for cosine distance
    )
    knn_model.fit(X_scaled)

    # ── 4. Query: find the K nearest neighbours ──────────────────────────────
    distances, indices = knn_model.kneighbors(profile_scaled)
    distances = distances[0]   # flatten outer dimension
    indices   = indices[0]     # flatten outer dimension

    # ── 5. Convert cosine distance to similarity score ───────────────────────
    # distance 0 = identical, 2 = opposite -> similarity = 1 - distance / 2
    similarities = 1.0 - distances / 2.0

    # ── 6. Map scores back onto the DataFrame ────────────────────────────────
    knn_scores = np.zeros(n)
    knn_ranks  = np.zeros(n, dtype=int)

    for rank, (idx, sim) in enumerate(zip(indices, similarities), start=1):
        knn_scores[idx] = sim    # similarity score for this activity
        knn_ranks[idx]  = rank   # 1 = best match

    df["knn_score"] = knn_scores
    df["knn_rank"]  = knn_ranks

    # Return sorted from best match (highest score) to worst
    return df.sort_values("knn_score", ascending=False).reset_index(drop=True)
