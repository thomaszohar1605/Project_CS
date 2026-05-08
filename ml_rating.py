# ml_rating.py  — full replacement
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Feature columns that must exist in locations.csv after running generate_features.py
FEATURE_COLS = [
    "feat_outdoor", "feat_duration",
    "feat_morning", "feat_afternoon", "feat_evening",
    "feat_adventure", "feat_culture", "feat_food",
    "feat_nature", "feat_nightlife", "feat_wellness",
]

# ---------- legacy helpers kept so existing imports don't break ----------

KEYWORDS = [
    "museum", "hike", "kayak", "bike", "ski", "swimming", "boat",
    "spa", "balloon", "climbing", "tasting", "dinner", "restaurant",
    "market", "garden", "gallery", "theatre", "cinema", "zoo",
    "library", "meditation", "bar", "club", "walk", "tour",
]

RATINGS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ratings.csv"
)


def extract_keyword(activity_name: str) -> str:
    name = activity_name.lower()
    for kw in KEYWORDS:
        if kw in name:
            return kw
    return "other"


def save_rating(activity_name, category, duration_hours, keyword, rating):
    if not os.path.exists(RATINGS_FILE):
        pd.DataFrame(columns=[
            "activity_name", "category", "duration_hours", "keyword", "rating"
        ]).to_csv(RATINGS_FILE, index=False)
    df = pd.read_csv(RATINGS_FILE)
    df = pd.concat([df, pd.DataFrame([{
        "activity_name": activity_name,
        "category": category,
        "duration_hours": duration_hours,
        "keyword": keyword,
        "rating": rating,
    }])], ignore_index=True)
    df.to_csv(RATINGS_FILE, index=False)


def predict_rating(activity_name, category, duration_hours, keyword):
    return None   # not used in new flow


def get_neighbours(activity_name, category, duration_hours, keyword):
    return []     # not used in new flow


def get_model_accuracy():
    return None   # not used in new flow


# ---------- NEW: KNN-based itinerary builder ----------

def _weight(rating: int) -> float:
    """Non-linear weight: punishes low ratings, rewards high ones."""
    return (rating / 3.0) ** 2


def build_itinerary_knn(
    rated_activities: list[dict],   # [{"activity_name": str, "rating": int}, ...]
    candidate_df: pd.DataFrame,     # all activities for this city, with FEATURE_COLS
    n_neighbors: int = 40,
) -> list[str]:
    """
    Given a small set of rated activities, return all candidate activities
    ranked by KNN cosine similarity to the user's preference profile.

    Parameters
    ----------
    rated_activities : list of dicts with keys "activity_name" and "rating"
    candidate_df     : DataFrame of city activities, must include FEATURE_COLS
    n_neighbors      : how many neighbors the KNN considers

    Returns
    -------
    List of activity_name strings, best match first.
    """
    # 1. Build feature matrix for all candidates
    df = candidate_df.copy().reset_index(drop=True)

    # Guard: if feature columns are missing, fall back to shuffled order
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return df["activity_name"].sample(frac=1).tolist()

    X = df[FEATURE_COLS].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Build user preference vector from rated activities
    rated_map = {r["activity_name"]: r["rating"] for r in rated_activities}

    vecs, weights = [], []
    for _, row in df.iterrows():
        name = row["activity_name"]
        if name in rated_map:
            rating = rated_map[name]
            vecs.append(X_scaled[_])
            weights.append(_weight(rating))

    if not vecs:
        # No overlap between rated names and candidates — return random order
        return df["activity_name"].sample(frac=1).tolist()

    vecs = np.array(vecs)
    weights = np.array(weights)
    user_vector = np.average(vecs, axis=0, weights=weights)

    # 3. Fit KNN on all candidates and find nearest neighbors
    k = min(n_neighbors, len(df))
    knn = NearestNeighbors(metric="cosine", n_neighbors=k)
    knn.fit(X_scaled)

    _, indices = knn.kneighbors(user_vector.reshape(1, -1))
    ranked_names = df.iloc[indices[0]]["activity_name"].tolist()

    # 4. Append any activities not yet in the ranked list (tail)
    ranked_set = set(ranked_names)
    tail = [n for n in df["activity_name"] if n not in ranked_set]

    return ranked_names + tail