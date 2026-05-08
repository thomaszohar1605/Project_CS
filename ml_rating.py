from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "feat_outdoor", "feat_duration",
    "feat_morning", "feat_afternoon", "feat_evening",
    "feat_adventure", "feat_culture", "feat_food",
    "feat_nature", "feat_nightlife", "feat_wellness",
]

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


def save_rating(username, activity_name, category, duration_hours, keyword, rating):
    # Create the file with a header if it does not exist yet
    if not os.path.exists(RATINGS_FILE):
        pd.DataFrame(columns=[
            "username", "activity_name", "category", "duration_hours", "keyword", "rating"
        ]).to_csv(RATINGS_FILE, index=False)
    df = pd.read_csv(RATINGS_FILE)
    df = pd.concat([df, pd.DataFrame([{
        "username":       username,
        "activity_name":  activity_name,
        "category":       category,
        "duration_hours": duration_hours,
        "keyword":        keyword,
        "rating":         rating,
    }])], ignore_index=True)
    df.to_csv(RATINGS_FILE, index=False)


def load_user_ratings(username):
    """
    Read ratings.csv and return only the rows for this user.
    Returns a list of dicts: [{"activity_name": ..., "rating": ...}, ...]
    Returns an empty list if the user has no past ratings.
    """
    if not os.path.exists(RATINGS_FILE):
        return []
    df = pd.read_csv(RATINGS_FILE)
    if "username" not in df.columns:
        return []
    user_rows = df[df["username"] == username]
    return user_rows[["activity_name", "rating"]].to_dict("records")


def predict_rating(activity_name, category, duration_hours, keyword):
    return None


def get_neighbours(activity_name, category, duration_hours, keyword):
    return []


def get_model_accuracy():
    return None


def build_itinerary_knn(
    rated_activities: list[dict],
    candidate_df: pd.DataFrame,
    n_neighbors: int = 40,
) -> list[str]:
    """
    Rank candidate activities by how well they match the user's preferences.

    Key idea: instead of a weighted average (which collapses low/high ratings
    toward the same point when features are similar), we compute two separate
    vectors — a LIKE vector and a DISLIKE vector — and score each candidate as:

        score = cosine_similarity(candidate, like_vector)
                - cosine_similarity(candidate, dislike_vector)

    This means activities similar to what the user liked rank HIGH, and
    activities similar to what the user disliked rank LOW — even when the
    underlying feature vectors are nearly identical across the candidate pool.
    """
    df = candidate_df.copy().reset_index(drop=True)

    # Guard: fall back to shuffled order if feature columns are missing
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return df["activity_name"].sample(frac=1).tolist()

    X = df[FEATURE_COLS].fillna(0).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    rated_map = {r["activity_name"]: r["rating"] for r in rated_activities}

    like_vecs    = []   # ratings 4–5
    dislike_vecs = []   # ratings 1–2
    neutral_vecs = []   # rating 3

    for pos, row in enumerate(df.itertuples(index=False)):
        name = row.activity_name
        if name not in rated_map:
            continue
        rating = rated_map[name]
        vec = X_scaled[pos]

        if rating >= 4:
            # Weight higher ratings more strongly
            w = (rating - 3) ** 2      # 4 → 1.0,  5 → 4.0
            like_vecs.extend([vec] * int(w * 10))
        elif rating <= 2:
            w = (3 - rating) ** 2      # 2 → 1.0,  1 → 4.0
            dislike_vecs.extend([vec] * int(w * 10))
        else:
            neutral_vecs.append(vec)

    # If no ratings matched candidates at all, return shuffled
    if not like_vecs and not dislike_vecs and not neutral_vecs:
        return df["activity_name"].sample(frac=1).tolist()

    # Build like / dislike vectors (fall back to neutral if one side is empty)
    if like_vecs:
        like_vector = np.mean(like_vecs, axis=0)
    elif neutral_vecs:
        like_vector = np.mean(neutral_vecs, axis=0)
    else:
        like_vector = None

    if dislike_vecs:
        dislike_vector = np.mean(dislike_vecs, axis=0)
    else:
        dislike_vector = None

    # Score every candidate
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    scores = []
    for pos in range(len(df)):
        vec = X_scaled[pos]
        score = 0.0
        if like_vector is not None:
            score += cosine_sim(vec, like_vector)
        if dislike_vector is not None:
            score -= cosine_sim(vec, dislike_vector)
        scores.append(score)

    # Sort candidates best → worst
    order = np.argsort(scores)[::-1]
    ranked_names = df.iloc[order]["activity_name"].tolist()
    return ranked_names