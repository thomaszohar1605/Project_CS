import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

RATINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings.csv")


def save_rating(activity_name, category, duration_hours, price_chf, rating):
    if not os.path.exists(RATINGS_FILE):
        df = pd.DataFrame(columns=["activity_name", "category",
                                    "duration_hours", "price_chf", "rating"])
        df.to_csv(RATINGS_FILE, index=False)
    df = pd.read_csv(RATINGS_FILE)
    new_row = {
        "activity_name": activity_name,
        "category": category,
        "duration_hours": duration_hours,
        "price_chf": price_chf,
        "rating": rating
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(RATINGS_FILE, index=False)


def predict_rating(activity_name, category, duration_hours, price_chf):
    if not os.path.exists(RATINGS_FILE):
        return None
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 5:
        return None
    df["category_code"] = df["category"].astype("category").cat.codes
    all_categories = df["category"].astype("category").cat.categories.tolist()
    if category in all_categories:
        category_code = all_categories.index(category)
    else:
        category_code = -1
    X = df[["category_code", "duration_hours", "price_chf"]]
    y = df["rating"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)
    new_activity = [[category_code, duration_hours, price_chf]]
    new_activity_scaled = scaler.transform(new_activity)
    predicted_rating = model.predict(new_activity_scaled)
    return int(predicted_rating[0])


def get_neighbours(activity_name, category, duration_hours, price_chf):
    """
    Return the 3 most similar activities that the KNN model used
    to make its prediction — so we can show the user WHY we recommended a score.
    Returns a list of dicts: [{"name": ..., "rating": ...}, ...]
    Returns an empty list if there is not enough data yet.
    """
    if not os.path.exists(RATINGS_FILE):
        return []
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 5:
        return []

    # Encode the category as a number (same as in predict_rating)
    df["category_code"] = df["category"].astype("category").cat.codes
    all_categories = df["category"].astype("category").cat.categories.tolist()
    if category in all_categories:
        category_code = all_categories.index(category)
    else:
        category_code = -1

    X = df[["category_code", "duration_hours", "price_chf"]]
    y = df["rating"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    new_activity = [[category_code, duration_hours, price_chf]]
    new_activity_scaled = scaler.transform(new_activity)

    # kneighbors() returns the positions (indices) of the 3 closest activities
    distances, indices = model.kneighbors(new_activity_scaled)

    neighbours = []
    for i in indices[0]:
        neighbours.append({
            "name":   df.iloc[i]["activity_name"],
            "rating": int(df.iloc[i]["rating"]),
        })

    return neighbours


def get_model_accuracy():
    if not os.path.exists(RATINGS_FILE):
        return None
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 10:
        return None
    df["category_code"] = df["category"].astype("category").cat.codes
    X = df[["category_code", "duration_hours", "price_chf"]]
    y = df["rating"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return round(accuracy * 100, 1)
