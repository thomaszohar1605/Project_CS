import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

RATINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ratings.csv")

# List of keywords we look for in activity names.
# The first keyword found in the name is used.

# More specific keywords come first so they take priority.
# e.g. "City Bike Tour" → "bike" (not "tour")
# e.g. "Brewery Beer Tasting" → "tasting" (not "bar")
KEYWORDS = [
    "museum", "hike", "kayak", "bike", "ski", "swimming", "boat",
    "spa", "balloon", "climbing", "tasting", "dinner", "restaurant",
    "market", "garden", "gallery", "theatre", "cinema", "zoo",
    "library", "meditation", "bar", "club", "walk", "tour",
]


def extract_keyword(activity_name):
    """
    Look at the activity name and return the first matching keyword.
    For example:
      "Kunsthaus Museum"       → "museum"
      "Uetliberg Sunrise Hike" → "hike"
      "Thermalbad & Spa"       → "spa"
      "City Bike Tour"         → "bike"
    If no keyword matches, return "other".
    """
    name = activity_name.lower()
    for keyword in KEYWORDS:
        if keyword in name:
            return keyword
    return "other"


def save_rating(activity_name, category, duration_hours, keyword, rating):
    # Create the file with a header if it does not exist yet
    if not os.path.exists(RATINGS_FILE):
        df = pd.DataFrame(columns=["activity_name", "category",
                                    "duration_hours", "keyword", "rating"])
        df.to_csv(RATINGS_FILE, index=False)

    df = pd.read_csv(RATINGS_FILE)

    new_row = {
        "activity_name": activity_name,
        "category":      category,
        "duration_hours": duration_hours,
        "keyword":        keyword,
        "rating":         rating,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(RATINGS_FILE, index=False)


def predict_rating(activity_name, category, duration_hours, keyword):
    if not os.path.exists(RATINGS_FILE):
        return None
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 5:
        return None

    # Encode category as a number
    df["category_code"] = df["category"].astype("category").cat.codes
    all_categories = df["category"].astype("category").cat.categories.tolist()
    category_code = all_categories.index(category) if category in all_categories else -1

    # Encode keyword as a number
    df["keyword_code"] = df["keyword"].astype("category").cat.codes
    all_keywords = df["keyword"].astype("category").cat.categories.tolist()
    keyword_code = all_keywords.index(keyword) if keyword in all_keywords else -1

    X = df[["category_code", "duration_hours", "keyword_code"]]
    y = df["rating"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    new_activity = [[category_code, duration_hours, keyword_code]]
    new_activity_scaled = scaler.transform(new_activity)

    predicted_rating = model.predict(new_activity_scaled)
    return int(predicted_rating[0])


def get_neighbours(activity_name, category, duration_hours, keyword):
    """
    Return the 3 most similar activities the model used to make its prediction.
    Each neighbour is a dict: {"name": ..., "rating": ..., "keyword": ...}
    """
    if not os.path.exists(RATINGS_FILE):
        return []
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 5:
        return []

    # Encode category
    df["category_code"] = df["category"].astype("category").cat.codes
    all_categories = df["category"].astype("category").cat.categories.tolist()
    category_code = all_categories.index(category) if category in all_categories else -1

    # Encode keyword
    df["keyword_code"] = df["keyword"].astype("category").cat.codes
    all_keywords = df["keyword"].astype("category").cat.categories.tolist()
    keyword_code = all_keywords.index(keyword) if keyword in all_keywords else -1

    X = df[["category_code", "duration_hours", "keyword_code"]]
    y = df["rating"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    new_activity = [[category_code, duration_hours, keyword_code]]
    new_activity_scaled = scaler.transform(new_activity)

    # Get the 3 closest neighbours
    distances, indices = model.kneighbors(new_activity_scaled)

    neighbours = []
    for i in indices[0]:
        neighbours.append({
            "name":    df.iloc[i]["activity_name"],
            "rating":  int(df.iloc[i]["rating"]),
            "keyword": df.iloc[i]["keyword"],
        })

    return neighbours


def get_model_accuracy():
    if not os.path.exists(RATINGS_FILE):
        return None
    df = pd.read_csv(RATINGS_FILE)
    if len(df) < 10:
        return None

    df["category_code"] = df["category"].astype("category").cat.codes
    df["keyword_code"]  = df["keyword"].astype("category").cat.codes

    X = df[["category_code", "duration_hours", "keyword_code"]]
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
