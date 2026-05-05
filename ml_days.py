"""
ml_days.py
----------
A simple Linear Regression model that predicts how many days (1 to 7)
a user should spend in Switzerland based on their interests.

Each interest is scored from 0 (not interested) to 5 (very interested).
The model learns from a small example dataset, then predicts for new users.

Libraries used:
  - numpy    : to work with numbers and arrays
  - pandas   : to organise data in a table (like Excel)
  - sklearn  : to build and evaluate the machine learning model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =============================================================================
# STEP 1 — Create a small example dataset
# =============================================================================
# In real life this would come from a database of past users.
# Here we invent 20 example users so the model has something to learn from.
#
# Each row = one user
# Each column = how interested they are in that activity (score 0 to 5)
# Last column = how many days they actually spent in Switzerland
# =============================================================================

data = {
    # ---------- Activity preferences (0 = no interest, 5 = love it) ----------
    "outdoor_nature":          [5, 4, 1, 2, 5, 3, 0, 4, 2, 5, 1, 3, 4, 0, 5, 2, 3, 4, 1, 5],
    "culture_history":         [2, 3, 5, 4, 1, 4, 5, 2, 3, 1, 4, 5, 2, 5, 0, 3, 4, 1, 5, 2],
    "food_drink":              [3, 2, 4, 5, 2, 3, 4, 1, 5, 2, 3, 4, 1, 3, 4, 5, 2, 3, 4, 1],
    "nightlife_entertainment": [1, 2, 3, 5, 0, 2, 4, 3, 5, 1, 2, 3, 5, 4, 1, 4, 3, 2, 5, 0],
    "relaxation_wellness":     [4, 5, 2, 1, 3, 5, 2, 4, 1, 3, 5, 2, 3, 1, 4, 2, 5, 3, 1, 4],
    "adventure_sports":        [5, 4, 1, 0, 5, 2, 1, 5, 0, 4, 2, 1, 4, 0, 5, 1, 2, 5, 0, 4],

    # ---------- Time-of-day preferences (0 = never, 5 = always) ----------
    "morning":                 [5, 4, 3, 1, 5, 4, 2, 5, 1, 4, 3, 2, 5, 1, 4, 3, 5, 2, 1, 5],
    "afternoon":               [4, 5, 4, 3, 4, 3, 4, 4, 3, 5, 4, 3, 4, 3, 5, 4, 3, 4, 3, 4],
    "evening":                 [2, 3, 4, 5, 1, 3, 5, 2, 5, 2, 4, 5, 3, 5, 2, 4, 3, 2, 5, 1],

    # ---------- Target variable: ideal number of days (1 to 7) ----------
    "ideal_days":              [7, 6, 4, 3, 7, 5, 3, 6, 4, 6, 4, 5, 6, 3, 7, 4, 5, 6, 3, 7],
}

# Put the data into a pandas DataFrame (like a spreadsheet)
df = pd.DataFrame(data)

print("=" * 55)
print("Our example dataset (first 5 rows):")
print("=" * 55)
print(df.head())
print(f"\nTotal rows in dataset: {len(df)}")
print()


# =============================================================================
# STEP 2 — Separate features (X) and target (y)
# =============================================================================
# X = the inputs the model uses to make its prediction
# y = the answer we want the model to learn to predict
# =============================================================================

# X contains all columns except "ideal_days"
X = df.drop(columns=["ideal_days"])

# y is just the "ideal_days" column
y = df["ideal_days"]

print("=" * 55)
print("Features used to predict (X):")
print("=" * 55)
print(list(X.columns))
print()
print("What we want to predict (y): ideal_days")
print()


# =============================================================================
# STEP 3 — Split into training and testing data
# =============================================================================
# We keep 80% of the data to TRAIN the model (so it can learn)
# We keep 20% to TEST the model (to check if it's accurate)
# This way we don't cheat by testing on data it already saw
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to testing
    random_state=42     # fixed number so results are the same every run
)

print("=" * 55)
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")
print()


# =============================================================================
# STEP 4 — Create and train the Linear Regression model
# =============================================================================
# "Training" means the model looks at the training data and finds the best
# straight line (formula) that connects the features to the ideal_days.
#
# After training, the model has learned something like:
#   ideal_days = 0.4 × outdoor + 0.2 × culture + 0.1 × morning + ... + 2.1
# =============================================================================

model = LinearRegression()
model.fit(X_train, y_train)   # this is where the learning happens

print("=" * 55)
print("Model trained successfully!")
print()

# Show what the model learned (the "weights" for each feature)
print("What the model learned (importance of each feature):")
print("-" * 55)
for feature, weight in zip(X.columns, model.coef_):
    direction = "↑ more days" if weight > 0 else "↓ fewer days"
    print(f"  {feature:<28} {weight:+.3f}  ({direction})")
print(f"\n  Base value (intercept): {model.intercept_:.2f}")
print()


# =============================================================================
# STEP 5 — Evaluate how accurate the model is on the test data
# =============================================================================
# Mean Absolute Error (MAE) tells us how far off our predictions are on average.
# For example, MAE = 0.8 means we're off by less than 1 day on average — good!
# =============================================================================

y_predicted = model.predict(X_test)

# Clip predictions to stay between 1 and 7 (our valid range)
y_predicted_clipped = np.clip(y_predicted, 1, 7)
y_predicted_clipped = np.round(y_predicted_clipped).astype(int)

mae = mean_absolute_error(y_test, y_predicted_clipped)

print("=" * 55)
print("Model accuracy on test data:")
print("=" * 55)
print(f"  Mean Absolute Error: {mae:.2f} days")
print(f"  (On average, our prediction is off by {mae:.2f} days)")
print()


# =============================================================================
# STEP 6 — Predict for a new user
# =============================================================================
# Now we use the trained model to predict how many days a new user should stay.
# We just fill in their preference scores (0 to 5) and call model.predict().
# =============================================================================

# Example: a user who loves outdoor activities and adventure
new_user = {
    "outdoor_nature":          [5],   # loves it
    "culture_history":         [2],   # a little
    "food_drink":              [3],   # moderate
    "nightlife_entertainment": [1],   # not really
    "relaxation_wellness":     [2],   # a little
    "adventure_sports":        [5],   # loves it
    "morning":                 [5],   # loves mornings
    "afternoon":               [4],   # likes afternoons
    "evening":                 [2],   # not an evening person
}

new_user_df = pd.DataFrame(new_user)

# Make the prediction
raw_prediction = model.predict(new_user_df)[0]

# Clip to valid range (1 to 7) and round to nearest whole number
final_prediction = int(np.clip(round(raw_prediction), 1, 7))

print("=" * 55)
print("Predicting for a new user:")
print("=" * 55)
print("  Outdoor & Nature       : 5/5 (loves it)")
print("  Culture & History      : 2/5 (a little)")
print("  Food & Drink           : 3/5 (moderate)")
print("  Nightlife              : 1/5 (not really)")
print("  Relaxation & Wellness  : 2/5 (a little)")
print("  Adventure & Sports     : 5/5 (loves it)")
print("  Morning person         : 5/5")
print("  Afternoon person       : 4/5")
print("  Evening person         : 2/5")
print()
print(f"  Raw model output : {raw_prediction:.2f}")
print(f"  After clipping   : {final_prediction} days")
print()
print(f"  ✅ Recommended trip length: {final_prediction} days in Switzerland")
print()


# =============================================================================
# BONUS — A reusable function you can call from your Streamlit app
# =============================================================================

def predict_trip_days(outdoor, culture, food, nightlife, relaxation, adventure,
                      morning, afternoon, evening):
    """
    Predict how many days a user should spend in Switzerland.

    Parameters (all scores from 0 to 5):
        outdoor      : interest in Outdoor & Nature
        culture      : interest in Culture & History
        food         : interest in Food & Drink
        nightlife    : interest in Nightlife & Entertainment
        relaxation   : interest in Relaxation & Wellness
        adventure    : interest in Adventure & Sports
        morning      : preference for morning activities
        afternoon    : preference for afternoon activities
        evening      : preference for evening activities

    Returns:
        An integer between 1 and 7 (the recommended number of days)
    """

    # Put the user's scores into a DataFrame
    user_input = pd.DataFrame({
        "outdoor_nature":          [outdoor],
        "culture_history":         [culture],
        "food_drink":              [food],
        "nightlife_entertainment": [nightlife],
        "relaxation_wellness":     [relaxation],
        "adventure_sports":        [adventure],
        "morning":                 [morning],
        "afternoon":               [afternoon],
        "evening":                 [evening],
    })

    # Ask the model for a prediction
    raw = model.predict(user_input)[0]

    # Round and clip to stay between 1 and 7
    result = int(np.clip(round(raw), 1, 7))

    return result


# Quick test of the function
days = predict_trip_days(
    outdoor=4, culture=3, food=5, nightlife=2,
    relaxation=4, adventure=2,
    morning=3, afternoon=5, evening=4
)
print("=" * 55)
print(f"Function test → predicted trip length: {days} days")
print("=" * 55)
