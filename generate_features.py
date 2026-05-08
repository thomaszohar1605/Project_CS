# generate_features.py  — run once from your project folder
import pandas as pd

df = pd.read_csv("locations.csv")

# outdoor/indoor
io_map = {"outdoor": 1.0, "both": 0.5, "indoor": 0.0}
df["feat_outdoor"] = df["indoor_outdoor"].str.lower().map(io_map).fillna(0.5)

# duration (normalized)
max_dur = df["max_useful_days"].max()
df["feat_duration"] = df["max_useful_days"] / max_dur

# time slot flags
df["feat_morning"]   = df["time_slot"].str.contains("Morning",   na=False).astype(float)
df["feat_afternoon"] = df["time_slot"].str.contains("Afternoon", na=False).astype(float)
df["feat_evening"]   = df["time_slot"].str.contains("Evening",   na=False).astype(float)

# category one-hot
cats = {
    "Adventure & Sports":        "feat_adventure",
    "Culture & History":         "feat_culture",
    "Food & Drink":              "feat_food",
    "Outdoor & Nature":          "feat_nature",
    "Nightlife & Entertainment": "feat_nightlife",
    "Relaxation & Wellness":     "feat_wellness",
}
for cat, col in cats.items():
    df[col] = (df["category"] == cat).astype(float)

df.to_csv("locations.csv", index=False)
print("Done — features added to locations.csv")