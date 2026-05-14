**Welcome to TravelWithMe — Your Swiss Vacation Planner**

TravelWithMe is a personalised Swiss vacation planning app that builds a custom day-by-day itinerary based on your destination, travel season, and activity preferences — all in three simple steps.


**Step 1 — Plan your trip**

Begin by setting the basic parameters of your trip:
- **Your Swiss destination** — choose the city or region you want to explore
- **Trip duration** — how many days you plan to stay
- **Travel season** — the time of year you'll be visiting (spring, summer, autumn, or winter)

These three inputs form the foundation of your itinerary.


**Step 2 — Tell us what you love**

Next, rate your interest in each of the six activity categories on a scale from 1 to 5:
- **1** — I am not interested in this type of activity
- **5** — I love this type of activity

This preference profile is fed into a **KNN-based recommendation model**, which queries a database of Swiss activities and selects the ones that best match what you're looking for. The more honest your ratings, the more tailored your itinerary will be.


**Step 3 — Explore your personalised itinerary**

Once your preferences are submitted, the app generates a complete day-by-day plan. For each day, you'll see a structured schedule broken down into **morning, afternoon, and evening**, with a recommended activity for each slot.

The itinerary also includes:
- **Weather information** — if you're travelling in spring, the app pulls real-time forecasts via a live weather API. For other seasons, it displays average weather conditions typical of that time of year, so you always know what to expect
- **An activity breakdown** — a visual overview of the types of activities included in your plan
- **A downloadable PDF** — export your full itinerary to take it with you, with all recommended activities laid out clearly


**You're ready — enjoy your Swiss adventure!**


ml_rating.py  


Machine-learning module using a real scikit-learn K-Nearest Neighbors model.

How it works
1.  The user rates 6 activity categories on a 1-5 slider (Step 2) and
    picks a travel season in Step 1 (spring / summer / fall / winter).

2.  Each activity in the database is encoded as a 15-dimensional
    numeric feature vector:
        - 6 one-hot values  — which category the activity belongs to
        - 1 value           — indoor (0) / both (0.5) / outdoor (1)
        - 3 binary values   — whether Morning / Afternoon / Evening is allowed
        - 1 value           — duration normalised to [0, 1]
        - 4 binary values   — whether the activity runs in each season

3.  A single USER PROFILE VECTOR is built from the slider ratings and
    the chosen season. Categories are weighted by (rating/3)^2.
    The four season flags in the profile are set to 1 for the chosen
    season and 0 for all others.

4.  We fit a scikit-learn NearestNeighbors model on all activities already filtered to the chosen season.

5.  We call .kneighbors() with the user profile as the query.
    Closer cosine distance = more similar to the user's taste.

6.  Activities are ranked best to worst; the itinerary builder picks top-down.