"""
weather.py
==========
Fetches a daily weather forecast from the Open-Meteo API (free, no key needed).
Returns real-time data for today and the coming days.
"""

import requests

# ── Weather Code Mapping ──────────────────────────────────────────────────────

# Maps WMO weather interpretation codes (returned by Open-Meteo) to a short
# icon label and a human-readable description.
# Format: code: (icon, description)
WEATHER_CODES = {
    0:  ("Sun",   "Clear sky"),
    1:  ("Sun",   "Mostly clear"),
    2:  ("Cloud", "Partly cloudy"),
    3:  ("Cloud", "Overcast"),
    45: ("Fog",   "Fog"),
    48: ("Fog",   "Fog"),
    51: ("Rain",  "Light drizzle"),
    53: ("Rain",  "Drizzle"),
    55: ("Rain",  "Heavy drizzle"),
    61: ("Rain",  "Light rain"),
    63: ("Rain",  "Rain"),
    65: ("Rain",  "Heavy rain"),
    71: ("Snow",  "Light snow"),
    73: ("Snow",  "Snow"),
    75: ("Snow",  "Heavy snow"),
    80: ("Rain",  "Rain showers"),
    81: ("Rain",  "Rain showers"),
    82: ("Storm", "Heavy showers"),
    95: ("Storm", "Thunderstorm"),
    96: ("Storm", "Thunderstorm + hail"),
    99: ("Storm", "Thunderstorm + hail"),
}

# ── Forecast Fetcher ──────────────────────────────────────────────────────────

def get_weather(lat: float, lon: float, num_days: int) -> list[dict]:
    """
    Fetch a daily weather forecast for one location.
    Returns a list with one dict per day:
        [
            {
                "date":  "2026-05-09",
                "min":   8,
                "max":   18,
                "rain":  0.2,
                "icon":  "Sun",
                "label": "Clear sky"
            },
            ...
        ]
    Returns an empty list if the API call fails.
    Note: this is REAL-TIME data from Open-Meteo — it reflects the
    actual forecast for today and the coming days at the chosen city.
    """

    # Open-Meteo forecast endpoint (no API key required)
    url = "https://api.open-meteo.com/v1/forecast"

    # Request daily max/min temperature, total precipitation, and weather code
    # Timezone is fixed to Switzerland so dates align with local Swiss time
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         "temperature_2m_max,temperature_2m_min,"
                         "precipitation_sum,weather_code",
        "forecast_days": num_days,
        "timezone":      "Europe/Zurich",
    }

    # Attempt the API call; return an empty list on any network or HTTP error
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        data = response.json()
    except Exception:
        return []

    # ── Parse Response ────────────────────────────────────────────────────────

    daily = data.get("daily", {})
    days  = daily.get("time", [])  # List of date strings, one per forecast day

    forecast = []

    for i in range(len(days)):
        # Look up the WMO weather code for this day; fall back to "Unknown" if unrecognised
        code        = daily["weather_code"][i]
        icon, label = WEATHER_CODES.get(code, ("?", "Unknown"))

        # Build a clean dict for each day and append it to the results list
        forecast.append({
            "date":  daily["time"][i],
            "min":   round(daily["temperature_2m_min"][i]),   # °C, rounded to nearest integer
            "max":   round(daily["temperature_2m_max"][i]),   # °C, rounded to nearest integer
            "rain":  daily["precipitation_sum"][i],           # mm of precipitation
            "icon":  icon,
            "label": label,
        })

    return forecast

# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run a quick sanity check using Zurich's coordinates
    for day in get_weather(lat=47.37, lon=8.54, num_days=3):
        print(day)