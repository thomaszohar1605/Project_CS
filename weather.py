"""
weather.py
==========
Fetches a daily weather forecast from the Open-Meteo API (free, no key needed).
Returns real-time data for today and the coming days.
"""

import requests

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
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         "temperature_2m_max,temperature_2m_min,"
                         "precipitation_sum,weather_code",
        "forecast_days": num_days,
        "timezone":      "Europe/Zurich",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    daily    = data.get("daily", {})
    days     = daily.get("time", [])
    forecast = []

    for i in range(len(days)):
        code       = daily["weather_code"][i]
        icon, label = WEATHER_CODES.get(code, ("?", "Unknown"))
        forecast.append({
            "date":  daily["time"][i],
            "min":   round(daily["temperature_2m_min"][i]),
            "max":   round(daily["temperature_2m_max"][i]),
            "rain":  daily["precipitation_sum"][i],
            "icon":  icon,
            "label": label,
        })

    return forecast


if __name__ == "__main__":
    # Quick test: Zurich
    for day in get_weather(lat=47.37, lon=8.54, num_days=3):
        print(day)
