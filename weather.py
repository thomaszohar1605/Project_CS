"""
weather.py
==========

A tiny helper that asks the Open-Meteo API for the weather forecast
of a place in Switzerland.

Open-Meteo is free, needs no API key, and answers with simple JSON.
Docs: https://open-meteo.com/en/docs
"""

import requests


# Open-Meteo describes the weather with a number called a "weather code".
# We translate the most common codes into a friendly emoji + label.
# Full list of codes: https://open-meteo.com/en/docs (search for "WMO Weather codes")
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
    Get a daily weather forecast for one place.

    Parameters
    ----------
    lat : latitude  (e.g. 47.37 for Zurich)
    lon : longitude (e.g.  8.54 for Zurich)
    num_days : how many days ahead to forecast (1 to 16)

    Returns
    -------
    A list with one dict per day, like:
        [
            {"date": "2026-05-04", "min": 8, "max": 18,
             "rain": 0.2, "icon": "Sun", "label": "Clear sky"},
            ...
        ]
    Returns an empty list if the API call fails.
    """

    # 1. Build the request.
    #    We tell Open-Meteo which place, which values we want, and
    #    that we want the times in Swiss local time.
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "forecast_days": num_days,
        "timezone": "Europe/Zurich",
    }

    # 2. Call the API. If anything goes wrong (no internet, bad reply...),
    #    we return an empty list so the rest of the app keeps working.
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()       # raise an error if status is 4xx / 5xx
        data = response.json()
    except Exception:
        return []

    # 3. The reply has parallel lists, one value per day.
    #    Example shape:
    #    data["daily"] = {
    #        "time":                ["2026-05-04", "2026-05-05", ...],
    #        "temperature_2m_max":  [18.3, 17.1, ...],
    #        "temperature_2m_min":  [ 8.5,  9.2, ...],
    #        "precipitation_sum":   [ 0.0,  3.4, ...],
    #        "weather_code":        [   0,    61, ...],
    #    }
    daily = data.get("daily", {})
    days = daily.get("time", [])

    # 4. Re-shape it into one dict per day, easier to use later.
    forecast = []
    for i in range(len(days)):
        code = daily["weather_code"][i]
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


# Quick self-test: run `python weather.py` to see Zurich's forecast.
if __name__ == "__main__":
    for day in get_weather(lat=47.37, lon=8.54, num_days=3):
        print(day)
