
import requests


# Open-Meteo describes the weather using a code known as a ‘weather code’.
# We translate the most common codes into user-friendly emojis accompanied by a caption.
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
   

    # Build the request.
    # We tell Open-Meteo the location, the desired values and
    # that we want the times shown in Swiss local time.
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "forecast_days": num_days,
        "timezone": "Europe/Zurich",
    }

    # Call the API. If anything goes wrong,
    # we return an empty list so the rest of the app keeps working.
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()       # raise an error if status is 4xx / 5xx
        data = response.json()
    except Exception:
        return []

    # The reply has parallel lists, one value per day.
   
    daily = data.get("daily", {})
    days = daily.get("time", [])

    # Think of it as a daily dictionary that will make it easier to use later on.
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


# A quick test: run `python weather.py` to see the weather forecast for Zurich.
if __name__ == "__main__":
    for day in get_weather(lat=47.37, lon=8.54, num_days=3):
        print(day)
