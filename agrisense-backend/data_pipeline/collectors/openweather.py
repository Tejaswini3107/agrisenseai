import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from fastapi import HTTPException
import redis
import json

# Load environment variables
load_dotenv()

# Get API key
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Initialize Redis connection for caching
try:
    _redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True, socket_connect_timeout=2)
    _redis.ping()
except Exception:
    _redis = None


def get_current_weather(lat, lon):
    """
    Fetch current weather data for given latitude and longitude.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        dict: Dictionary containing temperature, humidity, wind_speed, rainfall_mm, condition, location_name

    Raises:
        HTTPException: If API call fails with status 503
    """
    try:
        # Check cache first
        cache_key = f"weather:current:{lat}:{lon}"
        if _redis is not None:
            try:
                cached = _redis.get(cache_key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass
        
        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract weather data
        weather_data = {
            "temperature": float(data["main"]["temp"]),
            "humidity": float(data["main"]["humidity"]),
            "rainfall_mm": float(data.get("rain", {}).get("1h", 0.0)),
            "wind_speed": float(data["wind"]["speed"] * 3.6),  # Convert m/s to km/h
            "condition": data["weather"][0]["description"],
            "location_name": data["name"],
        }

        # Cache the result (30 minute expiry)
        if _redis is not None:
            try:
                _redis.setex(cache_key, 1800, json.dumps(weather_data))
            except Exception:
                pass

        return weather_data

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=503, detail="Weather service unavailable")


def get_forecast(lat, lon):
    """
    Fetch 5-day weather forecast with 3-hour intervals and aggregate to daily summaries.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        list: List of dicts, one per day, with keys: date, temperature, humidity, rainfall_mm, wind_speed

    Raises:
        HTTPException: If API call fails with status 503
    """
    try:
        # Check cache first
        cache_key = f"weather:forecast:{lat}:{lon}"
        if _redis is not None:
            try:
                cached = _redis.get(cache_key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass
        
        url = "https://api.openweathermap.org/data/2.5/forecast"

        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric", "cnt": 40}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Group forecast data by date
        daily_data = {}

        for entry in data["list"]:
            # Extract date from datetime string (format: "2023-01-01 12:00:00")
            dt_str = entry["dt_txt"]
            date = dt_str.split(" ")[0]  # Get YYYY-MM-DD

            if date not in daily_data:
                daily_data[date] = {"temperatures": [], "humidities": [], "rainfalls": [], "wind_speeds": []}

            # Accumulate values for each day
            daily_data[date]["temperatures"].append(entry["main"]["temp"])
            daily_data[date]["humidities"].append(entry["main"]["humidity"])
            daily_data[date]["rainfalls"].append(entry.get("rain", {}).get("3h", 0.0))
            daily_data[date]["wind_speeds"].append(entry["wind"]["speed"] * 3.6)  # Convert m/s to km/h

        # Compute daily averages and totals
        forecast_list = []
        for date in sorted(daily_data.keys()):
            day_data = daily_data[date]

            forecast_dict = {
                "date": date,
                "temperature": round(sum(day_data["temperatures"]) / len(day_data["temperatures"]), 2),
                "humidity": round(sum(day_data["humidities"]) / len(day_data["humidities"]), 2),
                "rainfall_mm": round(sum(day_data["rainfalls"]), 2),
                "wind_speed": round(sum(day_data["wind_speeds"]) / len(day_data["wind_speeds"]), 2),
            }

            forecast_list.append(forecast_dict)

        # Cache the result (6 hour expiry)
        if _redis is not None:
            try:
                _redis.setex(cache_key, 21600, json.dumps(forecast_list))
            except Exception:
                pass

        return forecast_list

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
