import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def get_soil_moisture(lat, lon):
    """
    Fetch soil moisture (GWETROOT) data from NASA POWER API.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        float: Average soil moisture (root zone) rounded to 4 decimal places,
               or None if request fails or no valid data.
    """
    # Calculate date range: today - 7 days to today
    today = datetime.utcnow()
    start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    # NASA POWER API endpoint
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "GWETROOT",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract GWETROOT values
        gwetroot_data = data.get("properties", {}).get("parameter", {}).get("GWETROOT", {})

        if not gwetroot_data:
            return None

        # Filter out -999 (missing/invalid values)
        valid_values = [v for v in gwetroot_data.values() if v != -999]

        if not valid_values:
            return None

        # Calculate and return average
        avg_moisture = np.mean(valid_values)
        return round(avg_moisture, 4)

    except requests.RequestException as e:
        print(f"Error fetching soil moisture data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error parsing soil moisture response: {e}")
        return None


def get_historical_weather(lat, lon, start_date, end_date):
    """
    Fetch historical weather data from NASA POWER API.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format

    Returns:
        pd.DataFrame: DataFrame with columns:
            - date
            - temperature (°C)
            - humidity (%)
            - rainfall_mm
            - wind_speed (km/h)
        Returns None if request fails or no data available.
    """
    # NASA POWER API endpoint
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,WS2M,GWETROOT",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract weather parameters
        properties = data.get("properties", {})
        parameters = properties.get("parameter", {})

        t2m = parameters.get("T2M", {})  # Temperature (°C)
        rh2m = parameters.get("RH2M", {})  # Humidity (%)
        prectotcorr = parameters.get("PRECTOTCORR", {})  # Precipitation (mm)
        ws2m = parameters.get("WS2M", {})  # Wind speed (m/s)
        gwetroot = parameters.get("GWETROOT", {})  # Soil moisture (fraction)

        if not all([t2m, rh2m, prectotcorr, ws2m]):
            return None

        # Build lists for DataFrame
        dates = []
        temperatures = []
        humidities = []
        rainfalls = []
        wind_speeds = []
        soil_moistures = []

        # Iterate through dates in the data
        for date_str in sorted(t2m.keys()):
            # Parse date (format: YYYYMMDD)
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue

            # Extract values and replace -999 with NaN
            temp = t2m.get(date_str, -999)
            humid = rh2m.get(date_str, -999)
            precip = prectotcorr.get(date_str, -999)
            wind = ws2m.get(date_str, -999)
            soil_moisture_raw = gwetroot.get(date_str, -999)

            # Replace -999 with NaN
            temp = np.nan if temp == -999 else temp
            humid = np.nan if humid == -999 else humid
            precip = np.nan if precip == -999 else precip
            wind = np.nan if wind == -999 else wind
            soil_moisture_raw = np.nan if soil_moisture_raw == -999 else soil_moisture_raw

            # Convert wind speed from m/s to km/h (multiply by 3.6)
            wind = wind * 3.6 if not np.isnan(wind) else np.nan

            # Convert soil moisture from fraction (0-1) to percentage (0-100)
            soil_moisture = (soil_moisture_raw * 100) if not np.isnan(soil_moisture_raw) else np.nan

            dates.append(date_obj)
            temperatures.append(temp)
            humidities.append(humid)
            rainfalls.append(precip)
            wind_speeds.append(wind)
            soil_moistures.append(soil_moisture)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "date": dates,
                "temperature": temperatures,
                "humidity": humidities,
                "rainfall_mm": rainfalls,
                "wind_speed": wind_speeds,
                "soil_moisture": soil_moistures,
            }
        )

        # Handle missing soil moisture values (>10% threshold)
        missing_pct = (df["soil_moisture"].isna().sum() / len(df)) * 100
        if missing_pct > 10:
            df["soil_moisture"].fillna(df["soil_moisture"].median(), inplace=True)
            print(f"  Warning: {missing_pct:.1f}% missing soil_moisture values. Filled with median.")

        return df if len(df) > 0 else None

    except requests.RequestException as e:
        print(f"Error fetching historical weather data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error parsing weather response: {e}")
        return None


def get_soil_moisture(lat, lon):
    """
    Fetch soil moisture (GWETROOT) data from NASA POWER API for the past 7 days.
    Returns average soil moisture percentage (0-100), or 35.0 as safe default on error.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        float: Average soil moisture as percentage (0-100), rounded to 2 decimal places.
               Returns 35.0 (moderate soil moisture) on error.
    """
    try:
        # Compute date range
        today = datetime.now().strftime("%Y%m%d")
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

        # NASA POWER API endpoint
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        params = {
            "parameters": "GWETROOT",
            "community": "AG",
            "format": "JSON",
            "latitude": lat,
            "longitude": lon,
            "start": seven_days_ago,
            "end": today,
        }

        # Fetch data with 30 second timeout
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract GWETROOT values
        gwetroot_dict = data.get("properties", {}).get("parameter", {}).get("GWETROOT", {})

        if not gwetroot_dict:
            return 35.0

        # Filter out -999.0 values (missing/invalid)
        valid_values = [v for v in gwetroot_dict.values() if v != -999.0]

        if not valid_values:
            return 35.0

        # Compute mean, convert to percentage, and round to 2 decimal places
        avg_soil_moisture = np.mean(valid_values)
        soil_moisture_percent = avg_soil_moisture * 100

        return round(soil_moisture_percent, 2)

    except Exception as e:
        print(f"Error fetching soil moisture data: {e}")
        return 35.0
