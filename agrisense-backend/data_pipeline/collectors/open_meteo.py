import requests

# WMO Weather Interpretation Codes → human-readable condition string
_WMO_CONDITIONS = {
    0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
    45: 'fog', 48: 'icy fog',
    51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle',
    61: 'light rain', 63: 'rain', 65: 'heavy rain',
    71: 'light snow', 73: 'snow', 75: 'heavy snow', 77: 'snow grains',
    80: 'rain showers', 81: 'heavy rain showers', 82: 'violent rain showers',
    85: 'snow showers', 86: 'heavy snow showers',
    95: 'thunderstorm', 96: 'thunderstorm with hail', 99: 'thunderstorm with heavy hail',
}

def _wmo_to_condition(code):
    return _WMO_CONDITIONS.get(int(code), 'partly cloudy')

def _reverse_geocode(lat, lon):
    """Resolve lat/lon to a city name using Open-Meteo geocoding."""
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'lat': lat, 'lon': lon, 'format': 'json'},
            headers={'User-Agent': 'agrisense-backend/1.0'},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        addr = d.get('address', {})
        return (addr.get('village') or addr.get('town') or addr.get('city')
                or addr.get('county') or addr.get('state') or 'Farm location')
    except Exception:
        return 'Farm location'


def get_current_weather_meteo(lat, lon):
    """
    Fetch current weather data from Open-Meteo API (real-time, no API key needed).

    Returns:
        dict with temperature, humidity, rainfall_mm, wind_speed, condition, location_name
        Returns None on error.
    """
    try:
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code',
            'wind_speed_unit': 'kmh',
            'timezone': 'auto',
        }

        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        cur = data['current']

        location_name = _reverse_geocode(lat, lon)

        return {
            'temperature': float(cur['temperature_2m']),
            'humidity': float(cur['relative_humidity_2m']),
            'rainfall_mm': float(cur['precipitation']),
            'wind_speed': float(cur['wind_speed_10m']),
            'condition': _wmo_to_condition(cur.get('weather_code', 0)),
            'location_name': location_name,
        }

    except Exception as e:
        print(f'Open-Meteo current weather failed: {str(e)}')
        return None


def get_forecast_meteo(lat, lon):
    """
    Fetch 7-day weather forecast from Open-Meteo API.
    
    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
    
    Returns:
        list: List of daily forecast dicts with keys: date, temperature, humidity, rainfall_mm, wind_speed
              Returns None on error
    """
    try:
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean,weather_code',
            'wind_speed_unit': 'kmh',
            'forecast_days': 7,
            'timezone': 'auto',
        }
        
        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        forecast_list = []
        dates = data['daily']['time']
        
        for i in range(len(dates)):
            temperature = round((data['daily']['temperature_2m_max'][i] + data['daily']['temperature_2m_min'][i]) / 2, 2)
            humidity = round(float(data['daily']['relative_humidity_2m_mean'][i]), 2)
            rainfall_mm = round(float(data['daily']['precipitation_sum'][i]), 2)
            wind_speed = round(float(data['daily']['wind_speed_10m_max'][i]), 2)
            wmo_code = int(data['daily'].get('weather_code', [0]*len(dates))[i] or 0)
            condition = _wmo_to_condition(wmo_code)

            day_dict = {
                'date': dates[i],
                'temperature': temperature,
                'humidity': humidity,
                'rainfall_mm': rainfall_mm,
                'wind_speed': wind_speed,
                'condition': condition,
            }
            
            forecast_list.append(day_dict)
        
        return forecast_list
    
    except Exception as e:
        print(f'Open-Meteo forecast failed: {str(e)}')
        return None
