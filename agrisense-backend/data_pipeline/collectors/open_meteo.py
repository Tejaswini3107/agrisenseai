import requests


def get_current_weather_meteo(lat, lon):
    """
    Fetch current weather data from Open-Meteo API.
    
    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
    
    Returns:
        dict: Weather data with keys: temperature, humidity, rainfall_mm, wind_speed
              Returns None on error
    """
    try:
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m',
            'wind_speed_unit': 'kmh'
        }
        
        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'temperature': float(data['current']['temperature_2m']),
            'humidity': float(data['current']['relative_humidity_2m']),
            'rainfall_mm': float(data['current']['precipitation']),
            'wind_speed': float(data['current']['wind_speed_10m'])
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
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean',
            'wind_speed_unit': 'kmh',
            'forecast_days': 7,
            'timezone': 'auto'
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
            
            day_dict = {
                'date': dates[i],
                'temperature': temperature,
                'humidity': humidity,
                'rainfall_mm': rainfall_mm,
                'wind_speed': wind_speed
            }
            
            forecast_list.append(day_dict)
        
        return forecast_list
    
    except Exception as e:
        print(f'Open-Meteo forecast failed: {str(e)}')
        return None
