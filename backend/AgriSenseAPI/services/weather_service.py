import os
import requests

def fetch_external_weather(city: str):
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return {'error': 'OPENWEATHER_API_KEY not set'}
    url = f'https://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        return {'error': 'request_failed', 'detail': str(e)}
    js = r.json()
    return {
        'city': js.get('name'),
        'temperature': js.get('main', {}).get('temp'),
        'humidity': js.get('main', {}).get('humidity'),
        'description': js.get('weather', [{}])[0].get('description')
    }
