from data_pipeline.collectors.openweather import get_current_weather, get_forecast

print("=" * 60)
print("Testing OpenWeather API Functions")
print("=" * 60)

# Test coordinates (Lahore, Pakistan)
latitude = 31.5497
longitude = 74.3436

print("\n1. Testing get_current_weather(lat, lon)")
print("-" * 60)
try:
    current_weather = get_current_weather(latitude, longitude)
    if current_weather:
        print(f"✓ Current weather for ({latitude}, {longitude}):")
        for key, value in current_weather.items():
            print(f"  {key}: {value}")
    else:
        print("✗ Failed to fetch current weather")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

print("\n2. Testing get_forecast(lat, lon)")
print("-" * 60)
try:
    forecast = get_forecast(latitude, longitude)
    if forecast and len(forecast) > 0:
        print(f"✓ Retrieved {len(forecast)} days of forecast data")
        print("\nForecast details:")
        for day in forecast:
            print(f"  Date: {day['date']}")
            print(f"    Temperature: {day['temperature']}°C")
            print(f"    Humidity: {day['humidity']}%")
            print(f"    Rainfall: {day['rainfall_mm']}mm")
            print(f"    Wind Speed: {day['wind_speed']} km/h")
            print()
    else:
        print("✗ Failed to fetch forecast data")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

print("=" * 60)
print("Tests completed!")
print("=" * 60)