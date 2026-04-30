from data_pipeline.collectors.nasa_power import get_soil_moisture, get_historical_weather
from datetime import datetime, timedelta

print("=" * 60)
print("Testing NASA POWER API Functions")
print("=" * 60)

# Test coordinates (Lahore, Pakistan)
latitude = 31.5497
longitude = 74.3436

print("\n1. Testing get_soil_moisture(lat, lon)")
print("-" * 60)
try:
    soil_moisture = get_soil_moisture(latitude, longitude)
    if soil_moisture is not None:
        print(f"✓ Soil Moisture (GWETROOT) for ({latitude}, {longitude}): {soil_moisture}")
        print(f"  (Value between 0=dry and 1=saturated)")
    else:
        print("✗ Failed to fetch soil moisture data")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n2. Testing get_historical_weather(lat, lon, start_date, end_date)")
print("-" * 60)
try:
    # Get data for last 7 days
    end_date = datetime.utcnow().strftime("%Y%m%d")
    start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")

    print(f"Fetching weather data from {start_date} to {end_date}")
    print(f"Location: ({latitude}, {longitude})")

    df = get_historical_weather(latitude, longitude, start_date, end_date)

    if df is not None and len(df) > 0:
        print(f"✓ Retrieved {len(df)} days of historical weather data")
        print("\nDataFrame head:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
        print("\nDataFrame statistics:")
        print(df.describe())
    else:
        print("✗ Failed to fetch historical weather data")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Tests completed!")
print("=" * 60)

