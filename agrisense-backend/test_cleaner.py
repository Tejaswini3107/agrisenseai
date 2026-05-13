import pandas as pd
import numpy as np
from data_pipeline.processing.cleaner import clean_weather_dataframe

print("=" * 60)
print("Testing Weather Data Cleaner Function")
print("=" * 60)

# Create a test DataFrame with missing values and out-of-range values
print("\n1. Creating test DataFrame with various data issues")
print("-" * 60)

test_data = {
    "temperature": [25.5, np.nan, 32.0, -10, 60.0, 20.0, np.nan, 28.5],
    "humidity": [65.0, 75.0, np.nan, 120.0, -5.0, 80.0, 90.0, np.nan],
    "rainfall_mm": [5.0, np.nan, 0.0, 15.0, 600.0, np.nan, 25.0, 10.0],
    "wind_speed": [10.5, 15.0, np.nan, 200.0, 5.0, np.nan, 30.0, 25.5],
}

df_original = pd.DataFrame(test_data)

print("\nOriginal DataFrame:")
print(df_original)
print("\nOriginal DataFrame Info:")
print(df_original.info())
print("\nMissing values count:")
print(df_original.isnull().sum())

# Apply cleaning
print("\n2. Applying clean_weather_dataframe()")
print("-" * 60)
df_cleaned = clean_weather_dataframe(df_original)

print("\nCleaned DataFrame:")
print(df_cleaned)
print("\nCleaned DataFrame Info:")
print(df_cleaned.info())

# Verify the results
print("\n3. Verification of Cleaning Results")
print("-" * 60)

print("\nTemperature column:")
print(f"  Range: {df_cleaned['temperature'].min():.2f} to {df_cleaned['temperature'].max():.2f}°C")
print(
    f"  All values in [-5, 55]? {(df_cleaned['temperature'] >= -5).all() and (df_cleaned['temperature'] <= 55).all()}"
)
print(f"  No NaN values? {not df_cleaned['temperature'].isnull().any()}")

print("\nHumidity column:")
print(f"  Range: {df_cleaned['humidity'].min():.2f} to {df_cleaned['humidity'].max():.2f}%")
print(f"  All values in [0, 100]? {(df_cleaned['humidity'] >= 0).all() and (df_cleaned['humidity'] <= 100).all()}")
print(f"  No NaN values? {not df_cleaned['humidity'].isnull().any()}")

print("\nRainfall column:")
print(f"  Range: {df_cleaned['rainfall_mm'].min():.2f} to {df_cleaned['rainfall_mm'].max():.2f}mm")
print(
    f"  All values in [0, 500]? {(df_cleaned['rainfall_mm'] >= 0).all() and (df_cleaned['rainfall_mm'] <= 500).all()}"
)
print(f"  No NaN values? {not df_cleaned['rainfall_mm'].isnull().any()}")

print("\nWind Speed column:")
print(f"  Range: {df_cleaned['wind_speed'].min():.2f} to {df_cleaned['wind_speed'].max():.2f} km/h")
print(f"  All values in [0, 150]? {(df_cleaned['wind_speed'] >= 0).all() and (df_cleaned['wind_speed'] <= 150).all()}")
print(f"  No NaN values? {not df_cleaned['wind_speed'].isnull().any()}")

print("\nDataFrame Statistics:")
print(df_cleaned.describe())

print("\n4. Test Summary")
print("-" * 60)
print(f"✓ Original rows: {len(df_original)}")
print(f"✓ Cleaned rows: {len(df_cleaned)}")
print(f"✓ Rows removed: {len(df_original) - len(df_cleaned)}")
print(
    f"✓ All required columns present? {all(col in df_cleaned.columns for col in ['temperature', 'humidity', 'rainfall_mm', 'wind_speed'])}"
)
print(f"✓ No NaN values in critical columns? {not df_cleaned[['temperature', 'humidity']].isnull().any().any()}")
print(f"✓ Index properly reset? {list(df_cleaned.index) == list(range(len(df_cleaned)))}")

print("\n" + "=" * 60)
print("Tests completed successfully!")
print("=" * 60)
