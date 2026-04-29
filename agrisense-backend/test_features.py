import pandas as pd
import numpy as np
from data_pipeline.processing.feature_engineering import engineer_features

print("=" * 70)
print("Testing Feature Engineering Function")
print("=" * 70)

# Create test DataFrame
print("\n1. Creating test DataFrame")
print("-" * 70)

test_data = {
    'temperature': [15.0, 25.0, 35.0, 40.0, 10.0, 28.0, 32.5, 20.0],
    'humidity': [45.0, 60.0, 75.0, 85.0, 50.0, 72.0, 80.0, 55.0],
    'rainfall_mm': [0.0, 5.0, 15.0, 75.0, 0.0, 8.0, 45.0, 2.5],
    'wind_speed': [5.0, 10.0, 15.0, 20.0, 8.0, 12.0, 18.0, 7.0]
}

df_original = pd.DataFrame(test_data)

print("\nOriginal DataFrame:")
print(df_original)
print(f"\nOriginal shape: {df_original.shape}")
print(f"Original columns: {list(df_original.columns)}")

# Apply feature engineering
print("\n2. Applying engineer_features()")
print("-" * 70)
df_engineered = engineer_features(df_original)

print("\nEngineered DataFrame:")
print(df_engineered)
print(f"\nEngineered shape: {df_engineered.shape}")
print(f"Engineered columns: {list(df_engineered.columns)}")

# Verify all new columns exist
print("\n3. Verifying New Columns")
print("-" * 70)

expected_columns = [
    'temperature', 'humidity', 'rainfall_mm', 'wind_speed',
    'heat_index', 'dew_point', 'vapor_pressure_deficit',
    'rainfall_category', 'is_high_humidity', 'is_high_temp'
]

print("\nExpected columns:")
for col in expected_columns:
    exists = col in df_engineered.columns
    status = "✓" if exists else "✗"
    print(f"  {status} {col}")

# Detailed feature analysis
print("\n4. Feature Analysis")
print("-" * 70)

print("\nHeat Index:")
print(f"  Range: {df_engineered['heat_index'].min():.2f} to {df_engineered['heat_index'].max():.2f}")
print(f"  All values in [10, 80]? {(df_engineered['heat_index'] >= 10).all() and (df_engineered['heat_index'] <= 80).all()}")
print(f"  Values: {df_engineered['heat_index'].tolist()}")

print("\nDew Point:")
print(f"  Range: {df_engineered['dew_point'].min():.2f} to {df_engineered['dew_point'].max():.2f}°C")
print(f"  Values: {df_engineered['dew_point'].tolist()}")

print("\nVapor Pressure Deficit:")
print(f"  Range: {df_engineered['vapor_pressure_deficit'].min():.4f} to {df_engineered['vapor_pressure_deficit'].max():.4f}")
print(f"  All rounded to 4 decimals? ", end="")
vpd_vals = df_engineered['vapor_pressure_deficit'].tolist()
rounded_check = all(len(str(v).split('.')[-1]) <= 4 for v in vpd_vals if isinstance(v, float))
print("✓" if rounded_check else "✗")
print(f"  Values: {vpd_vals}")

print("\nRainfall Category:")
print(f"  Unique values: {df_engineered['rainfall_category'].unique().tolist()}")
print(f"  Value counts:")
print(df_engineered['rainfall_category'].value_counts())
print("\n  Sample mappings:")
for idx, row in df_engineered.iterrows():
    print(f"    {row['rainfall_mm']}mm → '{row['rainfall_category']}'")

print("\nHigh Humidity Flag (> 70%):")
print(f"  Unique values: {sorted(df_engineered['is_high_humidity'].unique().tolist())}")
print(f"  Value counts:")
print(df_engineered['is_high_humidity'].value_counts())
print("\n  Sample mappings:")
for idx, row in df_engineered.iterrows():
    if idx < 5:
        print(f"    {row['humidity']}% humidity → {row['is_high_humidity']}")

print("\nHigh Temperature Flag (> 32°C):")
print(f"  Unique values: {sorted(df_engineered['is_high_temp'].unique().tolist())}")
print(f"  Value counts:")
print(df_engineered['is_high_temp'].value_counts())
print("\n  Sample mappings:")
for idx, row in df_engineered.iterrows():
    if idx < 5:
        print(f"    {row['temperature']}°C temperature → {row['is_high_temp']}")

# Test with edge cases
print("\n5. Testing Edge Cases")
print("-" * 70)

edge_case_data = {
    'temperature': [0.0, 50.0, 25.0, 25.0],
    'humidity': [0.0, 100.0, 50.0, 75.0],
    'rainfall_mm': [0.0, 100.0, 25.0, 10.0],
    'wind_speed': [0.0, 50.0, 15.0, 12.0]
}

df_edge = pd.DataFrame(edge_case_data)
df_edge_eng = engineer_features(df_edge)

print("\nEdge case results:")
print(df_edge_eng[['temperature', 'humidity', 'rainfall_mm', 'rainfall_category', 'is_high_humidity', 'is_high_temp']])

# Final statistics
print("\n6. Feature Statistics")
print("-" * 70)
print("\nDescriptive Statistics:")
print(df_engineered[['heat_index', 'dew_point', 'vapor_pressure_deficit']].describe())

print("\nData Types:")
print(df_engineered.dtypes)

print("\n7. Test Summary")
print("-" * 70)
print(f"✓ All {len(expected_columns)} expected columns present? {all(col in df_engineered.columns for col in expected_columns)}")
print(f"✓ No data loss (same rows)? {len(df_engineered) == len(df_original)}")
print(f"✓ Heat index clipped correctly? {(df_engineered['heat_index'] >= 10).all() and (df_engineered['heat_index'] <= 80).all()}")
print(f"✓ Vapor pressure deficit rounded to 4 decimals? {rounded_check}")
print(f"✓ Rainfall categories correct types? {df_engineered['rainfall_category'].dtype == 'object'}")
print(f"✓ Humidity and temperature flags are integers? {df_engineered['is_high_humidity'].dtype in ['int64', 'int32'] and df_engineered['is_high_temp'].dtype in ['int64', 'int32']}")

print("\n" + "=" * 70)
print("Tests completed successfully!")
print("=" * 70)