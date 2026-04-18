import sys
import os

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')

import pandas as pd
import numpy as np
from data_pipeline.collectors.nasa_power import get_historical_weather
from data_pipeline.processing.cleaner import clean_weather_dataframe
from data_pipeline.processing.feature_engineering import engineer_features

# Define locations with coordinates
LOCATIONS = [
    {
        'name': 'Punjab',
        'lat': 30.9,
        'lon': 75.8
    },
    {
        'name': 'Maharashtra',
        'lat': 19.0,
        'lon': 76.1
    },
    {
        'name': 'Andhra Pradesh',
        'lat': 15.9,
        'lon': 79.7
    },
    {
        'name': 'Tamil Nadu',
        'lat': 11.1,
        'lon': 77.3
    },
    {
        'name': 'Uttar Pradesh',
        'lat': 27.0,
        'lon': 80.9
    },
    {
        'name': 'Faisalabad',
        'lat': 31.4,
        'lon': 73.1
    },
    {
        'name': 'Rahim Yar Khan',
        'lat': 28.4,
        'lon': 70.3
    }
]

def generate_climate_dataset():
    """
    Generate a comprehensive climate dataset from historical weather data
    across multiple Indian locations.
    """
    print("=" * 70)
    print("Generating Climate Dataset")
    print("=" * 70)
    
    all_dataframes = []
    
    # Fetch data for each location
    for location in LOCATIONS:
        print(f"\nFetching data for {location['name']}...", end=" ")
        try:
            df = get_historical_weather(
                location['lat'],
                location['lon'],
                '20190101',
                '20231231'
            )
            
            if df is not None and len(df) > 0:
                # Add location column
                df['location'] = location['name']
                all_dataframes.append(df)
                print(f"✓ ({len(df)} rows)")
            else:
                print("✗ No data retrieved")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not all_dataframes:
        print("\n✗ No data was retrieved for any location!")
        return
    
    # Concatenate all DataFrames
    print(f"\nConcatenating {len(all_dataframes)} location datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"✓ Combined dataset: {len(combined_df)} rows")
    
    # Apply data cleaning
    print("\nApplying data cleaning...")
    combined_df = clean_weather_dataframe(combined_df)
    print(f"✓ After cleaning: {len(combined_df)} rows")
    
    # Apply feature engineering
    print("\nApplying feature engineering...")
    combined_df = engineer_features(combined_df)
    print(f"✓ Features engineered")
    
    # Add crop yield column (realistic agricultural model)
    print("\nGenerating crop yield data based on climate factors...")
    
    def calculate_crop_yield(row):
        """
        Calculate realistic crop yield (tons/hectare) based on climate factors.
        Optimal conditions: temp 25-30°C, humidity 60-75%, rainfall 500-750mm/season, wind 0-15 km/h
        """
        np.random.seed(hash(row['date']) % 2**32)  # Reproducible randomness per date
        
        temp = row['temperature']
        humidity = row['humidity']
        rainfall = row['rainfall_mm']
        wind = row['wind_speed']
        
        # Base yield (tons/hectare)
        base_yield = 3.5
        
        # Temperature factor (optimal 25-30°C)
        temp_factor = 1.0
        if 25 <= temp <= 30:
            temp_factor = 1.0  # Optimal
        elif 20 <= temp < 25:
            temp_factor = 0.9 + 0.02 * (temp - 20)
        elif 30 < temp <= 35:
            temp_factor = 1.0 - 0.04 * (temp - 30)
        else:
            temp_factor = max(0.3, 1.0 - 0.1 * abs(temp - 27.5))  # Penalty for extreme temps
        
        # Humidity factor (optimal 60-75%)
        humidity_factor = 1.0
        if 60 <= humidity <= 75:
            humidity_factor = 1.0  # Optimal
        elif 50 <= humidity < 60:
            humidity_factor = 0.85 + 0.015 * (humidity - 50)
        elif 75 < humidity <= 85:
            humidity_factor = 1.0 - 0.02 * (humidity - 75)
        else:
            humidity_factor = max(0.4, 1.0 - 0.02 * abs(humidity - 67.5))
        
        # Rainfall factor (accumulated, 500-750mm optimal for season)
        # Assume this is daily rainfall
        rainfall_factor = 1.0
        if rainfall <= 5:
            rainfall_factor = 0.7 + 0.06 * rainfall
        elif 5 < rainfall <= 15:
            rainfall_factor = 1.0
        elif 15 < rainfall <= 50:
            rainfall_factor = 1.0 - 0.005 * (rainfall - 15)
        else:
            rainfall_factor = max(0.4, 1.0 - 0.01 * rainfall)
        
        # Wind factor (stress at high speeds)
        wind_factor = 1.0
        if wind <= 15:
            wind_factor = 1.0  # Optimal
        elif 15 < wind <= 30:
            wind_factor = 1.0 - 0.02 * (wind - 15)
        else:
            wind_factor = max(0.5, 1.0 - 0.05 * wind)
        
        # Combined yield with realistic noise
        combined_factor = temp_factor * humidity_factor * rainfall_factor * wind_factor
        yield_value = base_yield * combined_factor
        
        # Add realistic random noise (±15%)
        noise = np.random.normal(1.0, 0.15)
        yield_value = max(0.5, yield_value * noise)
        
        return round(yield_value, 2)
    
    combined_df['crop_yield'] = combined_df.apply(calculate_crop_yield, axis=1)
    print(f"✓ Crop yield generated")
    
    # Compute climate risk based on weather factors
    def compute_climate_risk(row):
        """
        Calculate climate risk score (0.0-1.0) based on temperature, humidity, rainfall, and wind.
        Returns risk level: 'Low', 'Medium', or 'High'.
        """
        score = 0.0
        
        # Temperature risk
        temp = row['temperature']
        if temp > 35:
            score += 0.28
        elif temp > 30:
            score += 0.18
        else:
            score += 0.07
        
        # Humidity risk
        humidity = row['humidity']
        if humidity > 78:
            score += 0.28
        elif humidity > 68:
            score += 0.18
        else:
            score += 0.07
        
        # Rainfall risk
        rainfall_mm = row['rainfall_mm']
        if 25 <= rainfall_mm <= 110:
            score += 0.14
        elif rainfall_mm > 110:
            score += 0.04
        else:
            score += 0.09
        
        # Wind speed risk
        wind_speed = row['wind_speed']
        if wind_speed < 12:
            score += 0.09
        else:
            score += 0.03
        
        # Add random noise
        score += np.random.normal(0, 0.055)
        
        # Clip to valid range [0.0, 1.0]
        score = np.clip(score, 0.0, 1.0)
        
        # Determine risk level
        if score < 0.40:
            return 'Low'
        elif score <= 0.70:
            return 'Medium'
        else:
            return 'High'
    
    print("\nComputing climate risk scores...")
    combined_df['climate_risk'] = combined_df.apply(compute_climate_risk, axis=1)
    print(f"✓ Climate risk computed")
    
    # Print climate risk distribution
    print(f"\nClimate Risk Distribution:")
    print(combined_df['climate_risk'].value_counts())
    
    # Create data directory if it doesn't exist
    print("\nSaving dataset...")
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/climate_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"✓ Dataset saved to {output_path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"\nTotal rows saved: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    print(f"\nDate range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    print(f"\nLocation distribution:")
    print(combined_df['location'].value_counts())
    
    print(f"\nCrop Yield Statistics (tons/hectare):")
    print(combined_df['crop_yield'].describe())
    
    print(f"\nWeather Statistics:")
    print(combined_df[['temperature', 'humidity', 'rainfall_mm', 'wind_speed']].describe())
    
    print("\n" + "=" * 70)
    print("Dataset generation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    generate_climate_dataset()
