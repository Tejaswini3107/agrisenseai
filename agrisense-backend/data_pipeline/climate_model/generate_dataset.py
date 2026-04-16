import pandas as pd
import os
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
                '20200101',
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
    
    # Add climate risk column
    print("\nAdding climate risk classification...")
    
    def classify_climate_risk(row):
        """Classify climate risk based on temperature and humidity."""
        temperature = row['temperature']
        humidity = row['humidity']
        
        # High risk: temperature > 35 or humidity > 78
        if temperature > 35 or humidity > 78:
            return 'high'
        # Medium risk: (temperature 28-35) or (humidity 60-78)
        elif (temperature >= 28 and temperature <= 35) or (humidity >= 60 and humidity <= 78):
            return 'medium'
        # Low risk: otherwise
        else:
            return 'low'
    
    combined_df['climate_risk'] = combined_df.apply(classify_climate_risk, axis=1)
    print(f"✓ Climate risk classification complete")
    
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
    
    print(f"\nClimate Risk Distribution:")
    risk_dist = combined_df['climate_risk'].value_counts()
    print(risk_dist)
    print(f"\nClimate Risk Percentages:")
    risk_pct = (combined_df['climate_risk'].value_counts(normalize=True) * 100).round(2)
    for risk, pct in risk_pct.items():
        print(f"  {risk}: {pct}%")
    
    print(f"\nWeather Statistics:")
    print(combined_df[['temperature', 'humidity', 'rainfall_mm', 'wind_speed']].describe())
    
    print("\n" + "=" * 70)
    print("Dataset generation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    generate_climate_dataset()
