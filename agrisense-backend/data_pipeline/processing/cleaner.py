import pandas as pd
import numpy as np


def clean_weather_dataframe(df):
    """
    Clean and validate weather data in a pandas DataFrame.

    Args:
        df: DataFrame with columns: temperature, humidity, rainfall_mm, wind_speed

    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values in critical columns,
                      values clipped to valid ranges, and index reset.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Step 1: Fill missing temperature values with the column median
    df["temperature"] = df["temperature"].fillna(df["temperature"].median())

    # Step 2: Fill missing humidity values with the column median
    df["humidity"] = df["humidity"].fillna(df["humidity"].median())

    # Step 3: Fill missing rainfall_mm values with 0 (no rain is the safe default)
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0)

    # Step 4: Fill missing wind_speed values with the column median
    df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].median())

    # Step 5: Clip temperature to the range -5 to 55 (removes physically impossible values)
    df["temperature"] = df["temperature"].clip(lower=-5, upper=55)

    # Step 6: Clip humidity to the range 0 to 100
    df["humidity"] = df["humidity"].clip(lower=0, upper=100)

    # Step 7: Clip rainfall_mm to the range 0 to 500
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0, upper=500)

    # Step 8: Clip wind_speed to the range 0 to 150
    df["wind_speed"] = df["wind_speed"].clip(lower=0, upper=150)

    # Step 9: Drop any rows where temperature, humidity are still NaN after filling
    df = df.dropna(subset=["temperature", "humidity"])

    # Return the cleaned DataFrame with the index reset
    return df.reset_index(drop=True)
