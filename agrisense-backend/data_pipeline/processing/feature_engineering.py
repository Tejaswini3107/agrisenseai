import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Engineer features for agricultural modeling.
    
    Args:
        df: DataFrame containing temperature, humidity, rainfall_mm, wind_speed
        
    Returns:
        DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Heat Index
    df['heat_index'] = (
        -8.78469 
        + (1.61139 * df['temperature']) 
        + (2.33855 * df['humidity'])
        - (0.14611 * df['temperature'] * df['humidity'])
        - (0.01230 * df['temperature'] ** 2)
        - (0.01642 * df['humidity'] ** 2)
    )
    df['heat_index'] = df['heat_index'].clip(lower=10, upper=80)
    
    # Dew Point
    df['dew_point'] = df['temperature'] - ((100 - df['humidity']) / 5.0)
    
    # Vapor Pressure Deficit
    df['vapor_pressure_deficit'] = (
        0.6108 * np.exp(17.27 * df['temperature'] / (df['temperature'] + 237.3)) 
        * (1 - df['humidity'] / 100.0)
    )
    df['vapor_pressure_deficit'] = df['vapor_pressure_deficit'].round(4)
    
    # Rainfall Category
    def categorize_rainfall(rainfall):
        if rainfall == 0:
            return 'none'
        elif rainfall <= 10:
            return 'light'
        elif rainfall <= 50:
            return 'moderate'
        else:
            return 'heavy'
    
    df['rainfall_category'] = df['rainfall_mm'].apply(categorize_rainfall)
    
    # High Humidity Flag
    df['is_high_humidity'] = (df['humidity'] > 70).astype(int)
    
    # High Temperature Flag
    df['is_high_temp'] = (df['temperature'] > 32).astype(int)
    
    return df
