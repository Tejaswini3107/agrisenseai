import numpy as np
import joblib
from tensorflow import keras

def forecast_weather(days_history):
    """
    Predict weather for the next 3 days given 7 days of history.
    
    Args:
        days_history: Array of shape (7, 4) with past 7 days of:
                      [temperature, humidity, rainfall_mm, wind_speed]
    
    Returns:
        Dictionary with 3-day forecast for each feature
    """
    print("=" * 70)
    print("Weather Forecasting Using Trained LSTM")
    print("=" * 70)
    
    # Load model and scalers
    print("\n1. Loading trained model and scalers...")
    try:
        model = keras.models.load_model('models/lstm_weather_model.keras')
        scalers = joblib.load('data/sequences/scalers.pkl')
        weather_features = joblib.load('data/sequences/weather_features.pkl')
        print("✓ Model and scalers loaded")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run: python -m data_pipeline.climate_model.train_lstm")
        return None
    
    # Normalize input
    print("\n2. Normalizing input data...")
    X_scaled = np.zeros_like(days_history, dtype=np.float32)
    
    for i, feature in enumerate(weather_features):
        scaler = scalers[feature]
        X_scaled[:, i] = scaler.transform(days_history[:, i].reshape(-1, 1)).flatten()
    
    # Add batch dimension
    X_input = np.expand_dims(X_scaled, axis=0)  # Shape: (1, 7, 4)
    print(f"✓ Input shape: {X_input.shape}")
    
    # Make prediction
    print("\n3. Making 3-day forecast...")
    y_pred_scaled = model.predict(X_input, verbose=0)  # Shape: (1, 3, 4)
    y_pred_scaled = y_pred_scaled[0]  # Remove batch dimension
    
    # Inverse transform
    print("\n4. Inverse transforming predictions...")
    y_pred = np.zeros_like(y_pred_scaled)
    
    for i, feature in enumerate(weather_features):
        scaler = scalers[feature]
        y_pred[:, i] = scaler.inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()
    
    # Format output
    print("\n" + "=" * 70)
    print("3-Day Weather Forecast")
    print("=" * 70)
    
    forecast_dict = {}
    for day in range(3):
        day_num = day + 1
        print(f"\n📅 Day {day_num}:")
        day_forecast = {}
        
        for feature_idx, feature in enumerate(weather_features):
            value = y_pred[day, feature_idx]
            
            # Format based on feature
            if feature == 'temperature':
                formatted = f"{value:.2f}°C"
            elif feature == 'humidity':
                formatted = f"{value:.1f}%"
            elif feature == 'rainfall_mm':
                formatted = f"{value:.2f}mm"
            elif feature == 'wind_speed':
                formatted = f"{value:.2f}km/h"
            else:
                formatted = f"{value:.4f}"
            
            day_forecast[feature] = value
            print(f"   🌡️  {feature.replace('_', ' ').title()}: {formatted}")
        
        forecast_dict[f'day_{day_num}'] = day_forecast
    
    print("\n" + "=" * 70)
    
    return forecast_dict


if __name__ == "__main__":
    # Example: Create synthetic 7-day history
    print("Example: Making a forecast with synthetic data")
    
    # 7 days of weather history
    example_history = np.array([
        [25.0, 65.0, 2.5, 8.0],   # Day -6
        [26.0, 62.0, 0.0, 9.0],   # Day -5
        [27.0, 60.0, 0.0, 7.5],   # Day -4
        [26.5, 63.0, 1.2, 8.5],   # Day -3
        [28.0, 58.0, 0.0, 9.5],   # Day -2
        [29.0, 55.0, 0.0, 10.0],  # Day -1
        [28.5, 62.0, 3.0, 8.0]    # Today (Day 0)
    ])
    
    forecast = forecast_weather(example_history)
